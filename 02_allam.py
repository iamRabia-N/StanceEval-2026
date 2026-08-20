import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate", "peft"], check=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import re, gc, zipfile, random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABELS = ['Favor', 'Against', 'None']
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

test = None
judges = {}
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f in ('test_llmft_probs.npy', 'test_llmft2_probs.npy', 'test_ft14_probs.npy'):
            judges[f] = np.load(os.path.join(root, f))
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(root, f), nrows=2)
                if 'tweet_text' in df.columns:
                    test = pd.read_csv(os.path.join(root, f))
            except Exception:
                pass
assert len(judges) == 3 and test is not None

paths = {}
for root, dirs, files in os.walk('/kaggle/input'):
    for f in ['train.csv', 'dev.csv']:
        if f in files:
            paths[f] = os.path.join(root, f)

DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0640]')

def normalize(text):
    t = str(text)
    t = re.sub(r'https?://\S+|www\.\S+', ' رابط ', t)
    t = re.sub(r'@\w+', ' مستخدم ', t)
    t = t.replace('#', ' ').replace('_', ' ')
    t = DIAC.sub('', t)
    t = re.sub(r'[إأآا]', 'ا', t)
    t = re.sub(r'ى', 'ي', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def prep(path):
    df = pd.read_csv(path)
    df['stance'] = df['stance'].fillna('None')
    df['label'] = df['stance'].map(L2I)
    df['text_clean'] = df['text'].apply(normalize)
    return df[['target', 'text_clean', 'label']]

full_df = pd.concat([prep(paths['train.csv']), prep(paths['dev.csv'])], ignore_index=True)
test['text_clean'] = test['tweet_text'].apply(normalize)

j = [judges[k].argmax(-1) for k in sorted(judges.keys())]
votes = np.stack(j, axis=1)
pl = np.zeros(len(test), dtype=int)
support = np.zeros(len(test), dtype=int)
for i in range(len(test)):
    vals, cnts = np.unique(votes[i], return_counts=True)
    k = int(np.argmax(cnts))
    pl[i] = int(vals[k])
    support[i] = int(cnts[k])
mask = (support >= 2) & (pl != 2)
print('consensus pseudo labels:', int(mask.sum()))
pseudo = pd.DataFrame({'target': 'Women Driving', 'text_clean': test['text_clean'].values[mask], 'label': pl[mask]})
aug = pd.concat([full_df, pseudo, pseudo], ignore_index=True)
print('training size:', aug.shape)

CANDIDATES = ['ALLaM-AI/ALLaM-7B-Instruct-preview', 'silma-ai/SILMA-9B-Instruct-v1.0']
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')

ltok = None
base = None
MODEL_USED = None
for cand in CANDIDATES:
    try:
        ltok = AutoTokenizer.from_pretrained(cand)
        base = AutoModelForCausalLM.from_pretrained(cand, quantization_config=bnb, device_map={'': 0})
        MODEL_USED = cand
        break
    except Exception as e:
        print(cand, 'failed:', repr(e))
        ltok = None
        base = None
        gc.collect()
        torch.cuda.empty_cache()
assert base is not None, 'no candidate model loaded'
print('using model:', MODEL_USED)

ltok.padding_side = 'left'
if ltok.pad_token is None:
    ltok.pad_token = ltok.eos_token

def make_prompt(target, text):
    return f'Classify the stance of this Arabic tweet toward the target "{target}". Answer with exactly one word: Favor, Against, or None.\n\nTweet: {str(text)[:300]}\nAnswer:'

def wrap(content):
    try:
        return ltok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    except Exception:
        return content + ' '

class SFTDS(Dataset):
    def __init__(self, df):
        self.rows = df.reset_index(drop=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        prompt = wrap(make_prompt(r['target'], r['text_clean']))
        answer = I2L[r['label']] + (ltok.eos_token or '')
        p_ids = ltok(prompt, truncation=True, max_length=300, add_special_tokens=False)['input_ids']
        a_ids = ltok(answer, add_special_tokens=False)['input_ids']
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        pad = 340 - len(ids)
        if pad > 0:
            ids = [ltok.pad_token_id] * pad + ids
            labels = [-100] * pad + labels
            attn = [0] * pad + [1] * (340 - pad)
        else:
            ids = ids[-340:]
            labels = labels[-340:]
            attn = [1] * 340
        return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(attn)}

base = prepare_model_for_kbit_training(base)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
lora_model = get_peft_model(base, lcfg)

args = TrainingArguments(output_dir='/kaggle/working/arab', num_train_epochs=2, learning_rate=2e-4, per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_ratio=0.05, logging_steps=100, save_strategy='no', report_to='none', fp16=True, seed=SEED, optim='paged_adamw_8bit')
trainer = Trainer(model=lora_model, args=args, train_dataset=SFTDS(aug))
trainer.train()
lora_model.save_pretrained('/kaggle/working/arab_adapter')

lora_model.config.use_cache = True
lora_model.eval()

label_first_ids = [ltok(l, add_special_tokens=False)['input_ids'][0] for l in LABELS]
distinct = len(set(label_first_ids)) == 3
print('label first ids:', label_first_ids, 'distinct:', distinct)

def score_logprob():
    all_probs = []
    BS = 4
    for i in range(0, len(test), BS):
        rows = test.iloc[i:i + BS]
        msgs = [wrap(make_prompt(r['target'], r['text_clean'])) for _, r in rows.iterrows()]
        enc = ltok(msgs, return_tensors='pt', padding=True, truncation=True, max_length=340).to('cuda')
        with torch.no_grad():
            logits = lora_model(**enc).logits[:, -1, :]
        sel = logits[:, label_first_ids].float()
        all_probs.append(torch.softmax(sel, dim=-1).cpu().numpy())
    return np.concatenate(all_probs)

def score_generate():
    preds = []
    BS = 4
    for i in range(0, len(test), BS):
        rows = test.iloc[i:i + BS]
        msgs = [wrap(make_prompt(r['target'], r['text_clean'])) for _, r in rows.iterrows()]
        enc = ltok(msgs, return_tensors='pt', padding=True, truncation=True, max_length=340).to('cuda')
        with torch.no_grad():
            out = lora_model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=ltok.pad_token_id)
        for t in ltok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True):
            t = t.strip().lower()
            if t.startswith('favor'):
                preds.append(0)
            elif t.startswith('against'):
                preds.append(1)
            elif t.startswith('none'):
                preds.append(2)
            else:
                preds.append(-1)
    preds = np.array(preds)
    print('unparsed:', int((preds == -1).sum()))
    preds[preds == -1] = 0
    arr = np.full((len(test), 3), 0.05)
    for i, p in enumerate(preds):
        arr[i, p] = 0.9
    return arr

arab_probs = score_logprob() if distinct else score_generate()
np.save('/kaggle/working/test_arab_probs.npy', arab_probs)
counts = dict(zip(*np.unique(arab_probs.argmax(-1), return_counts=True)))
print('arabic model label counts', {I2L[k]: int(v) for k, v in counts.items()})

with open('/kaggle/working/predictions_a1_arabic.txt', 'w') as f:
    for p in arab_probs.argmax(-1):
        f.write(I2L[p] + '\n')
with zipfile.ZipFile('/kaggle/working/submission_a1_arabic.zip', 'w') as z:
    z.write('/kaggle/working/predictions_a1_arabic.txt', 'predictions.txt')

champ = judges['test_llmft2_probs.npy']
avg = (arab_probs + champ) / 2
with open('/kaggle/working/predictions_a2_arab_champ.txt', 'w') as f:
    for p in avg.argmax(-1):
        f.write(I2L[p] + '\n')
with zipfile.ZipFile('/kaggle/working/submission_a2_arab_champ.zip', 'w') as z:
    z.write('/kaggle/working/predictions_a2_arab_champ.txt', 'predictions.txt')
print('agreement with champion:', round(float((arab_probs.argmax(-1) == champ.argmax(-1)).mean()), 3))
