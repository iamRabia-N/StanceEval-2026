import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate", "peft"], check=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import re, gc, zipfile, random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
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
llmft_old = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'test_llmft_probs.npy':
            llmft_old = np.load(os.path.join(root, f))
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(root, f), nrows=2)
                if 'tweet_text' in df.columns:
                    test = pd.read_csv(os.path.join(root, f))
            except Exception:
                pass
assert test is not None and llmft_old is not None

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

model_dirs = []
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files and 'fin_' in root and root.endswith('final'):
        model_dirs.append(root)
model_dirs = sorted(set(model_dirs))
print('final models found:', len(model_dirs))

def predict_probs_dir(model_dir, df):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).half().cuda().eval()
    outs = []
    for i in range(0, len(df), 64):
        sub = df.iloc[i:i + 64]
        enc = tok(list(sub['target']), list(sub['text_clean']), truncation=True, max_length=128, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outs.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(outs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

bert_ens = np.mean([predict_probs_dir(md, test) for md in model_dirs], axis=0)

pl = llmft_old.argmax(-1)
agree = (pl == bert_ens.argmax(-1)) & (pl != 2)
print('agreement pseudo labels:', int(agree.sum()), dict(zip(*np.unique(pl[agree], return_counts=True))))
pseudo = pd.DataFrame({'target': 'Women Driving', 'text_clean': test['text_clean'].values[agree], 'label': pl[agree]})
aug = pd.concat([full_df, pseudo, pseudo], ignore_index=True)
print('augmented size:', aug.shape)

MODEL7 = 'Qwen/Qwen2.5-7B-Instruct'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
ltok = AutoTokenizer.from_pretrained(MODEL7)
ltok.padding_side = 'left'
if ltok.pad_token is None:
    ltok.pad_token = ltok.eos_token

def make_prompt(target, text):
    return f'Classify the stance of this Arabic tweet toward the target "{target}". Answer with exactly one word: Favor, Against, or None.\n\nTweet: {str(text)[:300]}\nAnswer:'

class SFTDS(Dataset):
    def __init__(self, df):
        self.rows = df.reset_index(drop=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        prompt = ltok.apply_chat_template([{'role': 'user', 'content': make_prompt(r['target'], r['text_clean'])}], tokenize=False, add_generation_prompt=True)
        answer = I2L[r['label']] + ltok.eos_token
        p_ids = ltok(prompt, truncation=True, max_length=300, add_special_tokens=False)['input_ids']
        a_ids = ltok(answer, add_special_tokens=False)['input_ids']
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        pad = 320 - len(ids)
        if pad > 0:
            ids = [ltok.pad_token_id] * pad + ids
            labels = [-100] * pad + labels
            attn = [0] * pad + [1] * (320 - pad)
        else:
            ids = ids[-320:]
            labels = labels[-320:]
            attn = [1] * 320
        return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(attn)}

base = AutoModelForCausalLM.from_pretrained(MODEL7, quantization_config=bnb, device_map={'': 0})
base = prepare_model_for_kbit_training(base)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
lora_model = get_peft_model(base, lcfg)
lora_model.print_trainable_parameters()

args = TrainingArguments(output_dir='/kaggle/working/qlora2', num_train_epochs=2, learning_rate=2e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4, warmup_ratio=0.05, logging_steps=100, save_strategy='no', report_to='none', fp16=True, seed=SEED, optim='paged_adamw_8bit')
trainer = Trainer(model=lora_model, args=args, train_dataset=SFTDS(aug))
trainer.train()
lora_model.save_pretrained('/kaggle/working/qlora2_adapter')

lora_model.config.use_cache = True
lora_model.eval()
preds = []
BS = 8
for i in range(0, len(test), BS):
    rows = test.iloc[i:i + BS]
    msgs = [ltok.apply_chat_template([{'role': 'user', 'content': make_prompt(r['target'], r['text_clean'])}], tokenize=False, add_generation_prompt=True) for _, r in rows.iterrows()]
    enc = ltok(msgs, return_tensors='pt', padding=True, truncation=True, max_length=320).to('cuda')
    with torch.no_grad():
        out = lora_model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=ltok.eos_token_id)
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
ft2 = np.full((len(test), 3), 0.05)
for i, p in enumerate(preds):
    ft2[i, p] = 0.9
np.save('/kaggle/working/test_llmft2_probs.npy', ft2)

variants = {}
variants['x1_ft_selftrained'] = ft2.argmax(-1)
x2 = (ft2 + llmft_old) / 2
variants['x2_ft_both_voted'] = x2.argmax(-1)

for name, prds in variants.items():
    counts = dict(zip(*np.unique(prds, return_counts=True)))
    print(name, 'label counts', {I2L[kk]: int(v) for kk, v in counts.items()})
    with open(f'/kaggle/working/predictions_{name}.txt', 'w') as f:
        for pp in prds:
            f.write(I2L[pp] + '\n')
    with zipfile.ZipFile(f'/kaggle/working/submission_{name}.zip', 'w') as z:
        z.write(f'/kaggle/working/predictions_{name}.txt', 'predictions.txt')
