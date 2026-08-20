import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate", "peft"], check=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import re, zipfile, random, numpy as np, pandas as pd, torch
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
arab = None
champ = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'test_arab_probs.npy':
            arab = np.load(os.path.join(root, f))
        if f == 'test_llmft2_probs.npy':
            champ = np.load(os.path.join(root, f))
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(root, f), nrows=2)
                if 'tweet_text' in df.columns:
                    test = pd.read_csv(os.path.join(root, f))
            except Exception:
                pass
assert test is not None and arab is not None and champ is not None

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
    df['sarc'] = df['sarcasm'].fillna('No').map(lambda x: 'Yes' if str(x).strip() == 'Yes' else 'No')
    df['text_clean'] = df['text'].apply(normalize)
    return df[['target', 'text_clean', 'label', 'sarc']]

full_df = pd.concat([prep(paths['train.csv']), prep(paths['dev.csv'])], ignore_index=True)
test['text_clean'] = test['tweet_text'].apply(normalize)
print('training size:', full_df.shape, '| sarcastic:', int((full_df['sarc'] == 'Yes').sum()))

MODEL7 = 'Qwen/Qwen2.5-7B-Instruct'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
ltok = AutoTokenizer.from_pretrained(MODEL7)
ltok.padding_side = 'left'
if ltok.pad_token is None:
    ltok.pad_token = ltok.eos_token

def make_prompt(target, text):
    return f'Analyze this Arabic tweet about the target "{target}". First decide if the tweet is sarcastic, then decide the stance of the author toward the target. Answer in exactly this format: Sarcasm: Yes or No. Stance: Favor or Against or None.\n\nTweet: {str(text)[:300]}\nAnswer:'

class SFTDS(Dataset):
    def __init__(self, df):
        self.rows = df.reset_index(drop=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        prompt = ltok.apply_chat_template([{'role': 'user', 'content': make_prompt(r['target'], r['text_clean'])}], tokenize=False, add_generation_prompt=True)
        answer = f'Sarcasm: {r["sarc"]}. Stance: {I2L[r["label"]]}' + ltok.eos_token
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

base = AutoModelForCausalLM.from_pretrained(MODEL7, quantization_config=bnb, device_map={'': 0})
base = prepare_model_for_kbit_training(base)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
lora_model = get_peft_model(base, lcfg)

args = TrainingArguments(output_dir='/kaggle/working/sarc', num_train_epochs=3, learning_rate=2e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4, warmup_ratio=0.05, logging_steps=100, save_strategy='no', report_to='none', fp16=True, seed=SEED, optim='paged_adamw_8bit')
trainer = Trainer(model=lora_model, args=args, train_dataset=SFTDS(full_df))
trainer.train()
lora_model.save_pretrained('/kaggle/working/sarc_adapter')

lora_model.config.use_cache = True
lora_model.eval()

preds = []
sarcs = []
BS = 8
for i in range(0, len(test), BS):
    rows = test.iloc[i:i + BS]
    msgs = [ltok.apply_chat_template([{'role': 'user', 'content': make_prompt(r['target'], r['text_clean'])}], tokenize=False, add_generation_prompt=True) for _, r in rows.iterrows()]
    enc = ltok(msgs, return_tensors='pt', padding=True, truncation=True, max_length=340).to('cuda')
    with torch.no_grad():
        out = lora_model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=ltok.eos_token_id)
    for t in ltok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True):
        m = re.search(r'stance:\s*(favor|against|none)', t, re.IGNORECASE)
        s = re.search(r'sarcasm:\s*(yes|no)', t, re.IGNORECASE)
        sarcs.append(s.group(1).lower() if s else 'no')
        if m:
            w = m.group(1).lower()
            preds.append(0 if w == 'favor' else 1 if w == 'against' else 2)
        else:
            preds.append(-1)

preds = np.array(preds)
print('unparsed:', int((preds == -1).sum()))
preds[preds == -1] = 0
print('predicted sarcastic on test:', sarcs.count('yes'), 'of', len(test))
sarc_probs = np.full((len(test), 3), 0.05)
for i, p in enumerate(preds):
    sarc_probs[i, p] = 0.9
np.save('/kaggle/working/test_sarc_probs.npy', sarc_probs)
counts = dict(zip(*np.unique(preds, return_counts=True)))
print('label counts:', {I2L[k]: int(v) for k, v in counts.items()})

def make_sub(name, probs):
    prds = probs.argmax(-1)
    with open(f'/kaggle/working/predictions_{name}.txt', 'w') as f:
        for pp in prds:
            f.write(I2L[pp] + '\n')
    with zipfile.ZipFile(f'/kaggle/working/submission_{name}.zip', 'w') as z:
        z.write(f'/kaggle/working/predictions_{name}.txt', 'predictions.txt')

make_sub('g1_sarcaware', sarc_probs)
make_sub('g2_sarc_trio', (sarc_probs + arab + champ) / 3)
print('agreement with qwen:', round(float((sarc_probs.argmax(-1) == champ.argmax(-1)).mean()), 3))
print('agreement with allam:', round(float((sarc_probs.argmax(-1) == arab.argmax(-1)).mean()), 3))
