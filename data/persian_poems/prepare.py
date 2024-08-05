import os
import tiktoken
import numpy as np
import pandas as pd

for poet in Path('./data/poem/Persian-poems/database/').glob('*.txt'):
    with open(poet, 'r') as file:
        data = file.readlines() + '\n'

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
gpt2_base = tiktoken.get_encoding("gpt2")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="gpt2_im",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={
        **gpt2_base._special_tokens,
        '<[EOT]>': 50257,
        '<[EON]>':50258, 
        '<[EOFMMBT]>': 50259,
        '<[EOSMMBT]>': 50260,
        '<[EOSM]>': 50261,
        '<[EOFM]>': 50262,
        '<[EOT]>': 50263,
        '<[SOT]>': 50264,
        '<[SON]>': 50265,
        '<[SOFMMBT]>': 50266,
        '<[SOSMMBT]>': 50267,
        '<[SOSM]>': 50268,
        '<[SOFM]>': 50269, 
        '<[SOT]>': 50270,
        '[CLS]': 50271
    }
)

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
