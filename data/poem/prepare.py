import os
import tiktoken
import numpy as np
import pandas as pd

df = pd.read_csv('./data/poem/poem.txt')
data = df['body'].str.cat(sep='\n')

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
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
        "<|startofstanza|>": 50259,
        "<|startofverse|>": 50260,
        "<|endofverse|>": 50261,

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
