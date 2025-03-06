""" French reddit dataset
https://www.kaggle.com/datasets/breandan/french-reddit-discussion/data
"""

import time 
import lxml.etree as ET
import pandas as pd
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import multiprocess as mp
import os
import numpy as np
from tqdm import tqdm

file_path="data/raw/spf.xml"

parser = ET.XMLParser(recover=True)
#Parses the file
tree = ET.parse(file_path, parser=parser)
xroot = tree.getroot()
dataset = []
start = time.time()
for node in xroot:
    for j in range(len(node.getchildren())):
        text = node.getchildren()[j].text
        dataset.append(text)
end = time.time()
print("time: ",end - start)
print(len(dataset))

tokenizer_dir = "data/tokenizer"

# Define special tokens
special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
print(f"Using custom tokenizer from {tokenizer_dir}")
vocab_file = os.path.join(tokenizer_dir, "gabgpt-vocab.json")
merges_file = os.path.join(tokenizer_dir, "gabgpt-merges.txt")
print(f"vocab file: {vocab_file}")
print(f"merges file: {merges_file}")
# Load the trained tokenizer
tokenizer = ByteLevelBPETokenizer(
    vocab_file,
    merges_file
)
eot = tokenizer.token_to_id("<|endoftext|>")
print(f"eot: {eot}")

def tokenize(article):
        # all documents start with end of sequence token
        tokens = [eot]
        tokens.extend(tokenizer.encode(article).ids)

        # convert to a uint16 numpy array - 2 bits per token
        return np.array(tokens, dtype=np.uint16)

nprocs = max(1, mp.cpu_count() - 1)
print(f"Using {nprocs} processes")
shard_size=1048576
output_dir = "data/tokenized/reddit"
os.makedirs(output_dir, exist_ok=True)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # pre-allocate memory for tokens in a shard
    all_tokens_np = np.empty((shard_size,),dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, dataset,chunksize=16):
        if token_count + len(tokens) < shard_size:
            # add tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"Tokenizing shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            filename = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
            # split the document into whatever fits in the shard
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            # save the shard
            np.save(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the shard with the remaining tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    
    # write the last shard
    if token_count > 0:
        filename = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
        np.save(filename, all_tokens_np[:token_count])