"""
Tokenize Wikipedia dataset and serialize it to disk
"""
import tiktoken
import os
import multiprocess as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

# Custom tokenizer for french language
use_custom_tokenizer = True

if use_custom_tokenizer:
    tokenizer = ByteLevelBPETokenizer(
        "french_tokenizer/gabgpt-vocab.json",
        "french_tokenizer/gabgpt-merges.txt"
    )
    # we do not add special tokens because we cannot trust the source dataset
    DATA_TOKENIZED_DIR = "data/tokenized/gabwikifr/"
    eot = tokenizer.token_to_id("<|endoftext|>")
else:
    tokenizer = tiktoken.get_encoding("gpt2")
    DATA_TOKENIZED_DIR = "data/tokenized/wikipedia_fr/"
    eot = tokenizer._special_tokens["<|endoftext|>"]

print(f"eot: {eot}")

# Load Wikipedia dataset in streaming mode (no full memory load)
dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")

DATA_TOKENIZED_DIR = "data/tokenized/gabwikifr/"
os.makedirs(DATA_TOKENIZED_DIR, exist_ok=True)

shard_size = int(1e6)  # 1 million tokens per shard

def tokenize(article):
    # all documents start with end of sequence token
    tokens = [eot]
    if use_custom_tokenizer:
        tokens.extend(tokenizer.encode(article["text"]).ids)
    else:
        # disallow special tokens in the article
        tokens.extend(tokenizer.encode_ordinary(article["text"]))
    # convert to a uint16 numpy array - 2 bits per token
    return np.array(tokens, dtype=np.uint16)

nprocs = max(1, mp.cpu_count() - 1)
print(f"Using {nprocs} processes")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # pre-allocate memory for tokens in a shard
    all_tokens_np = np.empty((shard_size,),dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, dataset["train"].shuffle(),chunksize=16):
        if token_count + len(tokens) < shard_size:
            # add tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"Tokenizing shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            filename = os.path.join(DATA_TOKENIZED_DIR, f"shard_{shard_index:06d}.npy")
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
        filename = os.path.join(DATA_TOKENIZED_DIR, f"shard_{shard_index:06d}.npy")
        np.save(filename, all_tokens_np[:token_count])