import os
import sys
import numpy as np

DATA_TOKENIZED_DIR = "data/tokenized/wikipedia_fr/"

def load_tokens(filename):
    nptokens = np.load(filename)
    nptokens = nptokens.astype(np.uint16)
    return nptokens

# get all npy files in the directory
files = sorted([os.path.join(DATA_TOKENIZED_DIR, f) for f in os.listdir(DATA_TOKENIZED_DIR) if f.endswith(".npy")])
print(f"Found {len(files)} files")

for file in files:
    #print(f"File: {file} - nb tokens : {len(load_tokens(file))}")
    # get filename without extension
    filename = os.path.splitext(os.path.basename(file))[0]
    # filename is shard_{index:06d}.npy
    index = int(filename.split("_")[1])
    print(f"File: {file} -  - index : {index}")


