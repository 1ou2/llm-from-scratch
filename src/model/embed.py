import torch

import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from data import DatasetFactory

gutenberg_dataset = DatasetFactory.create_dataset("gutenberg")
train_files, val_files, test_files = gutenberg_dataset.split_files()

g = torch.Generator().manual_seed(2147483647)
# vocabulary size -> number of tokens
vocab_size = 1000
# how many independent sequences will we process in parallel?
batch_size = 4
# what is the maximum context length for predictions?
block_size = 8
# how many training steps to perform?
max_iters = 5000
# size of our embedding vector space
embed_size = 30
# number of hidden layers
n_hidden = 64

# create embedding 
C = torch.randn((vocab_size, embed_size), generator=g)
# Hidden layer
W1 = torch.randn((embed_size * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
# output layer 
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# 
#from ..data.hg_tokenizer import Tokenizer

#train_files, val_files, test_files = split_files()
#print(len(train_files), len(val_files), len(test_files))



