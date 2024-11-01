import torch
from tokenizers import Tokenizer

import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
    
from data import DatasetFactory

def training_data():
    gutenberg_dataset = DatasetFactory.create_dataset("gutenberg")
    gutenberg_dataset.load()
    x_batches, y_batches = gutenberg_dataset.batch_data(type="train", batch_size=10, nb_batches=20)
    print(len(x_batches), len(y_batches))
    print(y_batches)
    for y in y_batches:
        print([y])

    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    for i,x in enumerate(x_batches):
        print(f"{tokenizer.decode(x)} --> {tokenizer.decode([y_batches[i]])}")
        
    return x_batches, y_batches


def train_model():

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


if __name__ == "__main__":
    x_batches, y_batches = training_data()
    train_model()


