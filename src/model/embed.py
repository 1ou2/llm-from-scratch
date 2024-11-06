import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
    
from data import DatasetFactory

def training_data(block_size,nb_batches):
    gutenberg_dataset = DatasetFactory.create_dataset("gutenberg")
    gutenberg_dataset.load()
    gutenberg_dataset.batch_data( block_size=block_size, nb_batches=nb_batches,type="mini",)

    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    for i,x in enumerate(gutenberg_dataset.x_batch):
        print(f"{tokenizer.decode(x)} --> {tokenizer.decode([gutenberg_dataset.y_batch[i]])}")
        print(f"{x=} ++> {gutenberg_dataset.y_batch[i]}")

    return gutenberg_dataset.x_batch, gutenberg_dataset.y_batch

def train_model(use_gpu=True):
    if use_gpu: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print("No GPU found, using CPU instead.")
    else:
        device = torch.device("cpu")

    
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == 'cuda':
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print("Running on Apple Silicon")

    
    g = torch.Generator(device=device).manual_seed(2147483647)
    # vocabulary size -> total number of tokens
    vocab_size = 1024
    # how many independent sequences will we process in parallel?
    #batch_size = 4
    # what is the maximum context length for predictions, aka number of input tokens
    block_size = 8
    # how many training steps to perform?
    max_iters = 5000
    # size of our embedding vector space
    embed_size = 16
    # number of hidden layers
    n_hidden = 64
    # number of batches to process in parallel
    nb_batches = 4

    # create embedding 
    C = torch.randn((vocab_size, embed_size), generator=g,device=device) # torch.Size([1024, 16])
    # Hidden layer
    W1 = torch.randn((embed_size * block_size, n_hidden), generator=g,device=device)
    b1 = torch.randn(n_hidden, generator=g,device=device)
    # output layer 
    W2 = torch.randn((n_hidden, vocab_size), generator=g,device=device)
    b2 = torch.randn(vocab_size, generator=g,device=device)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    print(f"{C.shape=}")
    print(f"{W1.shape=}")
    print(f"{b1.shape=}")
    print(f"{W2.shape=}")
    print(f"{b2.shape=}")

    X, Y = training_data(block_size, nb_batches)
    X = torch.tensor(X,device=device) # [32,8] -- [nb_batches , block_size]
    Y = torch.tensor(Y,device=device) # [32] -- [nb_batches]
    print(f"{X.shape=}")
    print(f"{Y.shape=}")
    print(f"{C.shape=}")
    # we have a batch of 32 elements
    # the block_size is 10 - we have 10 tokens of context
    # the embedding vector space size is 20
    emb = C[X] # [4,8,16]
    print(f"{emb.shape=}")
    print(emb.view(-1, block_size * embed_size).shape) # [4,128]

    h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1) # [4,64]
    logits = h @ W2 + b2 # [4,1024]
    #counts = logits.exp() # [4,1024]
    #probs = counts / counts.sum(1, keepdim=True) # [4,1024]
    #loss = -probs[torch.arange(4), Y].log().mean() # [4,1024] --> [4] --> [1]
    #print(f"{loss=}")
    loss = F.cross_entropy(logits, Y)
    print(f"{loss=}")

    # backward pass
    for p in parameters:
        p.grad = None

    loss.backward()

if __name__ == "__main__":
    use_gpu = True  # Set this to False to use CPU
    train_model(use_gpu)


