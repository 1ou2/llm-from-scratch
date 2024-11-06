import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import time
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
    
from data import DatasetFactory

def training_data(block_size,batch_size):
    gutenberg_dataset = DatasetFactory.create_dataset("gutenberg")
    gutenberg_dataset.load()
    gutenberg_dataset.batch_data( block_size=block_size, nb_batches=batch_size,type="mini",)

    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    for i,x in enumerate(gutenberg_dataset.x_batch):
        print(f"{tokenizer.decode(x)} --> {tokenizer.decode([gutenberg_dataset.y_batch[i]])}")
        print(f"{x=} ++> {gutenberg_dataset.y_batch[i]}")

    return gutenberg_dataset.x_batch, gutenberg_dataset.y_batch

def get_device(use_gpu=True):
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
    return device

def train_model(use_gpu=True, use_checkpoint=False):
    device = get_device(use_gpu)
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
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
    embed_size = 32
    # number of hidden layers
    n_hidden = 128
    # number of batches to process in parallel
    batch_size = 4

    if use_checkpoint:
        # load checkpoint
        # load the model
        C, W1, b1, W2, b2 = torch.load("data/model.pt", map_location=device)
    else:
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

    gutenberg = DatasetFactory.create_dataset("gutenberg")
    gutenberg.load()
    #gutenberg.batch_data( block_size=block_size, nb_batches=batch_size,type="mini",)
    gutenberg.batch_data(block_size=block_size, type="train")
    print("gutenberg size ", len(gutenberg.y_batch))


    #for i,x in enumerate(gutenberg_dataset.x_batch):
    #    print(f"{tokenizer.decode(x)} --> {tokenizer.decode([gutenberg_dataset.y_batch[i]])}")
    #    print(f"{x=} ++> {gutenberg_dataset.y_batch[i]}")
    #return gutenberg_dataset.x_batch, gutenberg_dataset.y_batch
    X = torch.tensor(gutenberg.x_batch,device=device)# [32,8] -- [batch_size , block_size]
    Y = torch.tensor(gutenberg.y_batch,device=device) # [32] -- [batch_size]
    print(f"{X.shape=}")
    print(f"{Y.shape=}")
    print(f"{C.shape=}")

    nb_iter = len(Y) // batch_size
    lr = 0.1
    
    # measure time
    #optimizer = torch.optim.Adam(parameters, lr=0.1)
    start = time.time()
    for i in range(nb_iter):
        #optimizer.zero_grad(set_to_none=True)
        # forward pass
        # get embedding for the ith row in C[X]
        emb = C[X[i*batch_size:(i+1)*batch_size]]
        #print(f"{emb.shape=}")
        #print(emb.view(-1, block_size * embed_size).shape) # [4,128]

        h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1) # [4,64]
        logits = h @ W2 + b2 # [4,1024]
        loss = F.cross_entropy(logits, Y[i*batch_size:(i+1)*batch_size])
        if i % 500 == 0:
            print(f"{i} | {lr=} | {loss=}")
        # backward pass
        for p in parameters:
            p.grad = None

        loss.backward()
        #optimizer.step()
        # update
        lr = 0.1
        if i > 20000:
            lr = 0.01
        if i > 80000:
            lr = 0.001
        for p in parameters:
            p.data += -lr * p.grad
    
    end = time.time()
    print(f"{i} | {loss=}")
    print(f"Time: {end - start}")
    print(f"Time per iteration: {(end - start) / nb_iter}")

    # save the model
    torch.save([C, W1, b1, W2, b2], "data/model.pt")

def inference(use_gpu,block_size):
    device = get_device(use_gpu)
    # load the model
    C, W1, b1, W2, b2 = torch.load("data/model.pt", map_location=device)
    # sample from the model
    g = torch.Generator(device=device).manual_seed(2147483647)
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    for i in range(10):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context], device=device)]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0 or len(out) > 30:
                break
        print(f"{tokenizer.decode(out)}")

if __name__ == "__main__":
    use_gpu = True  # Set this to False to use CPU
    train_model(use_gpu,use_checkpoint=True)
    inference(use_gpu, block_size=8)


