import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embed = 32
vocab_size = 1024
head_size = 16

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(1337)

class CharacterTokenizer():
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return "".join([self.itos[i] for i in l])
    
class DataManager():
    def __init__(self,data) -> None:
        self.data = data
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.batch_size = 4
        self.block_size = 8

    def get_batch(self,split):
        """return two tensors x (batch of data of inputs) and y (the ground truth)
        split : train or valid
        """
        data = self.train_data if split == "train" else self.val_data
        # we want batch_size indexes, that are in the range (len(data) - block_size)
        # e.g. : if len(data) = 2048, block_size = 8, and batch_size =4
        # then, we will get 4 random indexes, that are between 0 and 2040.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        # get  block_size consecutives tokens from the random index, and stack them in x
        # they become a row
        x = torch.stack([data[i:i+self.block_size] for i in ix]) # [4,8]
        # do the same for the target
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix]) #[4,8]
        x, y = x.to(device), y.to(device)
        return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        torch.manual_seed(1337)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # v1
        #self.sa_head = Head(n_embed)

        # v2
        #self.sa_heads = MultiHeadAttention(4, n_embed//4) # i.e 4 heads of 8 dimensional self attention
        #self.ffwd = FeedForward(n_embed)
        
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # (B,T,C) = (4,8,65)
        # B : batch -> nombre d’inputs en parallèle
        # T : time -> longueur max du contexte
        # C : channels -> dimension de l’embedding
        tok_emb = self.token_embedding_table(idx) # (B,T,C) = (4,8,65)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) = (8, 65)
        x = tok_emb + pos_emb # (B, T, C) = (4, 8, 65)
        #x = self.sa_heads(x) # apply one head of self attention # B,T,C
        #x = self.ffwd(x) # B,T,C
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size) = (4, 8, 65)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # before using cross entropy function we need to reshape the logits
            # see cross entropy doc for more details
            logits = logits.view(B*T, C) # (B*T, C) = (32, 65)
            targets = targets.view(B*T) # (B*T) = (32)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ idx is (B, T) array of indices in the current context
        The goal is to extend the sequence in the time dimension for each batch
        input (B,T) -> (B, T+1) -> (B, T+2) -> (B, T+3)
        """
        #torch.manual_seed(1337)
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # get last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # layer norm 1
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed) # second layer norm

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.LayerNorm(n_embed),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # projection layer
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # project the layer so that we can add it to the residual connection neuron
        out = self.proj(out)
        return out
    

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_text():
    with open("data/raw/shakespeare.txt", "r") as f:
        text = f.read()
    return text

def get_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    #print("".join(chars))
    #print(vocab_size)
    return chars


def lecture():
    text = get_text()
    chars = get_vocab(text)
    tokenizer = CharacterTokenizer(chars)
    str = "hii there"
    print(tokenizer.encode(str))
    print(tokenizer.decode(tokenizer.encode(str)))


    tensor_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"dataset {tensor_data.shape}, {tensor_data.dtype}")
    data = DataManager(tensor_data)

    xb,yb = data.get_batch("train")
    print(f"input size {xb.shape}")
    print(xb)
    print(f"output size {yb.shape}")
    print(yb)    

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b,:t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target is {target}")

def head():
    torch.manual_seed(1337)
    B,T,C = 4,8,32 # batch, time, channels
    x = torch.randn(B,T,C)
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x) # (B, T, 16)
    q = query(x) # (B, T, 16)
    v = value(x) # (B, T, 16)
    #print(k)
    print(f"{k.shape=}")
    print(f"{k.transpose(-2, -1).shape=}") # transpose(B, T, 16) ->  (B, 16, T) = (4, 16, 8)
    wei = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, 16) @ (B, 16, T) = (B, T, T) = (4, 8, 8)
    print(f"{wei.shape=}")
    print(wei[-1])
    
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    print(wei[-1])
    wei = F.softmax(wei, dim=-1) # (B, T, T) = (4, 8, 8)
    print(f"{wei.shape=}")
    print(wei[-1])
    sys.exit(0)
    torch.set_printoptions(precision=4,sci_mode=False)
    print(f"{v[-1]}")
    out = wei @ v # (B, T, T) @ (B, T, 16) -> (B,T,16) = (4,8,16)
    print(f"{out.shape=}")
    print(out[-1])
    

if __name__ == "__main__":

    text = get_text()
    chars = get_vocab(text)
    vocab_size=len(chars)
    tokenizer = CharacterTokenizer(chars)
    block_size = 8
    batch_size = 4
    #torch.manual_seed(1337)
    
    tensor_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"dataset {tensor_data.shape}, {tensor_data.dtype}")
    data = DataManager(tensor_data)

    xb,yb = data.get_batch("train")

    model = BigramLanguageModel()
    m = model.to(device)
    out, loss = m(xb,yb)
    print(f"{out.shape=}")
    print(f"{loss=}")
    idx =  torch.zeros((1, 1), dtype=torch.long)
    tokens = m.generate(idx, max_new_tokens=100)
    print(f"{tokens=}")
    tokens = tokens[0].tolist() # flatten to a python list
    print(tokenizer.decode(tokens))
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % 200 == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = data.get_batch("train")

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    tokens = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()
    print(tokenizer.decode(tokens))