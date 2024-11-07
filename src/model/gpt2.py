import os
import torch
import torch.nn as nn
from torch.nn import functional as F

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
       
        return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size) -> None:
        super().__init__()
        torch.manual_seed(1337)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # (B,T,C) = (4,8,65)
        # B : batch -> nombre d’inputs en parallèle
        # T : time -> longueur max du contexte
        # C : channels -> dimension de l’embedding
        logits = self.token_embedding_table(idx) # (B,T,C) = (4,8,65)
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

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

if __name__ == "__main__":
    text = get_text()
    chars = get_vocab(text)
    tokenizer = CharacterTokenizer(chars)
    block_size = 8
    batch_size = 4
    #torch.manual_seed(1337)
    
    tensor_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"dataset {tensor_data.shape}, {tensor_data.dtype}")
    data = DataManager(tensor_data)

    xb,yb = data.get_batch("train")

    m = BigramLanguageModel(len(chars))
    out, loss = m(xb,yb)
    print(f"{out.shape=}")
    print(f"{loss=}")
    idx =  torch.zeros((1, 1), dtype=torch.long)
    tokens = m.generate(idx, max_new_tokens=100)
    print(f"{tokens=}")
    tokens = tokens[0].tolist() # flatten to a python list
    print(tokenizer.decode(tokens))


