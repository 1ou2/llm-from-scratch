"""The verdict is a novel in the public domain"""

import urllib.request
import re
import tiktoken
from importlib.metadata import version
import torch
from torch.utils.data import Dataset, DataLoader

def download():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"

    file_path = "./data/raw/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def explore():
    with open("./data/raw/the-verdict.txt", "r",encoding="utf-8") as f:
        text = f.read()
        print(f"Len :{len(text)}")
        print(text[:1000])

def tiktok():
    print("titotken version",version("tiktoken"))
    tokenizer = tiktoken.get_encoding("gpt2")

    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
            "of someunknownPlace.")
    print(f"{text=}")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"{integers=}")
    strings = tokenizer.decode(integers)
    print(f"{strings=}")

    with open("./data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
        enc_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        print(f"{len(enc_text)=}")
        enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"{x=}")
    print(f"     {y=}")

    for i in range(1,context_size+1):
        context = enc_sample[:i]
        target = enc_sample[i]
        print(f"{tokenizer.decode(context)} --> {tokenizer.decode([target])}")


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        """Create a Sample dataset
        text : input text
        tokenizer : tokenizer object that converts string to tokens
        max_length : number of tokens to use as input
        stride : offset used when iterating over the inputs tokens """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # encode the text with tiktoken
        token_ids = tokenizer.encode(text)

        # split the text into chunks of max_length with stride stride
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        self.enc_text = token_ids


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=8, max_length=8,stride=8, 
                         shuffle=True,drop_last=True, num_workers=0):
    """Create a dataloader from a text file
    text : the content of the file
    batch_size : how many elements per batch
    max_length : how many tokens per inputs (the context length)
    stride : how we iterate on the tokens
    shuffle: should we shuffle the inputs
    drop_last : drop last batch if we don't have enough tokens
    num_workers : number of parallel processes to launch
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length,stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader

def load_data():
    with open("./data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
        dataloader = create_dataloader_v1(text, batch_size=3, max_length=5,
                                          stride=2, shuffle=False)
        tokenizer = tiktoken.get_encoding("gpt2")

        for i, batch in enumerate(dataloader):
            x, y = batch
            print(f"{i=} {x.shape=} {y.shape=}")
            print(f"{x=}")
            print(f"{y=}")
            if i == 3:
                break

def embed():
    vocab_size = 50256 # GPT-3 vocab size
    output_dim = 256 # embedding size
    context_length = 4 # context length
    # create a random embedding matrix
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # load the model
    with open("./data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    dataloader = create_dataloader_v1(text, batch_size=8, max_length=4,
                                          stride=4, shuffle=False)

    x,y = next(iter(dataloader))
    print(f"{x.shape=}")
    token_embedding = token_embedding_layer(x)
    print(f"{token_embedding.shape=}")

    # position embedding
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embddings = pos_embedding_layer(torch.arange(context_length))
    print(f"{pos_embddings.shape=}")

    input_embedding = token_embedding + pos_embddings
    print(f"{input_embedding.shape=}")

if __name__ == "__main__":
    #explore()
    #tiktok()
    #load_data()
    embed()
