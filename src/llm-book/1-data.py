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


def create_dataloader_v1(text, batch_size=4, max_length=256,stride=128, 
                         shuffle=True,drop_last=True, num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length,stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    #explore()
    tiktok()
