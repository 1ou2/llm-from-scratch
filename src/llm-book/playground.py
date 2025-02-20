import torch
import tiktoken
import torch.nn as nn

# Example with context_len = 4
context_len = 4

# Create a matrix of ones
ones = torch.ones(context_len, context_len)
print("Matrix of ones:")
print(ones)

# Apply torch.triu with diagonal=1
mask = torch.triu(ones, diagonal=1)
print("\nUpper triangular mask (diagonal=1):")
print(mask)

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
batch = torch.randn(2,5)
print(f"{batch=}")
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch)
print(f"{out=}")
print(f"{out.shape=}")
mean = out.mean(dim=-1,keepdim=True)
var = out.var(dim=-1, keepdim=True)
print(f"{mean=}")
print(f"{var=}")