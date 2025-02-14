import torch
import tiktoken

def self_attention():
    embedding_dim = 3
    input_sentence = "Alice a mangé une pizza et elle était délicieuse"

    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(input_sentence)
    print(f"{tokens=}")

    # embedding layer
    #embedding_layer = torch.nn.Embedding(len(tokenizer), embedding_dim)
    #print(f"{embedding_layer.weight.shape=}")

if __name__ == "__main__":
    self_attention()