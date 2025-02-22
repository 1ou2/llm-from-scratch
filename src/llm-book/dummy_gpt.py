import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

GPT_CONFIG_1 = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["embed_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = DummyLayerNorm(config["embed_dim"])
        self.out_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)
        

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def dummy_gpt2():
        # print options : use 2 digits only
    torch.set_printoptions(precision=2, sci_mode=False)

    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    print(f"{batch=}")
    batch = torch.stack(batch,dim=0)
    print(f"Stack:\n{batch}")

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_1)
    logits = model(batch)
    print(f"output shape: {logits.shape}")
    print(f"output:\n{logits}")