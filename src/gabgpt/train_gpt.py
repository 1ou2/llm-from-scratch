
from model_gpt import GPTModel
from dataloader import DataLoaderLite
import torch
import tiktoken
import torch.nn.functional as F
import time

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

HYPERS = {
    "lr": 6e-4,
    "warmup_steps": 1500,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "tokens_per_shard": int(1e6)
}

tokenizer = tiktoken.get_encoding("gpt2")

def generate_text_completion(model, text):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # model expects an input in batch format
    idx = torch.tensor(encoded).unsqueeze(0).to(device)

    #for _ in range(GPT_CONFIG["context_length"]-len(idx)):
    for _ in range(50):
        model.eval()
        with torch.no_grad():
            # logits shape : (batch_size, sequence_length, vocab_size)
            # usually batch_size is 1 for text generation
            logits = model(idx)
            # Get logits for the last token of the sequence
            # shape : (batch_size, vocab_size)
            logits = logits[:, -1, :]
            # shape : (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx.squeeze().tolist())

# -------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

B = 8
T = GPT_CONFIG["context_length"]

val_loader = DataLoaderLite(B, T, "valid")
train_loader = DataLoaderLite(B, T, "train")


# number of batches in one epoch
nb_train_steps = int(HYPERS["tokens_per_shard"] / (B * T)) * len(train_loader.shards)
nb_val_steps = int(len(val_loader.shards) * HYPERS["tokens_per_shard"] / (B * T))
print(f"nb_train_steps: {nb_train_steps}")
print(f"nb_val_steps: {nb_val_steps}")

model = GPTModel(GPT_CONFIG)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERS["lr"], betas=(HYPERS["beta1"], HYPERS["beta2"]), eps=HYPERS["eps"], weight_decay=HYPERS["weight_decay"])

t0 = time.time()
for step in range(nb_train_steps):
    if step > 50:
        break
    model.train()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, GPT_CONFIG["vocab_size"]), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"step: {step}, loss: {loss.item()}")
        text = generate_text_completion(model, "Je suis")
        print(f"text: ///{text}///")
t1 = time.time()
print(f"Time: {t1-t0}")
print(f"Tokens: {B}*{T}*{step} = {B*T*step}")
print(f"Tokens/s: {(B*T*step)/(t1-t0)}")