
from model_gpt import GPTModel
from dataloader import DataLoaderLite
from stats_training import TrainingStats
from transformers import get_linear_schedule_with_warmup
import torch
import tiktoken
import torch.nn.functional as F
import time
import os

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
    "epochs": 1,
    "lr": 6e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "tokens_per_shard": int(1e6)
}



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

def save_checkpoint(model, optimizer, scheduler, train_loader, config, epoch, step, loss, stats, save_dir):
    """
    Save the model, optimizer, scheduler, and epoch to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/checkpoint_{epoch}_{step}_{loss:.2f}.pt"
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'train_loader_state': train_loader.get_state(),
        "stats_file": stats.save_stats(save_dir)
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state,stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # If using GPU, move optimizer states to GPU
        if device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        raise ValueError("Optimizer state dict not found in checkpoint. Unable to load optimizer.")
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        raise ValueError("Scheduler state dict not found in checkpoint. Unable to load scheduler.")
    
    
    # Load statistics if they exist
    stats = TrainingStats()
    if 'stats_file' in checkpoint and os.path.exists(checkpoint['stats_file']):
        stats.load_stats(checkpoint['stats_file'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss'], checkpoint['train_loader_state'], stats

def get_last_checkpoint(save_dir):
    """
    Get the last checkpoint file in the given directory.
    Returns the path to the last checkpoint file.
    """
    checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith("checkpoint_")])

    return os.path.join(save_dir, checkpoints[-1])

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
start_epoch = 0
start_step = 0
checkpoint = "checkpoints/checkpoint_0_3000_5.50.pt"
useCheckpoint = True
stats = TrainingStats()

val_loader = DataLoaderLite(B, T, "valid")
train_loader = DataLoaderLite(B, T, "train")
tokenizer = tiktoken.get_encoding("gpt2")

# number of batches in one epoch
nb_train_steps = int(HYPERS["tokens_per_shard"] / (B * T)) * len(train_loader.shards)
nb_val_steps = int(len(val_loader.shards) * HYPERS["tokens_per_shard"] / (B * T))
print(f"nb_train_steps: {nb_train_steps}")
print(f"nb_val_steps: {nb_val_steps}")

total_steps = nb_train_steps * HYPERS["epochs"]
warmup_steps = total_steps // 10 # use 10% for warmup

model = GPTModel(GPT_CONFIG)
model.to(device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERS["lr"], betas=(HYPERS["beta1"], HYPERS["beta2"]), eps=HYPERS["eps"], weight_decay=HYPERS["weight_decay"])
optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERS["lr"])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

if useCheckpoint:
    checkpoint = get_last_checkpoint("checkpoints")
    if checkpoint is None:
        raise ValueError("No checkpoint found")
    start_epoch, start_step, loss, train_state, stats = load_checkpoint(checkpoint, model, optimizer, scheduler,device=device)
    shard_index = train_state["shard_index"]
    print(f"Loaded checkpoint: epoch {start_epoch}, step {start_step}, loss {loss} - shard: {shard_index}")
    train_loader.set_state(train_state)

t0 = time.time()
for epoch in range(start_epoch,HYPERS["epochs"]):
    for step in range(start_step, nb_train_steps):
        if step > 6000:
            break
        if step > start_step + 1000:
            break
        model.train()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, GPT_CONFIG["vocab_size"]), y.view(-1))
        loss.backward()  # backward pass
        optimizer.step() # update weights
        scheduler.step() # update learning rate
        optimizer.zero_grad() # reset gradients

        
        if step % 50 == 0:
            print(f"step: {step}, loss: {loss.item()}")
            stats.update(
                step=step,
                loss=loss.item(),
                lr=scheduler.get_last_lr()[0],
                shard_index=train_loader.current_shard_index
            )
        if step % 500 == 0:
            text = generate_text_completion(model, "Je suis")
            print(f"text: ///{text}///")
            stats.update(step=step, generated_text=text)
        if step % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, train_loader, GPT_CONFIG, epoch, step, loss.item(), stats,"checkpoints")
        
t1 = time.time()
print(f"Time: {t1-t0}")
print(f"Tokens: {B}*{T}*{step} = {B*T*step}")
print(f"Tokens/s: {(B*T*step)/(t1-t0)}")
print(f"Loss: {loss.item()}")
print(f"Shard index {train_loader.current_shard_index}")