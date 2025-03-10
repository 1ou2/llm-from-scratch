import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# ----------------------------------------------------------------------
# GPT Model
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # shortcut connection will be added
        # we will scale the standard deviation intialization by 1 / sqrt(n_head)
        # this is basically a flag for the weights_init function
        self.c_proj.SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 32768 # number of tokens: custom tokenizer used
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # standard deviation
        # ~ 0.03 for n_embed = 768
        std = self.config.n_embd ** -0.5
        if isinstance(module, nn.Linear):
            # because of shortcut connection, the gradient will increase more for some layers            
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
        return optimizer
    
# ----------------------------------------------------------------------
# Data Loader
# ----------------------------------------------------------------------


import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split, token_dir, process_rank=0, num_processes=1):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.split = split
        self.shards = []
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.token_dir = token_dir
        self.update_shard_list()
        self.reset()

    def update_shard_list(self):
        self.shards = sorted([os.path.join(self.token_dir, f) for f in os.listdir(self.token_dir) if f.endswith(".npy")])

        if self.split == "train":
            # remove the last shard
            if len(self.shards) > 1:
                # last shard may not be full
                self.shards.pop()
        if master_process:
            print(f"found {len(self.shards)} shards for split {self.split}")

    def get_state(self):
        return {
            "shard_index": self.current_shard_index,
            "token_index": self.current_token_index,
        }

    def set_state(self, state):
        self.reset()
        self.current_shard_index = state["shard_index"]
        self.current_token_index = state["token_index"]

    def reset(self):
        self.current_shard_index = 0
        # each process has a different offset in the shard
        # so that they don't overlap
        self.current_token_index = self.B * self.T * self.process_rank
        self.tokens = load_tokens(self.shards[self.current_shard_index])

    def next_batch(self):
        """Returns 2 batches of tokens of shape (B, T) - input batch and target batch"""
        # get B*T tokens + 1 because we need to predict the next token
        buffer = self.tokens[self.current_token_index: self.current_token_index + self.B * self.T+1]
        # get all tokens except the last one
        x = (buffer[:-1]).view(self.B, self.T)
        # target tokens are the ones that follow the input tokens
        # shift the tokens by 1 to the left
        y = (buffer[1:]).view(self.B, self.T)

        # advance index
        self.current_token_index += self.B * self.T * self.num_processes
        # check if we need to load the next shard
        if self.current_token_index + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            # cycle through the shards, enables to continue get batches for more than one epoch
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            # each process has a different offset in the shard
            # so that they don't overlap
            self.current_token_index = self.B * self.T * self.process_rank
        
        return x, y

# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------

def save_checkpoint(model, optimizer, train_loader,epoch, step, loss, save_dir):
    """
    Save the model, optimizer, scheduler, and epoch to a file.
    only keep last checkpoints (TRAINING["n_checkpoint_files"])
    """
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/checkpoint_{epoch}_{step:07d}_{loss:.2f}.pt"
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_loader_state': train_loader.get_state()
    }
    torch.save(checkpoint, path)

    # remove old checkpoints
    checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")])
    if len(checkpoints) > TRAINING["n_checkpoint_files"]:
        for f in checkpoints[:TRAINING["n_checkpoint_files"]]:
            os.remove(f"{save_dir}/{f}")

def load_parameters(path,device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state, stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path, map_location=device,weights_only=True)
    return checkpoint['train_loader_state'], checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def load_checkpoint(path, model, optimizer, device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state,stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    # Create a new state dict removing '_orig_mod' prefix from checkpoint
    # when using torch.compile with aot_eager backend, the keys have a prefix '_orig_mod.module.'
    # we need, to remov this prefix to be able to load the model otherwise we have a mismatch
    fixed_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.module.'):
            new_key = key.replace('_orig_mod.module.', '')
            fixed_state_dict[new_key] = value
        elif key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value

    # Compare the shapes of tensors
    for key in fixed_state_dict:
        if key in model.state_dict():
            ckpt_shape = fixed_state_dict[key].shape
            model_shape = model.state_dict()[key].shape
            if ckpt_shape != model_shape:
                print(f"Shape mismatch for {key}: checkpoint {ckpt_shape} vs model {model_shape}")
        else:
            print(f"Key {key} not found in model state dict")

    model.load_state_dict(fixed_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def get_last_checkpoint(save_dir):
    """
    Get the last checkpoint file in the given directory.
    Returns the path to the last checkpoint file.
    """
    # check if directory exists
    if not os.path.exists(save_dir):
        return None

    checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith("checkpoint_")])
    if len(checkpoints) == 0:
        return None
    return os.path.join(save_dir, checkpoints[-1])


def generate_text_completion(model, text):
    encoded = tokenizer.encode(text).ids

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

# ----------------------------------------------------------------------
# simple launch:
# python multigpu_train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 multigpu_train.py
# ----------------------------------------------------------------------
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?


if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
device_name = device
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)  # 0 is the GPU index

torch.manual_seed(4321) # seed for CPU
random.seed(4321) # seed for Python random module
if torch.cuda.is_available():
    torch.cuda.manual_seed(4321 + ddp_rank) # seed for GPU

from util import load_config, LogPrinter
GPT_CONFIG, HYPERS, FILES, TRAINING = load_config()
logger = LogPrinter(FILES["log_dir"] + FILES["log_file"])
stats_file = open(FILES["log_dir"] + FILES["stat_file"], "a")

# Write header for stats file (if it's empty)
if stats_file.tell() == 0:
    stats_file.write("epoch,step,loss,learning_rate,validation\n")

from datetime import datetime

# Get current date/time and format it
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


logger.log_print(f"--------------------------------------------")
logger.log_print(f"Training started at {current_time}")
logger.log_print(f"GPU: {device_name}")

B = TRAINING["micro_batch_size"]
T = GPT_CONFIG["context_length"]


nb_shards = TRAINING["n_shards"]
# each step we process a fixed sized batch
batch_size = TRAINING["batch_size"]
# this batch size will probably not fit in memory
# we will use gradient accumulation to simulate this batch
grad_accum_steps = batch_size // (B * T * ddp_world_size)
# we are sample +1 token for y
# tokens_per_shard = 1048577 = 2^20+1
epoch_train_steps = ((TRAINING["tokens_per_shard"]-1) // (batch_size)) * nb_shards
warmup_steps = epoch_train_steps // 20 # 5% of one epoch

# Big optimization here !
# change format from float32 to tf32 (tensor float32)
# this causes a little loss on precision but 8x througput on GPU processing and x3 in memory
# -> x3 performance increase
torch.set_float32_matmul_precision('high')


# load tokenizer
from tokenizers import ByteLevelBPETokenizer
special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

# Load the trained tokenizer
tokenizer = ByteLevelBPETokenizer(
    FILES["tokenizer_dir"] + "gabgpt-vocab.json",
    FILES["tokenizer_dir"] + "gabgpt-merges.txt"
)

# Add special tokens to the loaded tokenizer
tokenizer.add_special_tokens(special_tokens)

# Token loader
valid_token_dir = FILES["token_dir"] + "valid/"
train_token_dir = FILES["token_dir"] + "train/"

from dataloader import IndexedDataLoader
val_loader = IndexedDataLoader(B, T, split="valid", nb_shards=nb_shards, token_dir=valid_token_dir, process_rank=ddp_rank, num_processes=ddp_world_size)
train_loader = IndexedDataLoader(B, T, split="train", nb_shards=nb_shards, token_dir=train_token_dir, process_rank=ddp_rank, num_processes=ddp_world_size)


# custom learning rate function
from util import get_lr

# 1. create model
model = GPT(GPTConfig())
# 2. move to the correct GPU
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# 3. load checkpoint
checkpoint = get_last_checkpoint(FILES["checkpoint_dir"])
if checkpoint is None:
    start_epoch = 0
    start_step = 0
    shard_index = 0
    # fused = True -> performance improvement
    learning_rate = HYPERS["lr"]
else:
    train_loader_state, start_epoch, start_step, saved_loss = load_parameters(checkpoint)
    # the step saved was finished, so we need to increment it
    start_step += 1
    learning_rate = get_lr(start_step,start_epoch)
    train_loader.set_state(train_loader_state,fill_processed=True)
    shard_index = train_loader.get_shard_index()
    if master_process:
        logger.log_print(f"Checkpoint loaded : {checkpoint}")
        logger.log_print(f"| epoch: {start_epoch} | step: {start_step} | shard: {shard_index} | loss: {saved_loss} |")

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device_type=device_type)
if checkpoint is not None:
    load_checkpoint(checkpoint, model, optimizer, device=device)



# 4. compile
use_compile = TRAINING["use_compile"] 
if use_compile:
    model = torch.compile(model)


if master_process:
    logger.log_print(f"epoch: {start_epoch} | step: {start_step} | shard: {shard_index}")
    logger.log_print(f"total steps: {epoch_train_steps * HYPERS['epochs']} | grad accum steps: {grad_accum_steps} | warmup_steps: {warmup_steps}")
    logger.log_print(f"B: {B} | T: {T} | W: {ddp_world_size} : {B * T * ddp_world_size} tokens | Batch per shards: {(TRAINING["tokens_per_shard"]-1)// TRAINING["batch_size"]}")

assert start_step < epoch_train_steps, f"start_step {start_step} >= nb_train_steps {epoch_train_steps}"
assert TRAINING["checkpoint_interval"] % ((TRAINING["tokens_per_shard"]-1)// TRAINING["batch_size"]) == 0, f"checkpoint_interval {TRAINING['checkpoint_interval']} not divisible by number of batch per shards"
loss_accum = 0.0
step = start_step
# Training Loop
start_time = time.time()
t0 = time.time()
for epoch in range(start_epoch,HYPERS["epochs"]):
    for step in range(start_step, epoch_train_steps):
        if step > start_step + TRAINING["max_steps"]:
            break

        if (step > start_step) and master_process and step % TRAINING["log_interval"] == 0:
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_processed = B * T * ddp_world_size * TRAINING["log_interval"]
            tokens_per_sec = tokens_processed / dt

            logger.log_print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {get_lr(step,epoch):.7f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            
            # Write stats in CSV format for easy parsing
            stats_msg = f"{epoch},{step},{loss_accum.item()},{get_lr(step,epoch)},0\n"
            stats_file.write(stats_msg)
            stats_file.flush()
            t0 = time.time()

        if (step > start_step) and step % TRAINING["gen_text_interval"] == 0:
            model.eval()
            num_return_sequences = TRAINING["num_generation_sequences"]
            max_length = 32
            tokens = tokenizer.encode("Je suis").ids
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = tokenizer.decode(tokens)
                logger.log_print(f"rank {ddp_rank} sample {i}: {decoded}")

        if (step > start_step) and step % TRAINING["eval_interval"] == 0 and step > 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = TRAINING["validation_steps"]
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                logger.log_print(f"step {step:5d} | validation loss: {val_loss_accum.item():.4f} |")
                # Write stats in CSV format for easy parsing
                stats_msg = f"{epoch},{step},{val_loss_accum.item()},{get_lr(step,epoch)},1\n"
                stats_file.write(stats_msg)
                stats_file.flush()

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, epoch_train_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work

        if (step > start_step) and master_process and (step +1) %  TRAINING["checkpoint_interval"]  == 0:
            save_checkpoint(model, optimizer, train_loader, epoch, step, loss_accum.item(), FILES["checkpoint_dir"])

# save final checkpoint
if master_process:
    end_time = time.time()
    wrapup_message = f"""
    
    #######
    Time: {end_time - start_time }
    Processed Tokens: {B}*{T}*{step - start_step} * {ddp_world_size}= {B*T*(step - start_step)*ddp_world_size}
    Tokens/s: {(B*T*(step - start_step)*ddp_world_size)/(end_time - start_time)}
    Loss: {loss_accum.item()}
    Total Tokens: {B}*{T}*{step} = {B*T*step}
    Shard index: {train_loader.get_index()}
    """
    logger.log_print(wrapup_message)
    logger.close()
    stats_file.close()
if ddp:
    destroy_process_group()


        