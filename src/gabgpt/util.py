# ----------------------------------------------------------------------
# Load configuration from file
# ----------------------------------------------------------------------
import configparser
import ast
import os
import math

def load_config(config_file="config.txt"):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Convert string values to appropriate Python types
    def parse_value(value):
        try:
            # Try to evaluate as literal (for boolean, None, etc)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it fails, return as string
            return value

    # Create configuration dictionaries
    GPT_CONFIG = {
        key: parse_value(value)
        for key, value in config['model'].items()
    }
    
    HYPERS = {
        key: parse_value(value)
        for key, value in config['hypers'].items()
    }
    
    #
    FILES = {
        key: parse_value(value)
        for key, value in config['files'].items()
    }

    TRAINING = {
        key: parse_value(value)
        for key, value in config['training'].items()
    }
    
    return GPT_CONFIG, HYPERS, FILES, TRAINING

class LogPrinter:
    def __init__(self, log_file):
        # check if parent directory exists
        parent_dir = os.path.dirname(log_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.log_file = open(log_file,"a")

    def log_print(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def get_lr(step, epoch_steps):
    """Get learning rate"""
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warm_up = epoch_steps // 20 # 5%
    warm_up = 715
    epoch_steps = 19073
    # slow learning rate after one epoch
    if step > epoch_steps:
        return min_lr
    
    # go lineary from min_lr to max_lr
    if step < warm_up:
        return min_lr + (max_lr - min_lr) * step / warm_up
    
    # go from max_lr to min_lr using a cosine function to smooth out the learning rate
    decay_ratio = (step - warm_up) / (epoch_steps - warm_up)
    coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coefficient

epoch_steps = 19073
x = []
y = []
for step in range(epoch_steps + 1):
    if step % 10 == 0:
        x.append(step)
        y.append(get_lr(step, epoch_steps))
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.savefig("lr-gpt2.png")
# print first 20 values of y
print(y[:20])


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr2(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

epoch_steps = 19073
x = []
y = []
for step in range(epoch_steps + 1):
    if step % 100 == 0:
        x.append(step)
        y.append(get_lr2(step))

plt.plot(x, y)
plt.savefig("lr2-gpt2.png")