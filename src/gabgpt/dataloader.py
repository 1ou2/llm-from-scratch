
import os
import numpy as np
import torch

def load_tokens(filename):
    """Load tokens saved as a numpy array.
    Retuns a torch tensor of type long"""
    nptokens = np.load(filename)
    nptokens = nptokens.astype(np.uint16)
    return torch.tensor(nptokens,dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split, process_rank =0, num_processes=1):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.shards = []
        self.process_rank = process_rank
        self.num_processes = num_processes
        DATA_TOKENIZED_DIR = "data/tokenized/wikipedia_fr/"
        shards = sorted([os.path.join(DATA_TOKENIZED_DIR, f) for f in os.listdir(DATA_TOKENIZED_DIR) if f.endswith(".npy")])
        assert(len(shards) > 2)

        for shard in shards:
            filename = os.path.splitext(os.path.basename(shard))[0]
            # filename is shard_{index:06d}.npy
            index = int(filename.split("_")[1])
            if split == "valid" and index == 0:
                self.shards = [shard]
                break
            elif split == "train" and index > 0:
                self.shards.append(shard)
        if split == "train":
            self.shards = sorted(self.shards)
            # remove the last shard
            if len(self.shards) > 1:
                # last shard may not be full
                self.shards.pop()
        self.reset()

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
        self.current_token_index += self.B * self.T
        # check if we need to load the next shard
        if self.current_token_index + (self.B * self.T + 1) > len(self.tokens):
            # cycle through the shards, enables to continue get batches for more than one epoch
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            # each process has a different offset in the shard
            # so that they don't overlap
            self.current_token_index = self.B * self.T * self.process_rank
        
        return x, y