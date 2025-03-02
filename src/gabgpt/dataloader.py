
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
    def __init__(self, B, T, split, token_dir, process_rank =0, num_processes=1):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.split = split
        self.shards = []
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.token_dir = token_dir # DATA_TOKENIZED_DIR = "data/tokenized/wikipedia_fr/"
        self.update_shard_list()
        self.reset()

    def update_shard_list(self):
        self.shards = sorted([os.path.join(self.token_dir, f) for f in os.listdir(self.token_dir) if f.endswith(".npy")])

        if self.split == "train":
            # remove the last shard
            if len(self.shards) > 1:
                # last shard may not be full
                self.shards.pop()


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
            # check if we ran out of shards
            if self.current_shard_index + 1 >= len(self.shards):
                # try checking if a new shard is available
                # for optimization reasons, we might still be sending new shards to a remote GPU server
                self.update_shard_list()

            # cycle through the shards, enables to continue get batches for more than one epoch
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            # each process has a different offset in the shard
            # so that they don't overlap
            self.current_token_index = self.B * self.T * self.process_rank
        
        return x, y