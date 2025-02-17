import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

GPT_CONFIG_1 = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        d_in : dimension of input
        d_out : dimension of output
        context_length : 
        dropout : percentage of neurons to dropout
        num_heads : number of heads
        qkv_bias : use a bias ?
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # create an upper triangle mask of ones (excluding diagonal), and register as "mask", can be accessed using self.mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        # batch, number of tokens, input embedding dimension
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # reshape keys, queries, values
        # we have concatenated all heads in big matrixes
        # we need to unroll and split the d_out dimension into num_heads * head_dim
        # view :(b, num_tokens, d_out) -> (b,num_tokens, num_heads, head_dim)
        # transpose(1,2) : (b,num_tokens, num_heads, head_dim) -> (b,num_heads,num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # compute dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        
        # if mask value is true, fill with -infinity
        # [:num_tokens, :num_tokens] : is used to handle smaller input size
        # The syntax [:num_tokens] means "take all elements from the start up to num_tokens". It's equivalent to [0:num_tokens]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))

        d_k = keys.shape[-1]
        attn_weights = torch.nn.functional.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        # dropout some weights to prevent overfitting
        attn_weights = self.dropout(attn_weights)

        # combine heads
        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        # transpose(1, 2) : (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # contiguous() : to avoid error when using view() later
        # view : (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
   
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # add a linear projection
        # After concatenating all the attention heads together, we need to project the concatenated vector back to the expected output dimension
        # The projection allows the model to learn how to optimally combine and weight the information from different attention heads
        context_vec = self.out_proj(context_vec)

        return context_vec
    

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


if __name__ == "__main__":
    torch.manual_seed(123)
    # print options : use 2 digits only
    torch.set_printoptions(precision=2, sci_mode=False)

    input_sentence = "For sale: baby shoes, never worn"
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(input_sentence)
    input_tokens = torch.tensor(tokens)
    embedding_layer = torch.nn.Embedding(50257, 3)
    inputs_embedding = embedding_layer(input_tokens)
    print(f"{inputs_embedding=}")

    batch = torch.stack((inputs_embedding, inputs_embedding), dim=0)
    print(f"{batch.shape=}")

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, 2)
    context_vec = mha(batch)
    print(f"{context_vec=}")