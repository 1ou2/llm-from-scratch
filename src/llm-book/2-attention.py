import torch
import tiktoken
import torch.nn as nn


def self_attention():
    embedding_dim = 3
    vocab_size = 50257 # GPT-2 vocab size
    input_sentence = "For sale: baby shoes, never worn"

    # for reproducibility
    torch.manual_seed(1234)
    # print options : use 2 digits only 
    torch.set_printoptions(precision=2, sci_mode=False)
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #         For   sale  :    baby  shoes  ,  never  worn
    # tokens=[1890, 5466, 25, 5156, 10012, 11, 1239, 12666]
    tokens = tokenizer.encode(input_sentence)
    
    

    # embedding layer
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    print(f"{embedding_layer.weight.shape=}")

    input_tokens = torch.tensor(tokens)
    inputs_embedding = embedding_layer(input_tokens) # 8x3
    print(f"{inputs_embedding.shape=}")
    print(f"input_embedding=\n{inputs_embedding.data}")

    # query for the token number 3, which represents the word baby
    query_baby = inputs_embedding[3] # 1x3
    print(f"{query_baby.data=}")
    attn_scores_baby = torch.empty(inputs_embedding.shape[0], dtype=torch.float32) # 1x8

    for i, input_embedding in enumerate(inputs_embedding):
        attn_scores_baby[i] = torch.dot(query_baby, input_embedding) # attn[i] = dot(1x3,1x3) = scalar
    print(f"{attn_scores_baby.data=}")
    #attn_scores_baby = torch.matmul(query_baby, inputs_embedding.T) # matmul(1x3, 3x8) -> 1x8
    #print(f"{attn_scores_baby.data}")
    attn_weights_baby = torch.nn.functional.softmax(attn_scores_baby, dim=0) # 1x8
    print(f"{attn_weights_baby=}")

    #context_vector_baby = torch.matmul(attn_weights_baby, inputs_embedding) # matmul(1x8, 8x3) -> 1x3
    context_vector_baby = torch.zeros(embedding_dim, dtype=torch.float32)
    
    for i, input_embedding in enumerate(inputs_embedding):
        context_vector_baby += attn_weights_baby[i] * input_embedding
        #print(f"{context_vector_baby.data} = {attn_weights_baby[i].data:.2f} * {input_embedding.data}")
    
    print(f"{context_vector_baby.data=}")

    attn_scores = torch.matmul(inputs_embedding, inputs_embedding.T) # matmul(8x3, 3x8) -> 8x8
    attn_weights = torch.nn.functional.softmax(attn_scores,dim=1) # 8x8
    context_vector = torch.matmul(attn_weights, inputs_embedding) # matmul(8x8, 8x3) -> 8x3
    print(f"{context_vector.data}")
    print(f"{context_vector[3]=}")

def trainable_self_attention():
    embedding_dim = 3 # input dimensiom
    context_dim = 2 # output dimensiom
    vocab_size = 50257 # GPT-2 vocab size
    input_sentence = "For sale: baby shoes, never worn"



    # for reproducibility
    torch.manual_seed(1234)
    # print options : use 2 digits only
    torch.set_printoptions(precision=2, sci_mode=False)
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #         For   sale  :    baby  shoes  ,  never  worn
    # tokens=[1890, 5466, 25, 5156, 10012, 11, 1239, 12666]
    tokens = tokenizer.encode(input_sentence)


    # embedding layer
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    print(f"{embedding_layer.weight.shape=}")

    input_tokens = torch.tensor(tokens)
    inputs_embedding = embedding_layer(input_tokens) # 8x3
    print(f"{inputs_embedding.shape=}")
    print(f"input_embedding=\n{inputs_embedding.data}")

    query_params = torch.nn.Parameter(torch.randn(embedding_dim, context_dim,dtype=torch.float32))  # 3x2
    key_params = torch.nn.Parameter(torch.randn(embedding_dim, context_dim, dtype=torch.float32))   # 3x2
    value_params = torch.nn.Parameter(torch.randn(embedding_dim, context_dim, dtype=torch.float32)) # 3x2

    query_baby = inputs_embedding[3] @ query_params # 1x3 @ 3x2 -> 1x2
    key_baby = inputs_embedding[3] @ key_params # 1x3 @ 3x2 -> 1x2
    value_baby = inputs_embedding[3] @ value_params # 1x3 @ 3x2 -> 1x2

    print(f"{query_baby.shape=}")
    keys = inputs_embedding @ key_params # 8x3 @ 3x2 -> 8x2
    values = inputs_embedding @ value_params # 8x3 @ 3x2 -> 8x2
    print(f"{keys.shape=}")


    attn_scores_baby = torch.empty(inputs_embedding.shape[0], dtype=torch.float32)
    for i, key in enumerate(keys):
        attn_scores_baby[i] = torch.dot(query_baby, key) # attn[i] = dot(1x2, 1x2) = scalar
    
    print(f"{attn_scores_baby.data=}") # 8x1

    attn_scores_3 = query_baby @ keys.T # 1x2 @ 2x8 -> 1x8
    print(f"{attn_scores_3.data=}") # 8x1
    print(f"{attn_scores_3.shape=}") # 8x1

    d_k = keys.shape[-1]
    attn_weights_baby = torch.nn.functional.softmax(attn_scores_baby / torch.sqrt(torch.tensor(d_k)), dim=-1) # 1x8
    
    print(f"{attn_weights_baby=}")

    context_vector_baby = torch.matmul(attn_weights_baby, values) # matmul(1x8, 8x2) -> 1x2
    print(f"{context_vector_baby.data=}")

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.nn.functional.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        context_vec = attn_weights @ values
        return context_vec

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.nn.functional.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # create an upper triangle mask of ones (excluding diagonal), and register as "mask", can be accessed using self.mask
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len),diagonal=1))

    def forward(self, x):
        # batch, number of tokens, input embedding dimension
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # dimension 0 is the batch, transpose along dimension 1 and 2
        attn_scores = queries @ keys.transpose(1, 2)
        d_k = keys.shape[-1]
        # if mask value is true, fill with -infinity
        # [:num_tokens, :num_tokens] : is used to handle smaller input size 
        # The syntax [:num_tokens] means "take all elements from the start up to num_tokens". It's equivalent to [0:num_tokens]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))

        attn_weights = torch.nn.functional.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

if __name__ == "__main__":
    #trainable_self_attention()
    torch.manual_seed(123)
    # print options : use 2 digits only
    torch.set_printoptions(precision=2, sci_mode=False)
    sa_v1 = SelfAttentionV1(3, 2)
    input_sentence = "For sale: baby shoes, never worn"
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(input_sentence)
    input_tokens = torch.tensor(tokens)
    embedding_layer = torch.nn.Embedding(50257, 3)
    inputs_embedding = embedding_layer(input_tokens)
    print(f"{inputs_embedding=}")

    context_vec = sa_v1(inputs_embedding)
    print(f"{context_vec=}")

    batch = torch.stack((inputs_embedding, inputs_embedding), dim=0)
    print(f"{batch.shape=}")

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(3, 2, context_length, 0.0)
    context_vecs = ca(batch)
    print(f"{context_vecs.shape=}")
    print(f"{context_vecs=}")
