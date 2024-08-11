import torch
from bigramUtil import get_names, itos, stoi

if __name__ == "__main__":
    words = get_names("names.txt")

    BLOCK_SIZE = 3
    EMBEDDING_DIM = 2
    NB_NEURONS = 100
    X = []
    Y = []

    # group characters by block_size (e.g. 3)
    # X will contain the int representation of 3 characters, [5,13,13]
    # Y contains the expected next character to this sequence [1]
    for w in words[:1]:
        w = list(w) + ["."]
        #print(w)
        # initialize with [0,0,0], which represents ...
        block = [0] * BLOCK_SIZE
        for c in w:
            X.append(block)
            Y.append(stoi(c))
            #print(f"{block} --> {stoi(c)}")
            # remove first character in block and add the next one
            block = block[1:] + [stoi(c)]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"{X.shape=} {Y.shape=}")

    # we will embed every input character in a CDIM space     
    C = torch.randn((27,EMBEDDING_DIM))

    # let's say that C is a (27,2) tensor. For each character we have 2 numbers representing this character
    # if X is (1000,3) - 1000 samples, each sample is a trigram of 3 characters
    # then C[X] will be a (1000,3,2) Tensor
    emb = C[X]
    print(f"{emb.shape=}")

    W1 = torch.randn((BLOCK_SIZE*EMBEDDING_DIM,NB_NEURONS))
    B1 = torch.randn(NB_NEURONS)
    h = torch.tanh(emb.view(-1,BLOCK_SIZE*EMBEDDING_DIM) @ W1 + B1)
    print(h.shape)

    W1 = torch.randn((NB_NEURONS,27))
    B1 = torch.randn(27)

