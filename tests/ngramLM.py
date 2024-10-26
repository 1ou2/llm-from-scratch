import torch
from bigramUtil import get_names, itos, stoi
import torch.nn.functional as F

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
    for w in words[:]:
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



    g = torch.Generator().manual_seed(123654789)
    # we will embed every input character in a CDIM space     
    C = torch.randn((27,EMBEDDING_DIM),generator=g)
    # first hidden layer
    W1 = torch.randn((BLOCK_SIZE*EMBEDDING_DIM,NB_NEURONS),generator=g)
    B1 = torch.randn(NB_NEURONS,generator=g)
    
    # final layer
    W2 = torch.randn((NB_NEURONS,27),generator=g)
    B2 = torch.randn(27,generator=g)

    parameters = [C,W1,B1,W2,B2]
    for p in parameters:
        p.requires_grad = True
    nbparams = sum(p.nelement() for p in parameters)
    
    print(f"nb parameters: {nbparams}")

    # let's say that C is a (27,2) tensor. For each character we have 2 numbers representing this character
    # if X is (1000,3) - 1000 samples, each sample is a trigram of 3 characters
    # then C[X] will be a (1000,3,2) Tensor
    bestLoss = 1000
    for j in range (20000):

        # mini batch
        ix = torch.randint(0,X.shape[0], (32,))

        emb = C[X[ix]]
        h = torch.tanh(emb.view(-1,BLOCK_SIZE*EMBEDDING_DIM) @ W1 + B1)
        logits = h @ W2 + B2

        # manual way 
        #counts = logits.exp()
        #prob =  counts / counts.sum(1,keepdim=True)
        #loss = -prob[torch.arange(32),Y].log().mean()
        # using torch
        loss = F.cross_entropy(logits,Y[ix])

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        if j < 100000:
            rate = 0.1
        else:
            rate = 0.01
        for p in parameters:
            p.data += -rate * p.grad
        if j % 1000 == 0:
            currentLoss = loss.item()
            print(f"{j:4} : {loss.item()=}")
            if currentLoss < 2.5 and currentLoss < bestLoss:
                bestLoss = currentLoss
                print(f"saving model with loss {currentLoss}")
                torch.save(C,"ngram-C.model")
                torch.save(W1,"ngram-W1.model")
                torch.save(B1,"ngram-B1.model")
                torch.save(W2,"ngram-W2.model")
                torch.save(B2,"ngram-B2.model")
            emb = C[X[ix]]
    
    print(f"Best loss achieved {bestLoss}")
    emb = C[X]
    h = torch.tanh(emb.view(-1,BLOCK_SIZE*EMBEDDING_DIM) @ W1 + B1)
    logits = h @ W2 + B2

    # manual way 
    #counts = logits.exp()
    #prob =  counts / counts.sum(1,keepdim=True)
    #loss = -prob[torch.arange(32),Y].log().mean()
    # using torch
    loss = F.cross_entropy(logits,Y)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    print(f"Last loss achieved {loss.item()}")