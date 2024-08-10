# Bigram language model


from string import ascii_lowercase
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from bigramUtil import get_names, stoi, itos

# step by step tutorial
# Non optimized version for learning purposise
def stepTraining():
    words = get_names("names.txt")

    # create training set.
    nbwords = 1

    # xs the input, the integer representing a character
    # ys the prediction -> the following integer in the bigram
    xs, ys = [], []

    for w in words[:nbwords]:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = stoi(ch1)
            ix2 = stoi(ch2)
            xs.append(ix1)
            ys.append(ix2)

    # create tensor objects based on 
    tx = torch.tensor(xs)
    ty = torch.tensor(ys)
    # number of bigrams
    num = tx.nelement()

    print(f"Number of bigrams: {num}")
    print(f"{tx=}")
    # encode input as a vector of zeros and 1 one. The only representing the input character
    # if we have 10.000 bigrams, xenc is 10000x27 vector
    xenc = F.one_hot(tx, num_classes=27).float()
    print(f"xenc - one hot encoding - size {xenc.shape}")
    print(xenc)

    # Our neural net is a 27x27 tensor initialzed randomly
    # It is a 27 neurons network and each neurons receives 27 inputs
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27,27),generator=g,requires_grad=True)

    # logits is a 10000x27 @ 27x27 -> 10000x27 tensor
    logits = xenc @ W
    print(f"Logits {logits.shape}")
    print(logits)
    # exponentiate results, to have positive numbers. 
    # counts represents the number of times a bigram is counted
    # at this stage, it is a random number
    # 10000x27
    counts = logits.exp()
    print(f"Counts - {counts.shape}")
    print(counts)
    # normalize rows
    # all rows sum to 1, so that probs represent probabilities
    # 10000x27
    # probs [7] -> is a 27 tensor, giving the probability of the each next caracter, associated with the 7th input
    probs = counts / counts.sum(1,keepdim=True)
    print(f"Probs : {probs.shape}")
    print(probs)
    
    # calculate loss function
    # We know the results, they are stored in ty tensor
    # for the first input example we are looking at probs[0, ty[0]], which is the prediction of the network for the correct character
    # e.g. ty = [5,13,13,1]
    # probs[0,5] * probs[1,13] * probs[2,13] * probs[3,1]
    print("Probabilities assigned by network, for the correct character:")
    for i in range(num):
        proba = probs[i,ty[i]]
        print(f"Input {i} - proba: {proba.item()} - logproba {proba.log().item()}")
    print("compute in one pass")
    print(probs[torch.arange(num),ty].log())
    
    
    nlls = torch.zeros(num)
    for i in range(num):
        x = tx[i].item() # input character index
        y = ty[i].item() # label character index
        print("--------------")
        print(f"Bigram example {i}: {itos(x)}{itos(y)} - indexes {x},{y}")
        print(f"input : {x}")
        print(f"probability distribution associated with this input: {probs[i]}")
        print(f"label - actual next character: {y}")
        p = probs[i,y]
        print(f"probability assigned by network to the correct character: {p.item()}")
        logp = torch.log(p)
        nll = - logp.item()
        print(f"Negative log likelihood {nll}")
        print("--------------")
        nlls[i] = nll
    
    print(f"Average log likelihood - aka: GLOBAL Loss {nlls.mean().item()}")


if __name__ == "__main__":


    words = get_names("names.txt")

    # create training set.
    # xs the input, the integer representing a character
    # ys the prediction, the following integer in the bigram
    xs, ys = [], []

    for w in words[:]:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = stoi(ch1)
            ix2 = stoi(ch2)
            xs.append(ix1)
            ys.append(ix2)

    tx = torch.tensor(xs)
    ty = torch.tensor(ys)
    # number of bigrams
    num = tx.nelement()

    # Random initialization of weights
    # Our neural net is a 27x27 tensor initialzed randomly
    # It is a 27 neurons network and each neurons receives 27 inputs
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27,27),requires_grad=True,generator=g)

    # encode input as a vector of zeros and 1 one. The only 1 representing the input character
    # xenc is a "big" tensor. if we have 20000 bigrams in the training set, it is a (20000,27) tensor
    xenc = F.one_hot(tx, num_classes=27).float()
    
    
    step = 20
    iterations = 150
    for _ in range(iterations):


        # Forward pass
        
        # Matrix multiplication
        # logits represents log(count)
        # xenc size is nb_bigrams x 27  
        # W size is 27x 27
        # logits size is nb_bigrams x 27
        logits = xenc @ W

        # use exponential to have an equivalents of counts (positive number)
        counts = logits.exp()
        # normalize so that we can interpret the weights as probabilities
        probs = counts / counts.sum(1,keepdim=True)

        # compute loss function
        # mean value -> mean()
        # of all the negative logs -> -.log()
        # accross all probabilities for the correct character
        # 
        # [torch.arange(num), ty] = [0,y0], [1,y1], ...[n,Yn] 
        loss = -probs[torch.arange(num),ty].log().mean()

        # backward pass
        # upgrade the gradient
        W.grad = None
        loss.backward()

        # nudge data in opposite diretion of grandient
        W.data += -step* W.grad

    print(f"{loss=} - saving model")
    torch.save(W,"./W-bigram.model")

    print("\nInferring some names----")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(6):
        ix = 0
        genword = []
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1,keepdim=True)

            ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
            genword.append(itos(ix))
            if ix == 0:
                break
        print("".join(genword))

    