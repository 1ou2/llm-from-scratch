# Bigram language model


from string import ascii_lowercase
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# get all the names as a list
# input data is stored in a file, with one name per line
def get_names(filenanme:str)->list:
    names = []
    with open(filenanme,mode="r") as f:
        # strip \n and empty lines
        names = [line.strip().lower() for line in f if line.strip()]
    return names

# convert letter to integer
# letters are a to z. 
# Special token "." is used for beginning and end of sequence
# . is encoded as 0
# a as 1, etc...
def stoi(letter:str)->int:
    if letter == ".":
        return 0
    return ord(letter) -96

# convert integer to letter
# 0 -> .
# 1 -> a, 2->b, etc..
def itos(i:int) ->str:
    if i == 0:
        return "."
    return chr(i+96)




if __name__ == "__main__":
    words = get_names("names.txt")

    # create training set.
    # xs the input, the integer representing a character
    # ys the prediction, the following integer in the bigram
    xs, ys = [], []

    for w in words[:1]:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = stoi(ch1)
            ix2 = stoi(ch2)
            xs.append(ix1)
            ys.append(ix2)

    tx = torch.tensor(xs)
    ty = torch.tensor(ys)
    print(f"{tx=} {ty=}")

    # Random initialization of weights
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27,27),dtype=torch.float32,requires_grad=True,generator=g)

    # Forward pass
    # encode input as a vector of zeros and 1 one. The only representing the input character
    xenc = F.one_hot(tx, num_classes=27).float()

    # Matrix multiplication
    # logits represents log(count)
    # xenc is 5x27 (it depends on the length of the input word, len(word)+1) 
    # W is 27x 27
    # logits is 5x27

    logits = xenc @ W

    # use exponential to have an equivalents of counts (positive number)
    counts = logits.exp()
    # normalize so that we can interpret the weights as probabilities
    probs = counts / counts.sum(1,keepdim=True)
    print(probs)
    print(probs[0].sum())


