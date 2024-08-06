from string import ascii_lowercase
import matplotlib.pyplot as plt
import torch

def get_names(filenanme:str)->list:
    names = []
    with open(filenanme,mode="r") as f:
        # strip \n and empty lines
        names = [line.strip().lower() for line in f if line.strip()]
    return names

# convert letter to integer
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

def showBigrams(N):
    plt.figure(figsize=(14,14))
    plt.imshow(N,cmap="Blues")
    for i in range(size_n):
        for j in range(size_n):
            bigram = itos(i)+itos(j)
            plt.text(j,i,bigram,ha="center",va="bottom",color="grey")
            plt.text(j,i,N[i,j].item(),ha="center",va="top",color="grey")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    #filename = "sample-names.txt"
    filename = "names.txt"
    
    names = get_names(filename)
    
    nb_names = len(names)
    if nb_names < 10:
        print(names)
    else:
        print(f"List too long {nb_names} - truncating to 10")
        print(names[:10])

    #words = names[:5]
    # all possible letters in the alphabet used
    alphabet = "." + ascii_lowercase
    size_n = len(alphabet)

    words = names
   
    bigrams = dict()
    # 27x27 tensor containing all counts of bigram
    # Number of bigram "cd" is N[4,5]
    N = torch.zeros(size_n,size_n,dtype=torch.int32)
    for w in words:
        # . mark begining and end of word
        w = "." + w + "."

        # a bigram is composed of two consecutive letters or symbol (".")
        # iterate through the word and count the number of times this bigram appears
        for a,b in zip(w,w[1:]):
            bigrams[a+b] = bigrams.get(a+b,0) +1
            N[stoi(a),stoi(b)] = bigrams.get(a+b) 

    #print(N)
    #showBigrams(N)

    
    # total number of bigrams
    nb_bigrams = sum(bigrams.values())
    
    # dictionary 
    # bigram -> proba
    # e.g. an -> 0.0023
    probabilities = dict()
    # compute probability of each bigram
    for a in alphabet:
        for b in alphabet:
            probabilities[a+b] = bigrams.get(a+b,0)/nb_bigrams
    print(sorted(probabilities.items(),key=lambda it:it[1],reverse=True)[:10])

    
    P = torch.zeros(size_n,size_n,dtype=torch.float32)
    for a in alphabet:
        for b in alphabet:
            P[stoi(a),stoi(b)] = bigrams.get(a+b,0)/nb_bigrams
    
    g = torch.Generator().manual_seed(2147483647)
    p = torch.rand(3,generator=g)
    p = p /sum(p)
    print(p)
    
    # generate samples from the probability distribution
    samples = torch.multinomial(P[0],num_samples=1,replacement=True,generator=g)
    print(itos(samples.item()))