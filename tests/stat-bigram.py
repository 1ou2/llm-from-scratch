from string import ascii_lowercase
import matplotlib.pyplot as plt
import torch

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

# plot a 2D graph of all possible bigrams, with associated probability
# N : a 27x27 tensor object storing all probabilities for a given bigram
# N[2,3] -> probability of the bigram associated with 2 and 3 <-> bc
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

# generate samples from the probability distribution
def generateSamples(P):
    print("GENERATING Names")
    for _ in range(10):
        # index token for starting generation
        ix = 0
        genword = []
        while True:      
            # generate samples from the probability distribution
            # Given the probability distribution P and the current caracter ix, sample the next token      
            ix = torch.multinomial(P[ix],num_samples=1,replacement=True,generator=g).item()
            genword.append(itos(ix))
            # end generation when output character is the end token
            if ix == 0:
                break
        print("".join(genword))

if __name__ == "__main__":
    #filename = "sample-names.txt"
    filename = "names.txt"
    
    names = get_names(filename)
    
    """ nb_names = len(names)
    if nb_names < 10:
        print(names)
    else:
        print(f"List too long {nb_names} - truncating to 10")
        print(names[:10]) """

    #words = names[:5]
    # all possible letters in the alphabet used
    # . is a special token for marking beginning and end of word
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

    #showBigrams(N)
    #     
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
    #print("Top ten probabilities ",end="")
    #print(sorted(probabilities.items(),key=lambda it:it[1],reverse=True)[:10])

    
    # P is a 27x27 Tensor containing all the probabilities to compute the next token
    # given an input character C1, what is the probality of the second character C2
    # in order to do that we need to normalize each rows, so that the sum of P(ROWi) = 1
    #

    # convert to float
    # The +1, is what is called model smoothing. 
    # It ensures that we have no Zero in P. We want to avoid 0, which would mean impossible to have this combination
    # 
    P = (N+1).float()
    # We are diving a 27x27 by a 27x1 (the sum)
    P /= P.sum(dim=1,keepdim=True)
    
    g = torch.Generator().manual_seed(2147483647)
        
    # generate samples from the probability distribution
    generateSamples(P)

    #losslog = sum(torch.log(P))
    #print(losslog)

    # Evaluate quality of model
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = stoi(ch1)
            ix2 = stoi(ch2)
            prob = P[ix1,ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n+=1
            print(f"{ch1}{ch2} {prob:.4f} {logprob:.4f}")

    print(f"Normalized log likelihood: {-log_likelihood/n:.4f}")