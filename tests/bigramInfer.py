import torch
import torch.nn.functional as F
from bigramUtil import get_names, stoi, itos

# Infer words based on a model
#
#
if __name__ == "__main__":

    model_file = "./W-bigram.model"
    nb_words = 10

    # load previously saved model
    W = torch.load(model_file,weights_only=True)
    # initialize random generator
    g = torch.Generator().manual_seed(2147483647)

    for _ in range(nb_words):
        # start token is 0
        ix = 0
        genword = []
        while True:
            # our model takes a one hot vector as input
            # xenc shape is [1, 27]
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            # (1,27)@(27,27) -> logits is (1,27)
            logits = xenc @ W
            counts = logits.exp()
            # p shape is (1,27). It represents the probability of the next character
            #
            # # Example of p for index 10 (letter j)
            # [[0.0158, 0.4935, 0.0105, 0.0043, 0.0058, 0.1269, 0.0037, 0.0064, 0.0040,
            # 0.0296, 0.0070, 0.0076, 0.0051, 0.0124, 0.0043, 0.1470, 0.0040, 0.0113,
            # 0.0020, 0.0115, 0.0036, 0.0497, 0.0070, 0.0098, 0.0037, 0.0073, 0.0060]]
            #
            # 0.0158 or 1.58% is the probability that the next character is 0 (end of sequence)
            # 0.4935 or 49.35% is the probability that the next character is 1 (letter a)
            p = counts / counts.sum(1,keepdim=True)
            
            # sample one character using the probability distribution p 
            ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
            genword.append(itos(ix))
            # end token generated for this word
            if ix == 0:
                break
        print("".join(genword))
    
