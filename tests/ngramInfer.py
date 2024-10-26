import torch
import torch.nn.functional as F
from bigramUtil import get_names, stoi, itos

# Infer words based on a model
#
#
if __name__ == "__main__":

    nb_words = 20
    BLOCK_SIZE = 3

    # load previously saved model
    C = torch.load("ngram-C.model",weights_only=True)
    W1 = torch.load("ngram-W1.model",weights_only=True)
    B1 = torch.load("ngram-B1.model",weights_only=True)
    W2 = torch.load("ngram-W2.model",weights_only=True)
    B2 = torch.load("ngram-B2.model",weights_only=True)

    g = torch.Generator().manual_seed(123654789)

    for _ in range(nb_words):
        out = []
        context = [0] * BLOCK_SIZE
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1,-1)) @ W1 + B1
            logits = h @ W2 + B2
            probs = F.softmax(logits,dim=1)
            ix = torch.multinomial(probs,num_samples=1,generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        #print(out)
        print("".join(itos(i) for i in out))