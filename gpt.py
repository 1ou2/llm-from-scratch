#
# # Implementaton of GPT model
#

class Shakespeare:
    def __init__(self):
        self.data = ""
        self.chars = []

    
        shakespeare_dataset = "shakespeare.txt"

        self.data = open(shakespeare_dataset, 'r').read()

        print("Length of dataset in characters: ", len(self.data))
        # print last 500 characters
        #print(all_data[-500:])

        # for tokens we need to count unique characters
        self.chars = sorted(list(set(self.data)))
        print("Number of unique characters: ", len(self.chars))
        print(self.chars)


    def encoder(self):
        # create a mapping from characters to integers
        chars = sorted(list(set(self.data)))
        vocab_size = len(chars)
        print('total chars:', vocab_size)
        # create a mapping from integers to characters
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        return encode, decode

    def train(self):
        pass

    def generate(self):
        pass

    

if __name__ == "__main__":
    shakespeare = Shakespeare()

