# import dataset
from dataset import Dataset
import regex as re
import json

class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.position = 0

    def has_next(self):
        return self.position < len(self.text)

    def next(self):
        if self.position < len(self.text):
            token = self.text[self.position]
            self.position += 1
            return token
        else:
            return None
        
    def tokenize(self):
        tokens = []
        for c in self.text:
            tokens.append(ord(c))

        return tokens
        
    def test_utf8(self):
        text1 = "こんにちは、世界 ！"
        text2 = "Hello, world!éèà\n"
        # add emoji
        text3 = "Ha ! é \n👋"
        texts = [text1, text2, text3]

        for text in texts:
            print(f"Text: {text}")
            for c in text:
                print(f"Char: {c} - UTF-8: {ord(c)} - binary: ",end="")
                # get binary representation of the character
                for i in range(8):
                    print(f"{(ord(c) >> (7 - i)) & 1}", end="")
                print()
            print("\n")
            text_to_bits = list(map(bin,bytearray(text,encoding="utf-8")))
            print(text_to_bits)
            for b in text_to_bits:
                print(f"*{b}|{int(b,2)}* ",end="")
            print("\n")
            text_bytes = text.encode("utf-8") # raw bytes
            print(text_bytes)
            ids = list(text_bytes) # list of integers in range 0..255
            print(ids)

    def preprocess(self,text=""):
        gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if text:
            matches = gpt2pat.findall(text)
        else:
            matches = gpt2pat.findall(self.text)
        print(matches)

    def test_preprocess(self):
        texts = ["    Hello world!", "how are you  ", "hi word123 next"]
        for t in texts:
            print(t)
            self.preprocess(t)

class GPT2Tokenizer():
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.encoder = {}
        self.decoder = {}
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.text = ""
        self.position = 0

    def load_vocab(self):
        # files vocab.bpe and encoder.json are downloaded from the internet
        # load file from data directory
        with open("data/vocab.bpe", "r") as f:
            bpe_data = f.read()
            self.merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

        with open("data/encoder.json", "r") as f:
            self.encoder = json.load(f)
        

    def encode(self, text):
        pass

    def decode(self, tokens):
        pass

    def preprocess(self, text):
        gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if text:
            matches = gpt2pat.findall(text)
        else:
            matches = gpt2pat.findall(self.text)
        print(matches)

    def download_vocab(self):
        # download the vocabulary from the internet
        vocab_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe"
        encoder_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json"


class CharacterTokenizer():
    def __init__(self):
        pass

    def encode(self, text):
        tokens = []
        for c in text:
            tokens.append(ord(c))

        return tokens

    def decode(self, tokens):
        text = ""
        for token in tokens:
            text += chr(token)

        return text

def main():
    text = "Hello, world!éèà\n"
    japanese_text = "こんにちは、世界 ！!"
    tokenizer = Tokenizer(text)

    tokenizer.test_utf8()

    ct = CharacterTokenizer()
    tokens = ct.encode("salut toi !")
    print(tokens)
    print(ct.decode(tokens))
    tokens = ct.encode("ab zA Z!")
    print(tokens)

    tokenizer.test_preprocess()

if __name__ == "__main__":
    gpt2 = GPT2Tokenizer()
    gpt2.load_vocab()
    print(f"{len(gpt2.merges)=}")
    print(f"{len(gpt2.encoder)=}")

    min_interval = 1000
    max_interval = 1030
    iter = 0
    longest_key = max(gpt2.encoder, key=lambda k: len(k))
    print(f"Longest key: {longest_key} - ")
    # print first maxiter encoder
    for k, v in gpt2.encoder.items():
        
        iter += 1
        if iter == max_interval:
            break
        if iter > min_interval:
            print(f"{k} : {v}")

    k1028 = ""
    for k,v in gpt2.encoder.items():
        if v == 1028:
            k1028 = k
            break
    print(f"{k1028=}")
    print(f"{len(k1028)=}")
    text_bytes = k1028.encode("utf-8") # raw bytes
    print(text_bytes)
    ids = list(text_bytes) # list of integers in range 0..255
    print(ids)
    text_to_bits = list(map(bin,bytearray(k1028,encoding="utf-8")))
    print(f"{text_to_bits=}")
    print(ord("Ġ"))
    print(chr(288))
    text_to_bits = list(map(bin,bytearray(chr(288),encoding="utf-8")))
    print(f"{text_to_bits=}")
    prefix = "Ġ"
    print(prefix.encode("unicode_escape"))
    
    for k,v in gpt2.encoder.items():
        if v == 32827:
            print(f"{k=}")
            break
    k1028 = "\tagainst"
    print(f"{k1028=}")
    print(f"{len(k1028)=}")
    text_bytes = k1028.encode("utf-8") # raw bytes
    print(text_bytes)
    ids = list(text_bytes) # list of integers in range 0..255
    print(ids)

    for k,v in gpt2.encoder.items():
        if v == 160:
            print(f"{k=}")
            break