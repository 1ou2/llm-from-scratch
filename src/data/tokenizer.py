# import dataset
from dataset import Dataset

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
        
if __name__ == "__main__":
    text = "Hello, world!éèà\n"
    japanese_text = "こんにちは、世界 ！!"
    tokenizer = Tokenizer(text)


    tokens = tokenizer.tokenize()
    print(tokens)
    jap_tokens = Tokenizer(japanese_text).tokenize()
    print(jap_tokens)