# import dataset
from .dataset import Dataset

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
        
