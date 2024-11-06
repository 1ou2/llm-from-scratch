from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from data import DatasetFactory

"""

def split_files():
    file_dir = "data/preprocessed"
    files = list(Path(file_dir).glob("*.txt"))
    # convert path to string
    files = sorted([str(f) for f in files])


    # 80% for training, 10% for validation, 10% for testing
    split = [int(len(files) * 0.8), int(len(files) * 0.9)]
    train_files = files[:split[0]]
    val_files = files[split[0]:split[1]]
    test_files = files[split[1]:]
    
    return train_files, val_files, test_files
"""
def train_tokenizer(train_files,vocab_size=1024):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files=train_files, trainer=trainer)
    tokenizer.save("data/tokenizer.json")

def test_tokenizer():
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    print(tokenizer.get_vocab_size())
    text = "C’est 漢 ici auintpndot Ô "
    encoded = tokenizer.encode(text)
    print(encoded.ids)


if __name__ == "__main__":
    #gutenberg_dataset = DatasetFactory.create_dataset("gutenberg")
    #train_files = gutenberg_dataset.all_files()
    #train_tokenizer(train_files, vocab_size=1000)
    test_tokenizer()