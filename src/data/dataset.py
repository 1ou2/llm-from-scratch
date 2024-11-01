import json
from abc import ABC, abstractmethod
from pathlib import Path

class Dataset:
    def __init__(self, name,datadir="./data/raw"):
        self.name = name
        self.data = None
        self.datadir = datadir
        self.corpus = None
        self.all_datasets = {"fquad-train": "fquad-train.json", "fquad-valid": "fquad-valid.json","gutenberg":"gutenberg.txt"}
        if name not in self.all_datasets:
            raise ValueError(f"Dataset {name} not found")
        
        #self.load()  # Load the dataset when the class is instantiated"
        #self.load_corpus()

    @abstractmethod
    def load(self):
        """
        Load the dataset from the specified path.
        """
        # Construct the path to the dataset file
        path = f"{self.datadir}/{self.all_datasets[self.name]}"
        
        # Load the dataset from the file
        with open(path, "r") as f:
            self.data = json.load(f)


class _FquadDataset(Dataset):
    def load(self):
        path = f"{self.datadir}/{self.all_datasets[self.name]}"
        with open(path, "r") as f:
            self.data = json.load(f)

    def load_corpus(self):
        corpus = ""
        for sample in self.data["data"]:
            for paragraph in sample["paragraphs"]:
                corpus += paragraph["context"] + " "
                for qas in paragraph["qas"]:
                    corpus += qas["question"] + " "
                    for answer in qas["answers"]:
                        corpus += answer["text"] + " "
        self.corpus = corpus

    def explore(self):
        print(f"Dataset: {self.name}")
        print(f"Number of samples: {len(self.data['data'])}")
        for sample in self.data["data"]:
            print(f"Sample: {sample['title']}")
            print(f"Nb paragraphs: {len(sample['paragraphs'])}")
            for paragraph in sample["paragraphs"]:
                print(f"Paragraph: {paragraph['context']}")
                for qas in paragraph["qas"]:
                    print(f"Question: {qas['question']}")
                    for answer in qas["answers"]:
                        print(f"Answer: {answer['text']}")
            break

class _GutenbergDataset(Dataset):
    """
    Class for loading the Gutenberg books dataset."""
    def __init__(self, name, datadir="./data/preprocessed/gutenberg"):
        super().__init__(name, datadir)
        #self.load_corpus()

    def load(self):
        path = f"{self.datadir}/{self.all_datasets[self.name]}"
        with open(path, "r") as f:
            self.data = f.read()


    def split_files(self):
        file_dir = "data/preprocessed/gutenberg"
        files = list(Path(file_dir).glob("*.txt"))
        # convert path to string
        files = sorted([str(f) for f in files])


        # 80% for training, 10% for validation, 10% for testing
        split = [int(len(files) * 0.8), int(len(files) * 0.9)]
        train_files = files[:split[0]]
        val_files = files[split[0]:split[1]]
        test_files = files[split[1]:]
        
        return train_files, val_files, test_files

class DatasetFactory:
    @staticmethod
    def create_dataset(name, datadir="./data/raw"):
        if name.startswith("fquad"):
            return _FquadDataset(name, datadir)
        elif name == "gutenberg":
            return _GutenbergDataset(name, "data/preprocessed/gutenberg")
        elif name == "names":
            return _NamesDataset(name, datadir)
        else:
            raise ValueError(f"Unknown dataset: {name}")




if __name__ == "__main__":
    name = "fquad-valid"
    ds = DatasetFactory.create_dataset(name)
   
    ds.explore()
    print(len(ds.corpus))


    