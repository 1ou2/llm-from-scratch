import json
from abc import ABC, abstractmethod
from pathlib import Path
from tokenizers import Tokenizer

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
        train_files, val_files, test_files = self.split_files()
        # load the files
        self.train_corpus = ""
        self.val_corpus = ""
        self.test_corpus = ""
        self.x_batch = []
        self.y_batch = []
        self.current_batch = 0
        for f in train_files:
            with open(f, "r") as f:
                self.train_corpus += f.read()
        for f in val_files:
            with open(f, "r") as f:
                self.val_corpus += f.read()
        for f in test_files:
            with open(f, "r") as f:
                self.test_corpus += f.read()

    def batch_data(self, type="train", block_size=10,nb_batches=-1):
        """return two list of batches x and y (the ground truth)
        type : train, valid, test or mini
        block_size : number of elements in the batch
        nb_batches : maximum number of batches to return, -1 means no limit, process all the data
        """
        if type == "train":
            corpus = self.train_corpus
        elif type == "val":
            corpus = self.val_corpus
        elif type == "test":
            corpus = self.test_corpus
        elif type == "mini":
            corpus = """
C'était pendant la soirée du 10 mars 1793. Dix heures venaient de tinter
à Notre-Dame, et chaque heure, se détachant l'une après l'autre comme un
oiseau nocturne élancé d'un nid de bronze, s'était envolée triste,
monotone et vibrante.

La nuit était descendue sur Paris, non pas bruyante, orageuse et
entrecoupée d'éclairs, mais froide et brumeuse.

Paris lui-même n'était point ce Paris que nous connaissons, éblouissant
le soir de mille feux qui se reflètent dans sa fange dorée, le Paris aux
promeneurs affairés, aux chuchotements joyeux, aux faubourgs bachiques,
pépinière de querelles audacieuses, de crimes hardis, fournaise aux
mille rugissements: c'était une citée honteuse, timide, affairée, dont
les rares habitants couraient pour traverser d'une rue à l'autre, et se
précipitaient dans leurs allées ou sous leurs portes cochères, comme des
bêtes fauves traquées par les chasseurs s'engloutissent dans leurs
terriers.
C'était enfin, comme nous l'avons dit, le Paris du 10 mars 1793.

Quelques mots sur la situation extrême qui avait amené ce changement
dans l'aspect de la capitale, puis nous entamerons les événements dont
le récit fera l'objet de cette histoire.

La France, par la mort de Louis XVI, avait rompu avec toute l'Europe.
Aux trois ennemis qu'elle avait d'abord combattus, c'est-à-dire à la
Prusse, à l'Empire, au Piémont, s'étaient jointes l'Angleterre, la
Hollande et l'Espagne. La Suède et le Danemark seuls conservaient leur
vieille neutralité, occupés qu'ils étaient, du reste, à regarder
Catherine y déchirant la Pologne.
        """ 
        else:
            raise ValueError(f"Unknown type: {type}")

        
        # split the corpus into batches of size block_size
        tokens = []
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        for line in corpus.splitlines():
            line = line.strip()
            if line:
                line_tokens = tokenizer.encode(line)
                tokens.extend(line_tokens.ids)
                if nb_batches != -1:
                    if len(tokens) >= block_size + nb_batches +1:
                        break

        if nb_batches == -1:
            nb_batches = len(tokens) // block_size

        for i in range(0, len(tokens) - block_size -1):
            if nb_batches != -1 and len(self.y_batch) >= nb_batches:
                break
            x_batch = tokens[i:i+block_size]
            y = tokens[i+block_size]
            self.x_batch.append(x_batch)
            self.y_batch.append(y)


    def all_files(self):
        file_dir = "data/preprocessed/gutenberg"
        files = list(Path(file_dir).glob("*.txt"))
        # convert path to string
        files = sorted([str(f) for f in files])
        return files

    def split_files(self):
        files = self.all_files()

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


    