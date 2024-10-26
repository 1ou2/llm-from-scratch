import json
class Dataset:
    def __init__(self, name,datadir="./dataset/data"):
        self.name = name
        self.data = None
        self.datadir = datadir
        self.corpus = None
        self.load()  # Load the dataset when the class is instantiated"
        self.load_corpus()

    def load(self):
        """
        Load the dataset from the specified path.
        """
        # Construct the path to the dataset file
        path = f"{self.datadir}/{self.name}.json"
        
        # Load the dataset from the file
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
        """
        Explore the dataset and print information about it.
        """
        # Print the name of the dataset
        print(f"Dataset: {self.name}")

        # Print the number of samples in the dataset
        print(f"Number of samples: {len(self.data["data"])}")
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


if __name__ == "__main__":
    # Create an instance of the Dataset class
    name = "fquad-valid"
    ds = Dataset(name)

    print(len(ds.corpus))
    
    