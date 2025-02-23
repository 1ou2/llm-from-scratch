# download french wikipedia dataset from hugging face
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.fr")
print(ds["train"][0])
