"""Train a GPT model using data from wikipedia
Tokenize data and store it to file
"""
import torch
import tiktoken
from datasets import load_dataset
import os
from tqdm import tqdm

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load Wikipedia dataset in streaming mode (no full memory load)
dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")

# Define parameters
SEQ_LENGTH = 256  # Number of tokens per sequence
SAVE_DIR = "data/tokenized_wiki_data/"  # Folder to store tokenized data
os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn’t exist


# Process and save in chunks
chunk_size = 10000  # Number of sequences per saved file
buffer = []  # Temporary storage before saving
file_index = -1  # File numbering

# get index of last file saved to disk in SAVE_DIR
# file name is chunk_{file_index}.pt
# get the biggest file_index

for file in os.listdir(SAVE_DIR):
    if file.startswith("chunk_") and file.endswith(".pt"):
        current_file_index = int(file.split("_")[1].split(".")[0])
        if current_file_index > file_index:
            file_index = current_file_index

# print number of articles in dataset
# format number by adding a dot after thousand and millions, for lisibility
# e.g. :123456789 -> 123.456.789
def format_number(number):
    return "{:,}".format(number).replace(",", " ")
print(f"Number of articles in dataset : {format_number(len(dataset['train']))}")

MAX_ARTICLES = 50000
skipped = 0
# enumerate articles from dataset using tqdm
for i, article in tqdm(enumerate(dataset["train"]), desc="Processing articles",total=MAX_ARTICLES):
    # skip articles that were already processed
    if i <= file_index:
        skipped += 1
        continue
    if skipped > 0:
        print(f"Skipped {format_number(skipped)} articles")
        skipped = 0

    text = article["text"]

    tokens = tokenizer.encode(text) + tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})

    # Split into sequences of SEQ_LENGTH
    for j in range(0, len(tokens), SEQ_LENGTH):
        seq = tokens[j:j + SEQ_LENGTH]

        # Pad if too short
        if len(seq) < SEQ_LENGTH:
            seq += [tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})[0]] * (SEQ_LENGTH - len(seq))

        buffer.append(torch.tensor(seq, dtype=torch.long))

        # Save buffer to disk when full
        if len(buffer) >= chunk_size:
            save_path = os.path.join(SAVE_DIR, f"chunk_{i}.pt")
            torch.save(torch.stack(buffer), save_path)
            buffer = []  # Reset buffer
            
    if i - file_index > MAX_ARTICLES:
        break # we have processed enough articles

