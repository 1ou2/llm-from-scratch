import torch
import tiktoken
from chapter3_gpt import GPTModel,generate_text_simple
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset, random_split
from tqdm import tqdm
import glob
import random

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embed_dim": 468,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # unsqueeze adds the batch dimension
    # if we have 4 tokens shape goes from torch.Size([4]) to torch.Size([1, 4])
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat_list = token_ids.squeeze()
    return tokenizer.decode(flat_list.tolist())


class WikiDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length):
        """
        dataset: Hugging Face dataset (Wikipedia)
        tokenizer: tiktoken tokenizer
        seq_length: length of each training sequence
        """
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.data = []
        print(f"len dataset: {len(dataset["train"])}")
        for i, article in enumerate(dataset["train"]):
            if i % 1000 == 0:
                print(f"processing article {i}")
            if i >= 10000:  # Limit to 1000 articles for simplicity
                break
            tokens = tokenizer.encode(article["text"])
            
            # Add <|endoftext|> at the end of each article
            tokens.append(tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})[0])

            # Ensure correct chunking by padding only within an article
            excess = len(tokens) % self.seq_length
            if excess != 0:
                tokens.extend([tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})[0]] * (self.seq_length - excess))
            
            self.data.extend(tokens)  # Add to dataset buffer

    def __len__(self):
        return len(self.data) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        return torch.tensor(self.data[start:end], dtype=torch.long)

class TokenizedWikiDataset(Dataset):
    def __init__(self, files, split="train"):
        self.files = files
        self.NB_SEQ_PER_FILE = 10000 # FIXED number of sequences per chunk file
        self.split = split
        self.data = []  
        self.current_file_index = -1
        self.load_chunk_file(0)
        

    def load_chunk_file(self,file_index):
        if file_index == self.current_file_index:
            return # already loaded

        if file_index >= len(self.files):
            raise IndexError("File index out of range")

        print(f"Loading file {self.files[file_index]}")        
        self.current_file_index = file_index
        full_data = torch.load(self.files[self.current_file_index],weights_only=True)
        # shuffle data
        random.seed(123) # ensure that the split will always lead to same result
        random.shuffle(full_data)
        split_idx = int(len(full_data) * 0.9) # 90% for training
        self.data = full_data[:split_idx] if self.split == "train" else full_data[split_idx:]
        
    
    def __len__(self):
        return (len(self.files) * self.NB_SEQ_PER_FILE // 10) if self.split == "valid" else (len(self.files) * self.NB_SEQ_PER_FILE * 9 // 10)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        # Load next file if needed
        file_index = idx // (self.NB_SEQ_PER_FILE // 10 if self.split == "valid" else self.NB_SEQ_PER_FILE * 9 // 10)
        self.load_chunk_file(file_index)
        local_idx = idx % (self.NB_SEQ_PER_FILE // 10 if self.split == "valid" else self.NB_SEQ_PER_FILE * 9 // 10) # index within the file
        
        return self.data[local_idx]  # Return tokenized sequence


def eval_model():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    start_content = "Every effort moves you"
    token_ids = generate_text_simple(model, text_to_token_ids(start_content, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"])
    print(f"{token_ids.shape=}")
    print(f"{token_ids=}")
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"{decoded_text=}")

    text1 = "every effort moves"
    text2 = "I really like"
    in_tokens1 = text_to_token_ids(text1, tokenizer).squeeze(0)
    in_tokens2 = text_to_token_ids(text2, tokenizer).squeeze(0)
    inputs = torch.stack((in_tokens1, in_tokens2), dim=0)

    target1 = " effort moves you"
    target2 = " really like chocolate"
    target_tokens1 = text_to_token_ids(target1, tokenizer).squeeze(0)
    target_tokens2 = text_to_token_ids(target2, tokenizer).squeeze(0)
    targets = torch.stack((target_tokens1, target_tokens2), dim=0)
    print(f"{inputs.shape=}")
    print(f"{inputs=}")
    print(f"{targets.shape=}")
    print(f"{targets=}")

    with torch.no_grad():
        logits = model(inputs)
    # probas.shape=torch.Size([2, 3, 50257])
    probas = logits.softmax(dim=-1)

    # token_ids.shape=torch.Size([2, 3, 1])
    token_ids = torch.argmax(probas, dim=-1,keepdim=True)

    text_idx = 0
    # for batch 0 (text_idx)
    # get the probabilities of target tokens
    # [0,1,2] means get probas for tokens at positions 0, 1 and 2
    # target_tokens1 is a tensor of shape [3] (3 tokens)
    # targets[text_idx] is a tensor of shape [3] (3 tokens)
    # so we get the probas for the 3 tokens at positions 0, 1 and 2
    # target_probas_0 is a tensor of shape [3] (3 probabilities)
    target_probas_0 = probas[text_idx, [0,1,2], targets[text_idx]]
    text_idx = 1
    target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
    
    target_probas = torch.cat((target_probas_0, target_probas_1), dim=0)
    print(f"{target_probas=}")
    log_probas = torch.log(target_probas)
    print(f"{log_probas=}")
    neg_avg_log_probas = -log_probas.mean()
    print(f"{neg_avg_log_probas=}")

    print(f"{logits.shape=}")
    print(f"{targets.shape=}")
    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()
    print(f"{logits_flat.shape=}")
    print(f"{targets_flat.shape=}")
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"{loss=}")


def load_data():
    file_path = "data/raw/the-verdict.txt"
    with open(file_path, "r") as f:
        text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    total_chars = len(text)
    total_tokens = len(tokenizer.encode(text))
    print(f"{total_chars=}")
    print(f"{total_tokens=}")

def load_wikipedia():
    hf_dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")
    tokenizer = tiktoken.get_encoding("gpt2")
    wiki_dataset = WikiDataset(hf_dataset, tokenizer, 256)
    print(f"{len(wiki_dataset)=}")
    train_size = int(0.9 * len(wiki_dataset))
    # randomly allocate data to training and validation set
    # order of data is preserved
    train_dataset, val_dataset = random_split(wiki_dataset, [train_size, len(wiki_dataset) - train_size])
    print(f"{len(train_dataset)=}")
    print(f"{len(val_dataset)=}")
    # shuffle data order in train loader so that each epoch is different
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,drop_last=True)
    # don't shuffle validation as we want consistent result between evaluations
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True)
    # Example usage
    for batch in train_loader:
        print(batch)
        print(batch.shape)  # Expected: torch.Size([8, 1024])
        break

def load_tokenized_wikipedia():
    tokenizer = tiktoken.get_encoding("gpt2")
    all_files = glob.glob("data/tokenized_wiki_data/chunk_*.pt")
    train_dataset = TokenizedWikiDataset(all_files, split="train")
    valid_dataset = TokenizedWikiDataset(all_files, split="valid")

    # as we are loading using chunk file we cannot shuffle the training dataset
    # otherwise we are switching between chunk files
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, drop_last=True)

    for i,batch in enumerate(valid_loader):
        # check if batch is a tensor
        if not isinstance(batch, torch.Tensor):
            print(f"batch is not a tensor at index {i}")
            continue
        # check if batch size is [8,256]
        if batch.shape != torch.Size([8, 256]):
            print(f"batch shape is not [8,256] at index {i}")
            continue
        if i%1000 == 0:
            print(f"{i=}")


    


if __name__ == "__main__":
    #eval_model()
    #load_wikipedia()
    load_tokenized_wikipedia()