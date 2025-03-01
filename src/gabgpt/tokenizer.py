"""
Custom Tokenizer created from french dataset.
"""
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from datasets import load_dataset

def train():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>", ]

    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Load the dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")

    # Get 10% of the training split
    texts = dataset["train"].select(range(len(dataset["train"]) // 10))["text"]

    # Train on your dataset while adding special tokens
    tokenizer.train_from_iterator(
        texts, 
        vocab_size=32768, 
        min_frequency=2, 
        special_tokens=special_tokens
    )

    # Save the tokenizer
    # ls french_tokenizer/
    # gabgpt-merges.txt  gabgpt-vocab.json
    tokenizer.save_model("french_tokenizer",prefix="gabgpt")

def use_tokenizer():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "french_tokenizer/gabgpt-vocab.json",
        "french_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)

    # Test the tokenizer
    test_texts = [
        "Ceci est un exemple de phrase pour tester la tokenization.", 
        "Les réseaux de neurones artificiels sont fascinants !",
        "La programmation est une compétence essentielle pour les développeurs.",
        "L'IA est révolutionnaire dans le domaine de l'intelligence artificielle.",
        "une phrase accentuée à impersonnel et tête avait eu"
    ]

    for text in test_texts:
        output = tokenizer.encode(text)
        print(f"Text: {text}")
        print(f"Tokens: {output.tokens}")
        print(f"IDs: {output.ids}\n")
        print(f"Nb tokens: {len(output.ids)}")

    # Check if special tokens are recognized
    for token in special_tokens:
        encoded = tokenizer.encode(token)
        print(f"Special token '{token}' -> Tokens: {encoded.tokens}")
        print(f"Special token '{token}' -> ID: {encoded.ids}")


    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Method 2: Get the vocabulary size from the tokenizer
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Optional: You can also inspect some vocabulary items
    print("\nFirst 10 vocabulary items:")
    for token, id in list(vocab.items())[:10]:
        print(f"Token: {token:20} ID: {id}")

    print("\nLast 10 vocabulary items:")
    for token, id in list(vocab.items())[-10:]:
        print(f"Token: {token:20} ID: {id}")

if __name__ == "__main__":


    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "french_tokenizer/gabgpt-vocab.json",
        "french_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    txt = "Bonjour<|endoftext|>"
    encoded = tokenizer.encode(txt)
    print(encoded.ids)
    decoded_text = tokenizer.decode(encoded.ids)
    print(decoded_text)
    print(tokenizer.encode(decoded_text).ids)
    # print also special tokens
    eot = tokenizer.token_to_id("<|endoftext|>")
    print(f"{eot=}")
    
