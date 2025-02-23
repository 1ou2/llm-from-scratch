import torch
import tiktoken
from chapter3_gpt import GPTModel,generate_text_simple

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



if __name__ == "__main__":
    eval_model()

