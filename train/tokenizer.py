import torch


tokenizer = tiktoken.get_encoding("o200k_base")

def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Adiciona o Batch
    return encoded_tensor.to(device)


def token_ids_to_text(token_ids, tokenizer, device):
    flat = token_ids.squeeze(0) # Tira a dimensão de Batch
    return tokenizer.decode(flat.tolist())