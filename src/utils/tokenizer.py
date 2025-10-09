<<<<<<< HEAD:train/tokenizer.py
import torch, tiktoken

=======
import torch
import tiktoken
>>>>>>> 243177747acf6ef523836ff552f3064c9d5d2d7f:src/utils/tokenizer.py

tokenizer = tiktoken.get_encoding("o200k_base")

def text_to_token_ids(text, tokenizer, device="cpu"):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Adiciona o Batch
    return encoded_tensor.to(device)


def token_ids_to_text(token_ids, tokenizer, device="cpu"):
    flat = token_ids.squeeze(0) # Tira a dimensão de Batch
    return tokenizer.decode(flat.tolist())