import torch, time
from .tokenizer import tokenizer
from .model import GPT2ModelGQA
from .train import train_model
from .loaders import get_loaders
from .data import get_data

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = {
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 512,
    "context_length": 250,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 2,
    "max_epochs": 1,
    "dtype": torch.float32,
    "batch_size": 10
}


train_data, test_data, val_data = get_data()

train_loader, test_loader, val_loader = get_loaders(
    train_data=train_data, 
    test_data=test_data, 
    val_data=val_data, 
    max_length=config["context_length"], 
    batch_sz=config["batch_size"]
)


model = GPT2ModelGQA(config, device).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

params = sum(p.numel() for p in model.parameters())
params_gpt2 = params - sum(p.numel() for p in model.out_head.parameters())
print(f"Número de parâmetros: {params_gpt2:,}")

torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
tokens_processed = train_model(model=model, optimizer=optimizer, config=config, train_loader=train_loader, val_loader=val_loader)
end_time = time.time()

elapsed = end_time - start_time
tokens_per_sec = tokens_processed / elapsed
max_memory = torch.cuda.max_memory_allocated(device) / (1024**2) # MB

print(f"Tempo total: {elapsed:.2f} s")
print(f"Tokens/s: {tokens_per_sec:.2f}")
print(f"Memória máxima: {max_memory:.2f} MB")

try:
    model.save("model.pth")
except Exception as e:
    pass