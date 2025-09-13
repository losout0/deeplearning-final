import torch, time
from .tokenizer import tokenizer
from .model import GPT2ModelGQA
from .train import train_model
from .loaders import get_loaders
from .data import get_data
from .train import calc_loss_batch_by_cross_entropy


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
    "batch_size": 10,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "file_save": "model.pth"
}


train_data, test_data, val_data = get_data()
train_loader, test_loader, val_loader = get_loaders(
    train_data=train_data, 
    test_data=test_data, 
    val_data=val_data, 
    tokenizer=tokenizer,
    max_length=config["context_length"],
    batch_sz=config["batch_size"]
)


model = GPT2ModelGQA(config, device).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
model.train()


try:
    model.load_state_dict(torch.load(config["file_save"], map_location=device))
    print("Pesos carregados com sucesso!")
except FileNotFoundError:
    print("Nenhum modelo salvo encontrado, iniciando do zero...")


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

model.eval()

def compute_perplexity(model, data_loader, device):
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            loss = calc_loss_batch_by_cross_entropy(model, x, y, device)

            print(f"Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss_partial = total_loss / n_batches
                print(f"  Média parcial até aqui: {avg_loss_partial:.4f}")

    avg_loss = total_loss / n_batches
    print(f"Loss média final no dataset de teste: {avg_loss:.4f}")

    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexidade calculada: {perplexity.item():.2f}")

    return perplexity.item()


compute_perplexity(model, test_loader, device)


try:
    torch.save(model.state_dict(), config["file_save"])
    print("Modelo salvo")
except Exception as e:
    pass