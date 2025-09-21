import torch
import time
import csv
import os
import sys
from itertools import cycle
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import text_to_token_ids, token_ids_to_text, tokenizer
from src import generate_text
from src import get_loaders
from src import GPT2ModelGQA 

# --- Configuração do Treinamento ---
CONFIG = {
    # Configurações do Modelo
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 384,
    "context_length": 128,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 2,
    "dtype": torch.float32,

    # Configurações do Treinamento
    "max_iterations": 500,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "batch_size": 4,
    
    # Configurações de Avaliação e Logging
    "eval_freq": 50,
    "eval_iter": 20,
    "start_context": "Se o jardim",
    
    # Configurações de Arquivos
    "checkpoint_save_path": "checkpoints/checkpoint_latest.pth", 
    "best_model_save_path": "model_best.pth",
    "log_file": "training_log.csv"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

# --- Funções Auxiliares ---

def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(model, data_loader, device, num_batches):
    total_loss = 0.0
    if len(data_loader) == 0: return float('nan')
    num_batches = min(num_batches, len(data_loader))
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            input_batch, target_batch = next(data_iter)
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        except StopIteration: break
    return total_loss / num_batches if num_batches > 0 else float('nan')

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Amostra Gerada: '{decoded_text.replace(os.linesep, ' ')}'")
    model.train()

# --- Função Principal de Treinamento ---

def train_model_by_iterations(model, optimizer, train_loader, val_loader, config, device):
    start_time = time.time()
    log_file_path = config["log_file"]
    
    # Prepara o arquivo CSV
    log_header = ["iteration", "train_loss", "val_loss", "tokens_seen", "learning_rate", "timestamp"]
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    print("Iniciando o treinamento por iterações...")
    
    # Variável para rastrear a melhor perda de validação
    best_val_loss = float('inf') 

    train_data_iter = cycle(train_loader)
    tokens_seen = 0
    
    for step in range(config["max_iterations"]):
        input_batch, target_batch = next(train_data_iter)
        
        optimizer.zero_grad()
        loss = calc_loss_batch(model, input_batch, target_batch, device)
        loss.backward()
        optimizer.step()
        
        tokens_seen += input_batch.numel()
        
        is_last_step = (step == config["max_iterations"] - 1)
        if step % config["eval_freq"] == 0 or is_last_step:
            train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, config["eval_iter"])
            
            print(
                f"[Iteração {step:05d}/{config['max_iterations']}] | "
                f"Perda Treino: {train_loss:.3f} | "
                f"Perda Validação: {val_loss:.3f}"
            )
            
            # --- LÓGICA DE SALVAMENTO ---
            # 1. Salva o checkpoint mais recente
            torch.save(model.state_dict(), config["checkpoint_save_path"])
            
            # 2. Verifica se este é o melhor modelo e o salva
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config["best_model_save_path"])
                print(f"  -> Nova melhor perda de validação: {best_val_loss:.3f}. Modelo salvo em '{config['best_model_save_path']}'")
            # ---------------------------

            # Salva os resultados no arquivo CSV
            current_lr = optimizer.param_groups[0]['lr']
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            log_data = [step, f"{train_loss:.4f}", f"{val_loss:.4f}", tokens_seen, f"{current_lr:.6f}", timestamp]
            with open(log_file_path, "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)
                
            generate_and_print_sample(model, tokenizer, device, config["start_context"])
            print("-" * 50)

    total_time = time.time() - start_time
    print("Treinamento concluído!")
    print(f"Tempo total de treinamento: {total_time:.2f} segundos.")
    print(f"Resultados de log salvos em: '{log_file_path}'")

# --- Bloco de Execução Principal ---

if __name__ == "__main__":
    
    print("Carregando dados...")
    train_loader, test_loader, val_loader = get_loaders(
        data_path="data/processed", # Ajuste o caminho para seus dados
        tokenizer=tokenizer,
        max_length=CONFIG["context_length"],
        batch_sz=CONFIG["batch_size"]
    )
    
    print("Inicializando modelo...")
    model = GPT2ModelGQA(CONFIG, device=DEVICE).to(device=DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    
    # Tenta carregar o checkpoint mais recente para continuar o treinamento
    try:
        model.load_state_dict(torch.load(CONFIG["checkpoint_save_path"], map_location=DEVICE))
        print(f"Pesos do checkpoint '{CONFIG['checkpoint_save_path']}' carregados com sucesso!")
    except FileNotFoundError:
        print("Nenhum checkpoint encontrado, iniciando do zero.")
        
    model.train()
    train_model_by_iterations(model, optimizer, train_loader, val_loader, CONFIG, DEVICE)

    print("Fim do script.")