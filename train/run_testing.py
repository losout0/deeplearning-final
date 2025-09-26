from src import GPT2ModelGQA, GPT2ModelMHA
from src import get_loaders
from src import generate_text
from src import text_to_token_ids, token_ids_to_text, tokenizer
import torch
import time
import csv
import os
import sys
from pathlib import Path
import csv
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import torch


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# --- Configuração do Treinamento ---
# --- Configuração do Treinamento ---
CONFIG = {
    # Configurações do Modelo
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 512,
    "context_length": 256,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 2,
    "dtype": torch.float32,
    "batch_size": 4,
    
    # Log e Amostra
    "start_context": "Se o jardim",
    "test_log_file": "logs/test_results.csv",

    # Pesos salvos
    "best_model_paths": {
        "gqa": "D:\Faculdade\Período 8\Deep Learning\deeplearning-final\data\GQA\model_best.pth",
        "mha": "D:\Faculdade\Período 8\Deep Learning\deeplearning-final\data\gpt-multihead\model_best.pth",
    },
}

os.makedirs("logs", exist_ok=True)
os.makedirs("/data/GQA", exist_ok=True)
os.makedirs("/data/gpt-multihead", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

# --- Funções Auxiliares ---


def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(model, data_loader, device, num_batches):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')
    num_batches = min(num_batches, len(data_loader))
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            input_batch, target_batch = next(data_iter)
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        except StopIteration:
            break
    return total_loss / num_batches if num_batches > 0 else float('nan')


def evaluate_test_model_with_metrics(model, test_loader, device, log_dir="logs"):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    os.makedirs(log_dir, exist_ok=True)

    timestamp_file = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_file_path = os.path.join(log_dir, f"test_metrics_{timestamp_file}.csv")

    with open(log_file_path, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_idx", "loss", "perplexity",
                        "F1", "Precision", "Recall"])

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader, 1):
            x, y = x.to(device), y.to(device)

            loss = calc_loss_batch(model, x, y, device)
            total_loss += loss.item()
            n_batches += 1

            logits = model(x)
            preds = torch.argmax(logits, dim=-1)

            flat_labels = y.flatten().cpu().numpy()

            all_preds.extend(flat_preds)
            all_labels.extend(flat_labels)

            f1 = f1_score(flat_labels, flat_preds,
                          average='weighted', zero_division=0)
            precision = precision_score(
                flat_labels, flat_preds, average='weighted', zero_division=0)
            recall = recall_score(flat_labels, flat_preds,
                                  average='weighted', zero_division=0)

            perplexity = float(torch.exp(loss))

            with open(log_file_path, "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([batch_idx, f"{loss.item():.4f}", f"{perplexity:.2f}",
                                f"{f1:.4f}", f"{precision:.4f}", f"{recall:.4f}"])
                f.flush()

            print(
                f"Batch {batch_idx}/{len(test_loader)} concluída | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f} | F1: {f1:.4f}")

    avg_loss = total_loss / n_batches
    perplexity_total = float(torch.exp(torch.tensor(avg_loss)))

    f1_total = f1_score(all_labels, all_preds,
                        average='weighted', zero_division=0)
    precision_total = precision_score(
        all_labels, all_preds, average='weighted', zero_division=0)
    recall_total = recall_score(
        all_labels, all_preds, average='weighted', zero_division=0)
    metrics_total = {"F1": f1_total,
                     "Precision": precision_total, "Recall": recall_total}

    print(
        f"\nAvaliação completa | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity_total:.2f} | F1: {f1_total:.4f}")

    return avg_loss, perplexity_total, metrics_total, log_file_path


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Amostra Gerada: '{decoded_text.replace(os.linesep, ' ')}'")
    model.train()

# --- Função Principal de Treinamento ---
def test_model_and_log(model, test_loader, config, device, tokenizer, start_context=None):
    model.eval()

    log_file_path = config["test_log_file"]
    os.makedirs(os.path.dirname(log_file_path),
                exist_ok=True) if os.path.dirname(log_file_path) else None

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model", "loss", "perplexity",
                            "F1", "Precision", "Recall", "duration_s"])

    start_time = time.time()

    avg_loss, perplexity, metrics = evaluate_test_model_with_metrics(
        model, test_loader, device)
    duration = time.time() - start_time

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    model_name = type(model).__name__

    with open(log_file_path, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            model_name,
            f"{avg_loss:.4f}",
            f"{perplexity:.2f}",
            f"{metrics['F1']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{duration:.2f}"
        ])
        f.flush()

    print(f"[{model_name}] Test Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | "
          f"F1: {metrics['F1']:.4f} | Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | "
          f"Duration: {duration:.2f}s")

    if start_context is not None:
        generate_and_print_sample(model, tokenizer, device, start_context)

    return avg_loss, perplexity, metrics, duration

# --- Bloco de Execução Principal ---
if __name__ == "__main__":

    print("Carregando dados...")
    train_loader, test_loader, val_loader = get_loaders(
        data_path="data/processed",  # Ajuste o caminho para seus dados
        tokenizer=tokenizer,
        max_length=CONFIG["context_length"],
        batch_sz=CONFIG["batch_size"]
    )
    print(
    f"Tamanho do conjunto de treinamento: {len(train_loader)}\nTamanho do conjunto de teste: {len(test_loader)}\nTamanho do conjunto de validação: {len(val_loader)}")
    
    model_gqa = GPT2ModelGQA(CONFIG, device=DEVICE).to(device=DEVICE)
    try:
        model_gqa.load_state_dict(torch.load(
            CONFIG["best_model_paths"]["gqa"], map_location=DEVICE))
        print("Pesos do best_model " + CONFIG["best_model_paths"]["gqa"] + " carregados com sucesso!")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo " + CONFIG["best_model_paths"]["gqa"] + " não encontrado. Certifique-se de ter baixado o modelo.")

    model_mha = GPT2ModelMHA(CONFIG, device=DEVICE).to(device=DEVICE)

    try:
        model_mha.load_state_dict(torch.load(
            CONFIG["best_model_paths"]["mha"], map_location=DEVICE))
        print(f"Pesos do best_model " +
              CONFIG["best_model_paths"]["mha"] + " carregados com sucesso!")
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo " +
                                CONFIG["best_model_paths"]["mha"] + " não encontrado. Certifique-se de ter baixado o modelo.")
    
    print("GQA")
    loss_gqa, ppl_gqa, metrics_gqa, time_gqa = test_model_and_log(
        model_gqa, test_loader, CONFIG, DEVICE, tokenizer, CONFIG["start_context"])

    print("MHA")
    loss_mha, ppl_mha, metrics_mha, time_mha = test_model_and_log(
        model_mha, test_loader, CONFIG, DEVICE, tokenizer, CONFIG["start_context"])
    
    print("Fim do script.")
