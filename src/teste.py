from .gpt_2.model_geral import Transformer
from .utils.tokenizer import tokenizer, text_to_token_ids
import torch

# --- Configuração do Treinamento ---
CONFIG = {
    # Configurações do Modelo
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 512,
    "context_length": 256,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 8,
    "dtype": torch.float32,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "emb_dim_moe": 64,
    "apply_rope": False,

    # Configurações do Treinamento
    "max_iterations": 50000,
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "batch_size": 4,

    # Configurações de Avaliação e Logging
    "eval_freq": 200,
    "eval_iter": 50,
    "start_context": "Se o jardim",

    # Configurações de Arquivos
    "checkpoint_save_path": "checkpoints/checkpoint_latest.pth",
    "best_model_save_path": "checkpoints/model_best.pth",
    "log_file": "logs/training_log.csv"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

model = Transformer(
    config=CONFIG,
    device=DEVICE   
).to(DEVICE)

sentence = CONFIG["start_context"]

test_gpt_model = torch.randint(
    high=CONFIG["vocab_size"],
    size=(2, CONFIG["context_length"]),
    device=DEVICE
)

forward = model(test_gpt_model)
assert forward.shape == (2, CONFIG["context_length"], CONFIG["vocab_size"])

print(forward)