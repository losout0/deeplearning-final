import torch, time
import matplotlib.pyplot as plt
from .tokenizer import text_to_token_ids, token_ids_to_text
from .generate import generate_text
from .tokenizer import tokenizer
from matplotlib.ticker import MaxNLocator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_loss_batch_by_cross_entropy(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_by_cross_entropy(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device
            )

            total_loss += loss.item()
        else:
            break

    return total_loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    if hasattr(model, 'pos_embeddings'):
      context_size = model.pos_embeddings.weight.shape[0]
    elif hasattr(model, 'pos_encoding'):
      context_size = model.pos_encoding.pe.size(1)
    else:
      # Fallback padrão
      context_size = 256
      print(f"Usando context_size padrão: {context_size}")
    encoded = text_to_token_ids(start_context, tokenizer, device).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer, device)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer,
        state_dict, project, name, start_time
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    epochs_complete = state_dict.get("epoch", 0)
    batchs_complete = state_dict.get("batch", 0)
    accumulated_time = state_dict.get("train_time", 0)

    for epoch in range(epochs_complete, num_epochs):
        model.train()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            if epoch == epochs_complete and batch_idx < batchs_complete:
                continue

            optimizer.zero_grad()
            loss = calc_loss_batch_by_cross_entropy(
                model,
                input_batch,
                target_batch,
                device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        print("\nEXEMPLO DE GERAÇÃO:")
        generate_and_print_sample(model, tokenizer, device, start_context)

        elapsed_time = time.time() - start_time + accumulated_time

    return train_losses, val_losses, track_tokens_seen, elapsed_time


def plot_graph(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Perda no Treino")
    ax1.plot(epochs_seen, val_losses, linestyle="-.",label="Perda na Validação")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Vistos")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()


def train_model(model, optimizer, config, train_loader, val_loader):
    num_epochs = config["max_epochs"]
    start_time = time.time()
    state_dict = {}
    
    train_losses, val_loss, tokens_seen, total_train_time = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Bom dia!", tokenizer=tokenizer, state_dict=state_dict,
        start_time=start_time
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_graph(epochs_tensor, tokens_seen, train_losses, val_loss)

    return tokens_seen[-1], total_train_time