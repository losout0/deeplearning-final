import torch

def generate_text(model, idx, max_new_tokens, context_size=50, reset=False):
    model.eval()
    reset = False

    try:
        for i in range((max_new_tokens + 1) if reset else max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond, reset) if reset else model(idx_cond)
                if logits == None: continue

            reset = False

            logits = logits[:, -1, :]
            probas = torch.nn.functional.softmax(logits, dim=-1)
            idx_next_token = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next_token), dim=1)

        return idx

    except Exception as e:
        print(e)
        return idx