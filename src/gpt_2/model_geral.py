import torch


class RoPE(torch.nn.Module):
    def apply_rope(x, cos, sin):
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2:]

        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, context_length, num_heads, num_kv_groups, apply_rope, bias=False):
        super().__init__()

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.apply_rope = apply_rope
        self.num_kv_groups = num_kv_groups
        self.head_dim = dim_out // num_heads
        self.group_size = self.num_heads // self.num_kv_groups

        assert dim_out % num_heads == 0
        assert num_heads % num_kv_groups == 0

        self.wq = torch.nn.Linear(dim_in, dim_out, bias)
        self.wk = torch.nn.Linear(dim_in, num_kv_groups * self.head_dim, bias)
        self.wv = torch.nn.Linear(dim_in, num_kv_groups * self.head_dim, bias)
        self.wo_proj = torch.nn.Linear(dim_out, dim_out, bias)
        
        self.rope = RoPE()

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch, num_tokens,self.num_kv_groups,self.head_dim)
        
        #queries = self.rope.apply_rope(queries, cos, sin)
        #keys = self.rope.apply_rope(keys, cos, sin)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2).repeat_interleave(self.group_size, dim=1)
        values = values.transpose(1, 2).repeat_interleave(self.group_size,dim=1)

        attention = queries @ keys.transpose(2, 3)
        attention.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        causal_attention = torch.softmax(attention / keys.shape[-1]**0.5,dim=-1)
        assert keys.shape[-1] == self.head_dim

        context_vec = causal_attention @ values
        context_vec = context_vec.transpose(1, 2)

        concat = context_vec.contiguous().view(batch, num_tokens, self.dim_out)
        concat = self.wo_proj(concat)

        return concat


class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim, epsilon = 1e-5):
        super().__init__()
        self.gama = torch.nn.Parameter(torch.ones(embedding_dim))
        self.beta = torch.nn.Parameter(torch.zeros(embedding_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        return x * self.gama + self.beta


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class MoEFeedForward(torch.nn.Module):
    def __init__(self, num_experts, num_experts_per_token, emb_dim_moe, emb_dim, bias):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_token
        self.num_experts_per_tok = emb_dim_moe
        self.num_experts = num_experts
        self.emb_dim = emb_dim

        self.gate = torch.nn.Linear(emb_dim, num_experts, bias=bias)
        self.fc1 = torch.nn.ModuleList([torch.nn.Linear(emb_dim,emb_dim_moe,bias=bias) for _ in range(num_experts)])
        self.fc2 = torch.nn.ModuleList([torch.nn.Linear(emb_dim,emb_dim_moe,bias=bias) for _ in range(num_experts)])
        self.fc3 = torch.nn.ModuleList([torch.nn.Linear(emb_dim_moe,emb_dim,bias=bias) for _ in range(num_experts)])


    def forward(self, x):
        scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any(): continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0: continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self.emb_dim)


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim, bias=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                embedding_dim,
                4 * embedding_dim,
                bias=bias
            ),
            GELU(),
            torch.nn.Linear(
                4 * embedding_dim,
                embedding_dim,
                bias=bias
            )
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlockGQA(torch.nn.Module):
    def __init__(self, context_length, dim_in, dim_out, num_heads, num_kv_groups, apply_rope, num_experts, num_experts_per_token, emb_dim_moe, bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.emb_dim_moe = emb_dim_moe
        self.num_experts_per_token = num_experts_per_token
        self.exists_moe = True if num_experts > 0 else False

        self.grouped_query_attention = GroupedQueryAttention(
            dim_in=dim_in,
            dim_out=dim_out,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            context_length=context_length,
            apply_rope=apply_rope,
            bias=bias
        )

        if self.exists_moe:
            self.feed_forward = MoEFeedForward(
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                emb_dim_moe=emb_dim_moe,
                emb_dim=dim_in,
                bias=bias
            )
        else:
            self.feed_forward = FeedForward(
                embedding_dim=dim_in,
                bias=bias
            )

        self.norm1 = LayerNorm(
            embedding_dim=dim_in
        )

        self.norm2 = LayerNorm(
            embedding_dim=dim_in
        )


    def forward(self, x):
        residual_connection_attention = x
        norm1 = self.norm1(x)
        attention_masked = self.grouped_query_attention(norm1)
        add1 = attention_masked + residual_connection_attention

        residual_connetion_forward = add1
        norm2 = self.norm2(add1)
        forward = self.feed_forward(norm2)
        add2 = forward + residual_connetion_forward
        
        return add2
    

class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device

        self.embeddings = torch.nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["embedding_dim"]
        )

        self.pos_embeddings = torch.nn.Embedding(
            num_embeddings=config["context_length"],
            embedding_dim=config["embedding_dim"]
        )

        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlockGQA(
                context_length=config["context_length"],
                dim_in=config["embedding_dim"],
                dim_out=config["embedding_dim"],
                num_heads=config["num_heads"],
                num_kv_groups=config["num_kv_groups"],
                apply_rope=config["apply_rope"],
                num_experts=config["num_experts"],
                num_experts_per_token=config["num_experts_per_token"],
                emb_dim_moe=config["emb_dim_moe"],
                bias=config["bias"],
                
            )
            for _ in range(config["num_layers"])
        ])

        self.final_norm = LayerNorm(
            embedding_dim=config["embedding_dim"],
            epsilon=1e-5
        )

        self.out_head = torch.nn.Linear(
            config["embedding_dim"],
            config["vocab_size"],
            bias=config["bias"]
        )


    def forward(self, x):
        batch_size, context_length = x.shape
        input_emb = self.embeddings(x)
        
        if not self.apply_rope:
            pos_emb = self.pos_embeddings(torch.arange(context_length, device=self.device))
            input_emb = input_emb + pos_emb

        result_transformer_blocks = self.transformer_blocks(input_emb)
        norm = self.final_norm(result_transformer_blocks)
        logits = self.out_head(norm)

        return logits