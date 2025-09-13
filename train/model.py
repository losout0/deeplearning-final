import torch


class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, context_length, num_heads, num_kv_groups, bias=False):
        super().__init__()

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = dim_out // num_heads
        self.group_size = self.num_heads // self.num_kv_groups

        assert dim_out % num_heads == 0
        assert num_heads % num_kv_groups == 0

        self.wq = torch.nn.Linear(dim_in, dim_out, bias)
        self.wk = torch.nn.Linear(dim_in, num_kv_groups * self.head_dim, bias)
        self.wv = torch.nn.Linear(dim_in, num_kv_groups * self.head_dim, bias)
        self.wo_proj = torch.nn.Linear(dim_out, dim_out, bias)

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
    def __init__(self, context_length, dim_in, dim_out, num_heads, num_kv_groups, bias=False):
        super().__init__()

        self.grouped_query_attention = GroupedQueryAttention(
            dim_in=dim_in,
            dim_out=dim_out,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            context_length=context_length,
            bias=bias
        )

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


class GPT2ModelGQA(torch.nn.Module):
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
                bias=config["bias"]
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
        tok_emb = self.embeddings(x)
        pos_emb = self.pos_embeddings(
            torch.arange(context_length, device=self.device)
        )
        input_emb = tok_emb + pos_emb

        result_transformer_blocks = self.transformer_blocks(input_emb)
        norm = self.final_norm(result_transformer_blocks)
        logits = self.out_head(norm)

        return logits