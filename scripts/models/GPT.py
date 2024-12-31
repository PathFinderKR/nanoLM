# scripts/models/GPT.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = None
    context_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    dropout: float = 0.2


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for LayerNorm.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Normalized tensor of shape (B, T, C).
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super(MultiHeadAttention, self).__init__()
        assert config.n_embed % config.n_head == 0, "n_embed must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embed // self.n_head
        self.dropout = config.dropout
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)
        self.out = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        """
        Forward pass for Multi-Head Self-Attention with internal mask generation.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()

        # Query, Key, Value
        q = self.query(x)  # (B, T, C)
        k = self.key(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)

        mask = torch.tril(torch.ones(T, T, device=x.device)).bool().unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_head, T, T)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)  # (B, n_head, T, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Final linear layer
        output = self.out(attn_output)  # (B, T, C)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass for Feed Forward network.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, config: GPTConfig):
        super(DecoderLayer, self).__init__()
        self.ln1 = LayerNorm(config.n_embed)
        self.self_attn = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.n_embed)
        self.mlp = FeedForward(config)

    def forward(self, x):
        """
        Forward pass for a single Decoder Layer with internal masking.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def _init_weights(module):
    """
    Initialize weights of the model.

    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.context_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embed)
        self.linear = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.apply(_init_weights)

    def num_parameters(self):
        """
        Get the number of trainable parameters in the model.

        Returns:
            int: The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx):
        """
        Forward pass for GPT model.

        Args:
            idx (Tensor): Input tensor of token indices of shape (B, T).

        Returns:
            Tensor: Logits of shape (B, T, vocab_size).
        """
        B, T = idx.size()
        assert T <= self.config.context_size, "Sequence length exceeds context size"

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_ids = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos_ids)  # (1, T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)  # (B, T, C)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.linear(x)  # (B, T, vocab_size)

        return logits

    def loss(self, idx, targets):
        """
        Compute cross-entropy loss between model logits and targets.

        Args:
            idx (Tensor): Input tensor of token indices of shape (B, T).
            targets (Tensor): Target tensor of token indices of shape (B, T).

        Returns:
            Tensor: Scalar loss value.
        """
        logits = self.forward(idx)  # (B, T, vocab_size)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, tokenizer, device, temperature=1.0):
        """
        Generate text using the GPT model.

        Args:
            prompt (str): The initial text prompt.
            max_new_tokens (int): The number of tokens to generate.
            tokenizer (Tokenizer): Tokenizer with encode and decode methods.
            device (torch.device): Device to perform computation on.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.

        Returns:
            str: Generated text.
        """
        self.eval()
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        generated = input_ids.tolist()[0]

        for _ in range(max_new_tokens):
            if len(generated) > self.config.context_size:
                input_ids = torch.tensor(generated[-self.config.context_size:], dtype=torch.long, device=device).unsqueeze(0)
            else:
                input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)

        return tokenizer.decode(generated)
