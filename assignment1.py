import torch
import torch.nn as nn
import math

d_model = 64
n_head = 8
dropout = 0.1

batch_size = 2
seq_len = 16


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, d_model=64, n_head=8, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.residual_dropout = nn.Dropout(dropout)

        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the multi-head masked self-attention layer.
        You should not use network modules other than what defined in the __init__ function.

        Input & output shape: (batch_size, sequence_length, d_model)
        """
        # --- TODO: start of your code ---
        batch_size, seq_len, d_model = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (batch_size, n_head, seq_len, d_head)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len) == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # output projection
        y = self.residual_dropout(self.o_proj(y))
        return y

        # --- TODO: end of your code ---
        raise NotImplementedError


# causal_self_attention = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)

# Load the model
causal_self_attention = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)
causal_self_attention.load_state_dict(torch.load("causal_self_attention.pt"))


# Test the model
# shape: (batch_size, seq_len, d_model)
x = torch.load("x.pt")
# x = torch.rand(2, 16, d_model)

y = causal_self_attention(x)
y_expected = torch.load("y.pt")

assert y.shape == y_expected.shape, f"Expected shape: {y_expected.shape}, but got: {y.shape}"
assert torch.sum(y == y_expected) > 0.78 * batch_size * seq_len * d_model, "The output is incorrect."

print("The output is correct.")
