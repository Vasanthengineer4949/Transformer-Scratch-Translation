import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, attn_dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.num_heads = num_heads # Number of heads
        self.head_dim = d_model // num_heads # Dimension of vector seen by each head
        self.wq = nn.Linear(d_model, d_model, bias=False) # Wq
        self.wk = nn.Linear(d_model, d_model, bias=False) # Wk
        self.wv = nn.Linear(d_model, d_model, bias=False) # Wv
        self.wo = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(attn_dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]
        # Just apply the formula from the paper
        # (bs, h, seq_len, head_dim) --> (bs, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (bs, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (bs, h, seq_len, seq_len) --> (bs, h, seq_len, head_dim)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.wq(q) # (bs, seq_len, d_model)
        key = self.wk(k) # (bs, seq_len, d_model)
        value = self.wv(v) # (bs, seq_len, d_model)

        # (bs, seq_len, d_model) --> (bs, seq_len, h, head_dim) --> (bs, h, seq_len, head_dim)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (bs, h, seq_len, head_dim) --> (bs, seq_len, h, head_dim) --> (bs, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.head_dim)

        # Multiply by Wo
        # (bs, seq_len, d_model)  
        return self.w_o(x)