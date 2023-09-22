from config import *
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        self.vocab_size = VOCAB_SIZE
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )
    
    def forward(self, x):
        embeddings = self.embedding(x)
        scaled_embeddings = embeddings*math.sqrt(self.d_model)
        return scaled_embeddings
    
class PositionalEncoding(nn.Module):

    def __init__(self):
        self.d_model = D_MODEL

        