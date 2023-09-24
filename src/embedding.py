from config import *
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        self.vocab_size = VOCAB_SIZE
        self.embedding = nn.Embedding( # Embedding layer of shape -> (vocab_size, d_model)
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )
    
    def forward(self, x):
        x = self.embedding(x) # Create token embeddings
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self):
        self.d_model = D_MODEL 
        self.seq_len = MAX_SEQ_LEN
        self.dropout_p = DROPOUT_P
        self.dropout = nn.Dropout(self.dropout_p) # Dropout layer
        self.position_encodings = torch.zeros(self.seq_len, self.d_model) # Shape: (seq_len, d_model)
        self.positions = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(0) # Shape: (seq_len) -> (seq_len, 1)
        self.even_odd_i = torch.arange(0, self.d_model, 2).float() # 2i # Shape: (d_model/2)
        self.div_freqs_term = torch.pow(10000, self.even_odd_i/self.d_model) # 10000**2i/dmodel
        self.position_encodings[:, 0::2] = torch.sin(self.positions*self.div_freqs_term) # Shape: (seq_len, d_model)
        self.position_encodings[:, 1::2] = torch.cos(self.positions*self.div_freqs_term) # Shape: (seq_len, d_model)
        self.position_encodings = self.position_encodings.unsqueeze(0) # Shape: (1, seq_len, d_model)
        self.register_buffer(self.position_encodings) # to be a part of module state but not a parameter of module

    def forward(self, x):
        x = x + (self.position_encodings[:, :x.shape[1], :]) # Adding positional encodings
        x = self.dropout(x)
        return x

        