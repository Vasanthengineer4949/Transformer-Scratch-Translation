from config import *
from attention import MultiHeadAttention
from ffn import FeedForward
from layer_norm import ResidualConnection, LayerNorm
import torch.nn as nn
import torch

class EncoderBlock:

    def __init__(self):

        '''
        A Single Encoder block that can be stacked multiple times to form the encoder

        Returns:
        enc_block_out - Encoder Block Output
        '''

        self.d_model = D_MODEL # Embedding dimension
        self.num_heads = NUM_HEADS # Number of heads
        self.attn_dropout = ATTN_DROPOUT # Attention dropout value
        self.ff_dropout = FF_DROPOUT # Feed forward dropout value
        self.res_dropout = RES_DROPOUT # Residual connection dropout value
        self.attention = MultiHeadAttention(self.d_model, self.num_heads, self.attn_dropout) # Attention layer
        self.feed_forward = FeedForward(self.d_model, self.ff_dropout) # Feedforward layer
        self.res_connection1 = ResidualConnection(self.res_dropout) # Residual connection layer1
        self.res_connection2 = ResidualConnection(self.res_dropout) # Residual connection layer2

    def forward(self, x, src_attn_mask):
        attn_out = self.res_connection1(x, lambda x: self.attention(x, x, x, src_attn_mask)) # Attention and Normalized output
        enc_block_out = self.res_connection2(attn_out, self.feed_forward(attn_out)) # Feed Forward output normalized and produces encoder output
        return enc_block_out
    
class Encoder:

    def __init__(self, d_model: int, layers: nn.ModuleList):

        '''
        Transformer encoder which has a stack of encoder blocks joined together
        
        Returns:
        encoder_out: Encoder output
        '''