from config import *
from attention import MultiHeadAttention
from ffn import FeedForward
from layer_norm import ResidualConnection, LayerNorm
import torch.nn as nn
import torch

class DecoderBlock:

    def __init__(self):

        '''
        A Single Decoder block that can be stacked multiple times to form the decoder which uses three main blocks: Self Attention, Cross Attention, Feed Forward connected with some Residual Connections

        Returns:
        dec_block_out - Decoder Block Output
        '''

        self.d_model = D_MODEL # Embedding dimension
        self.num_heads = NUM_HEADS # Number of heads
        self.attn_dropout = ATTN_DROPOUT # Attention dropout value
        self.ff_dropout = FF_DROPOUT # Feed forward dropout value
        self.res_dropout = RES_DROPOUT # Residual connection dropout value
        self.self_attention = MultiHeadAttention(self.d_model, self.num_heads, self.attn_dropout) # Self Attention layer
        self.cross_attention = MultiHeadAttention(self.d_model, self.num_heads, self.attn_dropout) # Cross Attention
        self.feed_forward = FeedForward(self.d_model, self.ff_dropout) # Feedforward layer
        self.res_connection1 = ResidualConnection(self.res_dropout) # Residual connection layer1
        self.res_connection2 = ResidualConnection(self.res_dropout) # Residual connection layer2

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, src_attn_mask: torch.Tensor, tgt_attn_mask: torch.Tensor):
        self_attn_out = self.res_connection1(x, lambda x: self.attention(x, x, x, tgt_attn_mask)) # Self Attention and Normalized output
        cross_attn_out = self.res_connection2(self_attn_out, lambda self_attn_out: self.attention(self_attn_out, encoder_out, encoder_out, src_attn_mask)) # Cross Attention and Normalized output
        dec_block_out = self.res_connection3(cross_attn_out, self.feed_forward(cross_attn_out)) # Feed Forward output normalized and produces decoder output
        return dec_block_out
    
class Decoder:

    def __init__(self, d_model: int, num_layers: int):

        '''
        Transformer decoder which has a stack of decoder blocks joined together
        
        Args:
        d_model: Embedding_dimension
        num_layers: Number of layers in the stack

        Returns:
        decoder_out: Decoder output
        '''

        self.layers = nn.ModuleList([DecoderBlock() for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model, EPS)

    def forward(self, x:torch.Tensor, encoder_out: torch.Tensor, src_attn_mask: torch.Tensor, tgt_attn_mask: torch.Tensor):

        for layer in self.layers:
            x = layer(x, encoder_out, src_attn_mask, tgt_attn_mask)
        decoder_out = self.layer_norm(x)
        return decoder_out