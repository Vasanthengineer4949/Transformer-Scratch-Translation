import torch.nn as nn
import torch

class LayerNorm(nn.Module):

    def __init__(self, eps: float) -> None:

        '''
        Normalizes the values layerwise by the formula: X-mean/(std + eps)
        Here eps -> epsilon is a very small value preventing zero division.
        Here also two other parameters known as alpha and beta are introduced where alpha is multiplicative and beta is additive so that it introduces some deviations in the data creating some variety since by this normalization all values will fall under 0 to 1. 
        alpha and beta are learnable parameters

        Returns:
        layer_norm_out - Normalized layer value output
        '''

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Since both alpha and beta are scalar in nature but learnable parameters
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True) # Finding mean across the last dimension while retaining the dimension
        std = x.std(dim=-1, keepdim=True) # Finding mean across the last dimension while retaining the dimension
        x_normalized = (x - mean)/(std + self.eps) # X - mean / std + eps
        layer_norm_out = self.alpha * x_normalized + self.beta # Variance creation by alpha * x_normalized + beta
        return layer_norm_out

