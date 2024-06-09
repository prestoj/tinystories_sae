import torch
import torch.nn as nn
import math

class SAE(nn.Module):
    def __init__(self, d_model, d_encoder, num_active_features):
        super(SAE, self).__init__()
        self.d_model = d_model
        self.d_encoder = d_encoder

        # self.pre_bias = nn.Parameter(torch.randn(d_model) * math.sqrt(1 / (d_model)))
        # self.encoder_bias = nn.Parameter(torch.randn(d_encoder) * math.sqrt(1 / num_active_features))
        # self.decoder_weights = nn.Parameter(torch.randn(d_encoder, d_model) * math.sqrt(1 / d_model))
        # self.encoder_weights = nn.Parameter(self.decoder_weights.clone().detach().T)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.encoder_bias = nn.Parameter(torch.zeros(d_encoder))
        self.decoder_weights = nn.Parameter(torch.randn(d_encoder, d_model) * math.sqrt(1 / d_model))
        self.encoder_weights = nn.Parameter(self.decoder_weights.clone().detach().T)

    def encode(self, x, top_k):
        encoded = x - self.pre_bias
        encoded = torch.matmul(encoded, self.encoder_weights) + self.encoder_bias
        top_values, encoded_indices = torch.topk(encoded, top_k, dim=1)
        mask = torch.zeros_like(encoded)
        mask.scatter_(1, encoded_indices, 1)
        masked_encoded = encoded * mask
        return masked_encoded, encoded_indices

    def decode(self, masked_encoded):
        decoded = torch.matmul(masked_encoded, self.decoder_weights) + self.pre_bias
        return decoded

    def forward(self, x, top_k):
        masked_encoded, encoded_indices = self.encode(x, top_k)
        decoded = self.decode(masked_encoded)
        return decoded, encoded_indices