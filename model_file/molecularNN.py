import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class Mish(nn.Module):
    """Mish activation function."""

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MolecularNN(nn.Module):
    """
    A flexible Molecular Neural Network for regression or classification.

    Args:
        args (dict): A dictionary containing model parameters.
    """

    def __init__(self, args):
        super(MolecularNN, self).__init__()
        param = args

        fingerprint_embed_in = param['fingerprint_embed_in']
        fingerprint_embed_out = param['fingerprint_embed_out']
        feature_size = param['feature_size']
        n_layers = param['n_layers']
        n_output = param['n_output']
        dropout = param['dropout']
        self.task = param['task']
        self.output_embedding = param.get('output_embedding', False) # Option to output embedding
        self.sigmoid = nn.Sigmoid()
        self.fingerprint_encoder = nn.Linear(fingerprint_embed_in, fingerprint_embed_out)

        if feature_size != 0:
            self.fc_embs = nn.Linear(feature_size, fingerprint_embed_out)
            self.grucell = nn.GRUCell(fingerprint_embed_out, fingerprint_embed_out, bias=True)
            self.embs = nn.Linear(fingerprint_embed_out, fingerprint_embed_out)

        d_model = fingerprint_embed_out
        if n_layers == 1:
            self.FNN = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(Mish())
                self.proj.append(ScaleNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.FNN = nn.Sequential(*self.proj)

    def forward(self, fingerprint, output_embedding=False,feature=None):
        """
        Forward pass of the MolecularNN.

        Args:
            fingerprint (torch.Tensor): Input fingerprint vector.
            feature (torch.Tensor, optional): Additional feature vector. Defaults to None.

        Returns:
           torch.Tensor: Output tensor with the specified number of output dimension or embedding.
        """

        if feature is not None:
            fingerprint_embed = F.relu(self.fingerprint_encoder(fingerprint))
            features_embed = F.sigmoid(self.fc_embs(feature))
            all_features = self.grucell(fingerprint_embed, features_embed)
            all_features = F.sigmoid(self.embs(all_features))
            
        else:
            all_features = F.relu(self.fingerprint_encoder(fingerprint))
        
        if output_embedding:
            return all_features
        
        output = self.FNN(all_features)

        if (self.task == 'cls') & (not self.training):
           output = self.sigmoid(output)

        return output


    def clear(self, args):
        """Reinitializes model parameters."""
        self.__init__(args)