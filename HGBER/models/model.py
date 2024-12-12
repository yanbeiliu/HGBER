import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.nn.parameter import Parameter

class Encoder(torch.nn.Module):
    def __init__(self, settings):
        super(Encoder, self).__init__()
        self.settings = settings

        in_channels = settings['in_dim']
        out_channels = settings['latent_dim']

        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if settings['model'] in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif settings['model'] in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.settings['model'] == 'GAE':
            return self.conv2(x, edge_index)
        elif self.settings['model'] == 'VGAE':
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class Net_autoencoder(nn.Module):
    def __init__(self, settings):
        super(Net_autoencoder, self).__init__()
        self.model_type = settings['model']
        if self.model_type == 'GAE':
            self.net = GAE(encoder=Encoder(settings), decoder=None)
        if self.model_type == 'VGAE':
            self.net = VGAE(encoder=Encoder(settings), decoder=None)
    def forward(self, x, edge_index, edge_weight=None):
        z = self.net.encode(x, edge_index)
        return z

    def re_loss(self, z, train_pos_edge_index):
        if self.model_type == 'GAE':
            l = (self.net.recon_loss(z, train_pos_edge_index))
        if self.model_type == 'VGAE':
            l = self.net.recon_loss(z, train_pos_edge_index) + (1 / z.size(0))*self.net.kl_loss()

        return l

class Net_dg(nn.Module):
    def __init__(self, settings):
        super(Net_dg, self).__init__()
        in_channels = settings['H_dim']
        out_channels = settings['latent_dim']
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 2*out_channels),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.BatchNorm1d(2*out_channels),
            nn.Linear(2*out_channels, 2*out_channels),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.BatchNorm1d(2*out_channels),
            nn.Linear(2*out_channels, out_channels)
        )

    def forward(self, H):
        z_dg = self.mlp(H)
        return z_dg

class H(nn.Module):
    def __init__(self, settings):
        super(H, self).__init__()
        self.input_H_layer = Parameter(torch.Tensor(settings['num_node'], settings['H_dim']))
        torch.nn.init.xavier_normal_(self.input_H_layer.data)
    def forward(self):
        return self.input_H_layer


class clustering(nn.Module):
    def __init__(self, settings):
        super(clustering, self).__init__()
        self.clustering_layer = Parameter(torch.Tensor(settings['num_class'], settings['H_dim']))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)
    def forward(self):
        return self.clustering_layer
