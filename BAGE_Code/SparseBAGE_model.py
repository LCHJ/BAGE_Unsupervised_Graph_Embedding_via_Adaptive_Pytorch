# -*- coding:utf-8 -*-
# @Time : 2022/5/18 22:41
# @Author: LCHJ
# @File : sparseGAE.py
import sys

from Graph_Embedding.functions.data_process import l2_distance

sys.path.append('./..')
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj
from torch_geometric.utils import (add_self_loops, negative_sampling, remove_self_loops)
from torch_sparse import SparseTensor


# functions_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
# sys.path.append(functions_path + '/Public/Graph_Embedding/functions/')

EPS = 1e-15
MAX_LOGSTD = 10


class SparseBAGE(torch.nn.Module):

    def __init__(self, in_channels: int = 128, hidden_channels: int = 256, out_channels: int = 40,
                 num_classes: int = 40, num_layers: int = 3, dropout: float = 0.0, isSoftmax: bool = False):
        super(SparseBAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.liner = nn.Linear(out_channels, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=None)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.isSoftmax = isSoftmax

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.liner.reset_parameters()

    def forward(self, x: Tensor, adj_t: Adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        if self.isSoftmax:
            pred = self.liner(x)
            pred = self.log_softmax(pred)
            return pred, x
        else:
            return x, x


def gen_neg_edge_index(edge_index, num_nodes, num_neg=1):
    pos_edge_index, _ = remove_self_loops(edge_index)
    pos_edge_index, _ = add_self_loops(edge_index)
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes, num_neg_samples=num_neg * edge_index.shape[1])
    return neg_edge_index


def decoder(z: Tensor, edge_index: Adj, sigmoid=True):
    adj_re_value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    # adj_re_pos = torch.sparse.FloatTensor(edge_index, value, torch.Size([num_nodes, num_nodes])).coalesce()
    # from torch_sparse import SparseTensor
    # loop = 100
    # step = num_nodes // loop
    # z_sparse = SparseTensor.from_dense(z)
    # adj_re = SparseTensor.eye(loop, loop)
    # for k in range(0, loop):
    #     adj_re[k * step: (k + 1) * step] = z_sparse[k * step: (k + 1) * step].matmul(z_sparse.t())

    # return torch.sigmoid(adj_re.values())

    # adj_re = torch.matmul(z, z.t())
    # adj_re = torch.sigmoid(adj_re)
    return torch.sigmoid(adj_re_value) if sigmoid else adj_re_value


# loss_1 = mse_loss(graph_reconstruction, adjacency_matrix)  # / features.shape[0]
def loss_MSE(graph_reconstruction, adjacency_matrix):
    loss_1 = torch.pow((graph_reconstruction - adjacency_matrix).coalesce().values(), 2).sum()
    return loss_1


def recon_loss(z: Tensor, edge_index: Adj, neg_edge_index: Adj, num_pos=4):
    adj_re_value_pos = decoder(z, edge_index)
    adj_re_value_neg = decoder(z, neg_edge_index)
    pos_loss = -torch.log(adj_re_value_pos + EPS).mean()
    neg_loss = -torch.log(1 - adj_re_value_neg + EPS).mean()
    return num_pos * pos_loss + neg_loss


# loss_2 = torch.trace(Z.T.mm(laplacian_convolution_kernel).mm(Z))
def loss_Laplacian(Z, laplacian_convolution_kernel):
    Z_sparse = SparseTensor.from_dense(Z)
    loss_2 = torch.trace((Z_sparse.t()).matmul(laplacian_convolution_kernel).matmul(Z_sparse).to_dense())
    return loss_2


def adjacency_update(X, Adjacency, M_min=3, M_max=50, is_symmetry=True, is_round=False):
    # Input, X: n * d; adjacency: n *n
    # Count the number of non-zero elements

    index_nozero = []
    for i in range(X.shape[0]):
        index_nozero.append(torch.nonzero(Adjacency[i]).shape[0])
    index_nozero = torch.FloatTensor(index_nozero)
    M_gaussian = (torch.normal(mean=index_nozero, std=0.1)).ceil()
    M_gaussian = M_gaussian.type(torch.LongTensor)
    M_gaussian = M_gaussian.clamp(min=M_min, max=M_max)
    print("M_Gaussian = {}".format(M_gaussian))

    # Calculate the Adjacency matrix
    X = np.array(X)
    n = X.shape[0]
    D = torch.Tensor(l2_distance(X, X))
    _, idx = torch.sort(D)
    S = torch.zeros(n, n)
    for i in range(n):
        id = torch.LongTensor(idx[i][1:M_gaussian[i] + 1 + 1])
        di = D[i][id]
        S[i][id] = (torch.Tensor(di[M_gaussian[i]].repeat(di.shape[0])) - di) / (
                M_gaussian[i] * di[M_gaussian[i]] - torch.sum(di[0:M_gaussian[i]]) + 1e-4)

    if is_symmetry:
        adjacency_new = (S + S.T) / 2
    else:
        adjacency_new = S

    # Normalize the adjacency matrix to {0,1}
    if is_round:
        Index_0 = torch.nonzero(adjacency_new)
        for i in range(Index_0.shape[0]):
            row = Index_0[i][0]
            col = Index_0[i][1]
            adjacency_new[row][col] = 1
    return torch.Tensor(adjacency_new)
