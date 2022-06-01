import os
import sys

# sys.path.append('/home/tuzixini/zhangyunxing/Public/')
from Graph_Embedding.functions.data_process import get_weight_initial, GraphConstruction, l2_distance

functions_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(functions_path + '/Public/Graph_Embedding/functions/')
import torch
import numpy as np


class BAGE(torch.nn.Module):
    
    def __init__(self, d_0, d_1, d_2):
        super(BAGE, self).__init__()
        
        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        
        self.gconv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2),
        )
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)
    
    def encoder(self, adjacency_convolution, H_0):
        H_1 = self.gconv1(torch.matmul(adjacency_convolution, H_0))
        H_2 = self.gconv2(torch.matmul(adjacency_convolution, H_1))
        return H_2
    
    def graph_decoder(self, H_2):
        graph_re = GraphConstruction(H_2)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction
    
    def forward(self, adjacency_convolution, H_0):
        latent_representation = self.encoder(adjacency_convolution, H_0)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return graph_reconstruction, latent_representation


class VBAGE(torch.nn.Module):
    
    def __init__(self, d_0, d_1, d_2):
        super(VBAGE, self).__init__()
        
        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        
        self.gconv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.gconv2_mean[0].weight.data = get_weight_initial(d_2, d_1)
        
        self.gconv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.gconv2_std[0].weight.data = get_weight_initial(d_2, d_1)
    
    def encoder(self, Adjacency_Convolution, H_0):
        H_1 = self.gconv1((Adjacency_Convolution.mm(H_0)))
        H_2_mean = self.gconv2_mean(torch.matmul(Adjacency_Convolution, H_1))
        H_2_std = self.gconv2_std(torch.matmul(Adjacency_Convolution, H_1))
        return H_2_mean, H_2_std
    
    def reparametrization(self, H_2_mean, H_2_std):
        eps = torch.randn_like(H_2_std)
        # H_2_std 并不是方差，H_2_std = log(σ)
        std = torch.exp(H_2_std)
        # torch.randn 生成正态分布
        latent_representation = eps * std + H_2_mean
        return latent_representation
    
    # 解码隐变量
    def graph_decoder(self, latent_representation):
        graph_re = GraphConstruction(latent_representation)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction
    
    def forward(self, adjacency_convolution, H_0):
        H_2_mean, H_2_std = self.encoder(adjacency_convolution, H_0)
        latent_representation = self.reparametrization(H_2_mean, H_2_std)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return latent_representation, graph_reconstruction, H_2_mean, H_2_std


def adjacency_update(X, Adjacency, M_min=3, M_max=50, is_symmetry=True, is_round=False):
    ## Input, X: n * d; adjacency: n *n
    ## Count the number of non-zero elements
    
    index_nozero = []
    for i in range(X.shape[0]):
        index_nozero.append(torch.nonzero(Adjacency[i]).shape[0])
    index_nozero = torch.FloatTensor(index_nozero)
    M_gaussian = (torch.normal(mean=index_nozero, std=0.1)).ceil()
    M_gaussian = M_gaussian.type(torch.LongTensor)
    M_gaussian = M_gaussian.clamp(min=M_min, max=M_max)
    print("M_Gaussian = {}".format(M_gaussian))
    
    #### Calculate the Adjacency matrix
    X = np.array(X)
    n = X.shape[0]
    D = torch.Tensor(l2_distance(X, X))
    _, idx = torch.sort(D)
    S = torch.zeros(n, n)
    for i in range(n):
        id = torch.LongTensor(idx[i][1:M_gaussian[i] + 1 + 1])
        di = D[i][id]
        S[i][id] = (torch.Tensor(di[M_gaussian[i]].repeat(di.shape[0])) - di) \
                   / (M_gaussian[i] * di[M_gaussian[i]] - torch.sum(di[0:M_gaussian[i]]) + 1e-4)
    
    if is_symmetry:
        adjacency_new = (S + S.T) / 2
    else:
        adjacency_new = S
    
    ##### Normalize the adjacency matrix to {0,1}
    if is_round:
        Index_0 = torch.nonzero(adjacency_new)
        for i in range(Index_0.shape[0]):
            row = Index_0[i][0]
            col = Index_0[i][1]
            adjacency_new[row][col] = 1
    return torch.Tensor(adjacency_new)
