import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.nhid = nhid
        self.mlp = GCNlayer(nfeat, self.nhid, dropout)
        self.classifier = MyGCNConv(self.nhid, nclass)

    def forward(self, x,adj,drop=True):
        x = self.mlp(x,adj,drop)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls,adj,drop)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits

##    
class GCNlayer(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(GCNlayer, self).__init__()
        self.fc1 = MyGCNConv(input_dim, hid_dim)
        self.fc2 = MyGCNConv(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x,adj, drop=True):
        x = self.fc1(x,adj,drop)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x,adj,drop)
        return x
    
class MyGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def forward(self, x, adj, drop=True):
        x = torch.matmul(x, self.weight)+self.bias
        if drop ==False:
            adj = adj + torch.eye(adj.shape[0]).cuda()
            deg = torch.sum(adj, dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            norm = torch.diag(deg_inv_sqrt) @ adj @ torch.diag(deg_inv_sqrt)
            out = torch.matmul(norm, x)
            return out
        return x
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


