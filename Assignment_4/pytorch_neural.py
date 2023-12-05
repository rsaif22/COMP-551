import torch
from functools import reduce

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', a, b)
    res = res.view(-1, res.shape[1]*res.shape[2])
    return res

def torch_bin(x, cut_points, temperature=0.1):
    D = cut_points.shape[0]
    W = torch.linspace(1.0, D + 1.0, D + 1).view(1, -1)
    cut_points = cut_points.sort()[0]  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.tensor([0.0]), -cut_points]), 0)
    h = torch.matmul(x, W) + b
    res = torch.softmax(h / temperature, dim=1)
    return res

def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    leaf = reduce(torch_kron_prod, 
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)
