"""
utility functions
"""
import os
import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Dropout


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class MaxNFEException(Exception): pass


class MLP(torch.nn.Module):
    def __init__(self, opt, device):
        super(MLP, self).__init__()
        self.opt = opt
        self.device = device
        self.dropout_prob = self.opt["inside_dr"]
        
        # Define the first hidden layer
        self.fc1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
        self.dropout1 = Dropout(p=self.dropout_prob)
        # Define the second hidden layer
        self.fc2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
        self.fc3 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
        self.dropout2 = Dropout(p=self.dropout_prob)

    def forward(self, x):
        # Apply the first hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Apply the second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # x = (self.fc3(x))
        # x = self.dropout2(x)
        
        return x

class MLPX(torch.nn.Module):
    def __init__(self, opt, device):
        super(MLPX, self).__init__()
        self.opt = opt
        self.device = device
        
        # Define the first hidden layer
        self.fc1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
        # Define the second hidden layer
        self.fc2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1

    def forward(self, x):
        # Apply the first hidden layer
        x = F.relu(self.fc1(x))
        
        # Apply the second hidden layer
        x = F.relu(self.fc2(x))
        return x


def get_front(labels, seed_mask, p=0, hidden_dim=None):
  new_front = []

  def modify_mask(seed_mask, p):
    if p == 0:
      return seed_mask 
    elif p == 1:
      return torch.zeros_(seed_mask, dtype=torch.bool)
    else:
      random_mask = torch.rand(seed_mask.size()) < p
      modified_mask = seed_mask.clone()
      modified_mask[random_mask] = False
      return modified_mask

  modified_mask = modify_mask(seed_mask,p)

  if hidden_dim == None:
    L = labels.max() + 1
  else:
    L = hidden_dim 
    for i in range(0, L):
      mask = (labels == i)
      seeds = torch.where((mask*modified_mask) == True,1,-1)
      new_front.append(seeds)
  Front = torch.vstack(new_front).type(torch.float32)
  return Front


def get_front_dynamic(labels, seed_mask, p=0, hidden_dim=None, training=True):
  new_front = []

  def modify_mask(seed_mask, p):
    if p == 0:
      return seed_mask 
    elif p == 1:
      return torch.zeros_(seed_mask, dtype=torch.bool)
    else:
      modified_mask = F.dropout(seed_mask.float(), p=p, training=training)
      modified_mask = modified_mask.bool()
      return modified_mask

  modified_mask = modify_mask(seed_mask,p)

  if hidden_dim == None:
    L = labels.max() + 1
  else:
    L = hidden_dim 
    for i in range(0, L):
      mask = (labels == i)
      seeds = torch.where((mask*modified_mask) == True,1,-1)
      new_front.append(seeds)
  Front = torch.vstack(new_front).type(torch.float32)
  return Front


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight


#Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val
