import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
from data import get_dataset
from base_classes import ODEFunc
from torch_scatter import scatter
import math
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.dense.linear import Linear
from torch.nn import  Sigmoid, ReLU
from utils import MaxNFEException, MLP
from function_init import ODEFuncInit


class ODEFuncInitTow(ODEFuncInit):

  def __init__(self, in_features, out_features, opt, data, device, mask=None):
    super(ODEFuncInitTow, self).__init__(in_features, out_features, opt, data, device, mask)


  def init_before_tow(self, x):

      if self.opt['const_alpha_x_new']:
        self.x_new = 1
        self.x_new_un = 1
        self.Alpha =1
        self.Alpha_ =1
        self.Alpha__ =1
        self.Beta =1
        self.Beta_ =1
        self.Beta__ =1
        self.Gamma =1
        self.Gamma_ =1
        self.Gamma__ =1
        return 0

      # x.shape = N, F
      self.src = x[self.edge_index[0]] # E, F
      self.dst_k = x[self.edge_index[1]]  # E, F
      if self.opt['x_new_mlp'] == 0:
        self.x_new = self.mlp_list[9](self.dst_k - self.src)
      else:
        if self.opt["one_over_g"]:
          self.x_new = F.relu(self.per5(F.relu(self.per4(1.0/(self.opt["eps"] + torch.abs(self.dst_k - self.src))))))
        else:
          self.x_new = F.relu(self.per5(F.relu(self.per4(torch.abs(self.dst_k - self.src)))))
      self.x_new_un = F.relu(self.per7(F.relu(self.per6(x[self.edge_index_un[1]] - x[self.edge_index_un[0]]))))
      
      if self.opt['mlp_alpha']:
        self.Alpha = F.relu(self.pera2(F.relu(self.pera1(x))))
        self.Alpha_ = F.relu(self.pera2_(F.relu(self.pera1_(x))))
        self.Alpha__ = F.relu(self.pera2__(F.relu(self.pera1__(x))))
        self.Beta = F.relu(self.perb2(F.relu(self.perb1(x))))
        self.Beta_ = F.relu(self.perb2_(F.relu(self.perb1_(x))))
        self.Beta__ = F.relu(self.perb2__(F.relu(self.perb1__(x))))
        self.Gamma = F.relu(self.perg2(F.relu(self.perg1(x))))
        self.Gamma_ = F.relu(self.perg2_(F.relu(self.perg1_(x))))
        self.Gamma__ = F.relu(self.perg2__(F.relu(self.perg1__(x))))

      return 0
