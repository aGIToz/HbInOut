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
from torch.nn import  Sigmoid, ReLU, Tanh
from utils import MaxNFEException, MLP
from function_init_tow import ODEFuncInitTow
from function_operators import ODEFuncOperators

class ODEFuncOperatorsMore(ODEFuncOperators):

  def __init__(self, in_features, out_features, opt, data, device, mask=None):
    super(ODEFuncOperatorsMore, self).__init__(in_features, out_features, opt, data, device, mask)

  def get_flow_simple(self, grad):
    if self.opt["g_normalization"] == "N":
      f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * (grad), self.edge_index[0], dim=0, reduce = "sum") 
      f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degy[self.edge_index[0]],self.b)) * (grad), self.edge_index[1], dim=0, reduce = "sum")
    else: 
      f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * (grad), self.edge_index[0], dim=0, reduce = "sum") 
      f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.b)) * (grad), self.edge_index[1], dim=0, reduce = "sum") 
    return 0.5 * (f1 - f2)

  def iso_simple(self, x):
      grad = self.get_grad(x)
      if self.opt["L2_iso"]:
        grad_norm = self.grad_norm_l2(x)
        flow = self.get_flow_simple(grad*torch.pow(grad_norm[self.edge_index[0]], -self.opt["pow"])) # good for l2
      else:
        grad_norm = self.grad_norm_l1(x)
        flow = self.get_flow_simple(grad*torch.pow(self.opt["eps"] + grad_norm[self.edge_index[0]], -self.opt["pow"])) #good for l1
      return flow
