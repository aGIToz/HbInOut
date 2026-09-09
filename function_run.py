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
from function_operators_more import ODEFuncOperatorsMore
from utils import get_rw_adj
from torch_geometric.utils import to_undirected


class ODEFuncRun(ODEFuncOperatorsMore):

  def __init__(self, in_features, out_features, opt, data, device, mask=None):
    super(ODEFuncRun, self).__init__(in_features, out_features, opt, data, device, mask)
    # edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                      fill_value=opt['self_loop_weight'],
    #                                      num_nodes=data.num_nodes,
    #                                      dtype=data.x.dtype)
    # edge_index_un, edge_weight_un = to_undirected(edge_index, edge_weight)
    # self.m_vector = Linear(1, edge_index.shape[1], bias=False, weight_initializer='kaiming_uniform').weight.view(-1).to(self.device)

  def forward(self, t, x):

    # if self.nfe > self.opt["max_nfe"]:
    #   raise MaxNFEException
    self.nfe += 1

    self.init_before_tow(x)
    #ax4 =  self.laplacian_un(x) 

    if self.opt["model"] == "pure":
      f = self.Alpha * (torch.mm(self.grad_p_norm(x), (self.per1.weight))) -  (self.Beta) *  (torch.mm(self.grad_m_norm(x),(self.per2.weight)))

    # new models
    elif self.opt["model"] == "iso":
      f = self.iso(x)
    elif self.opt["model"] == "iso_simple":
      f = self.iso_simple(x)
    elif self.opt["model"] == "pseudo_hessian1":
      # okay let's think about the addition later pls.
      for i in range(self.opt["n_layers"]):
        m_vector = self.sigmoid(self.m_vectors[i].weight.view(-1)) #done
        # m_vector = (self.m_vectors[i].weight.view(-1)) #done
        # m_vector = self.rel(self.m_vector.weight.view(-1))
        # m_vector = self.tan(self.m_vector.weight.view(-1))
        # grad = self.get_grad2(x, m_vector)
        # out_f, in_f = self.get_flow2(grad)
        out_f, in_f = self.pseudo_hessian1(x,m_vector,i)
        # __import__('pdb').set_trace()
        out_f = self.mlps_out[i](out_f)
        in_f = self.mlps_in[i](in_f)
        x = out_f - in_f

      f = x
    elif self.opt["model"] == "hessian_I":
      for i in range(self.opt["n_layers"]):
        beta_out = self.beta_outs[i].weight.view(self.data.num_nodes, self.data.k_out, self.data.k_out)
        beta_in = self.beta_ins[i].weight.view(self.data.num_nodes, self.data.k_in, self.data.k_in)

        alpha_out = self.alpha_outs[i].weight
        alpha_in = self.alpha_ins[i].weight

        out_f, in_f = self.pseudo_hessian2(x, alpha_out, alpha_in, beta_out, beta_in)
        x = out_f - in_f
      f = x
    elif self.opt["model"] == "hessian_mlp":
      for i in range(self.opt["n_layers"]):
        beta_out = self.beta_outs[i].weight.view(self.data.num_nodes, self.data.k_out, self.data.k_out)
        beta_in = self.beta_ins[i].weight.view(self.data.num_nodes, self.data.k_in, self.data.k_in)
        # cross_beta_out = self.beta_outs[1].weight.view(self.data.num_nodes, self.data.k_out, self.data.k_out)
        # cross_beta_in = self.beta_ins[1].weight.view(self.data.num_nodes, self.data.k_in, self.data.k_in)

        alpha_out = self.alpha_outs[i].weight
        alpha_in = self.alpha_ins[i].weight

        # x = self.sigmoid(x)
        # out_f, in_f = self.pseudo_hessian2(x, alpha_out, alpha_in, beta_out, beta_in, cross_beta_out, cross_beta_in)
        out_f, in_f = self.pseudo_hessian2(x, alpha_out, alpha_in, beta_out, beta_in)
        # out_f, in_f = self.pseudo_hessian3(x, alpha_out, alpha_in, beta_out, beta_in)
        # __import__('pdb').set_trace()
        # out_f2, in_f2 = self.pseudo_hessian1(x)
        out_f = self.mlps_out[i](out_f)
        in_f = self.mlps_in[i](in_f)
        x = out_f - in_f
      f = x

    else:
       f = 0

    if self.opt["use_inner_decoder"]: f = self.mlp(f)
    if not self.run_pde:
      return f
    else:
      if self.opt['boundary']:
        return self.mask.view(-1,1) * f
      else:
        return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
