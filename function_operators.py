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


class ODEFuncOperators(ODEFuncInitTow):

  def __init__(self, in_features, out_features, opt, data, device, mask=None):
    super(ODEFuncOperators, self).__init__(in_features, out_features, opt, data, device, mask)

  def get_grad(self, x):
      if self.opt["g_normalization"] == "N":
        grad = torch.sqrt(self.mean_attention.view(-1,1))*((x[self.edge_index[1]]/torch.pow(self.degy[self.edge_index[1]], self.a)) - (x[self.edge_index[0]]/torch.pow(self.degx[self.edge_index[0]], self.b)))
      else : 
        grad = torch.sqrt(self.mean_attention.view(-1,1))*((x[self.edge_index[1]]/torch.pow(self.degx[self.edge_index[0]], self.a)) - (x[self.edge_index[0]]/torch.pow(self.degx[self.edge_index[0]], self.b)))
      return grad

  def get_grad2(self, x, m_vector):
      # need to have mulitple m and b vector, so really multi-attention step.
      # grad = torch.sqrt(self.mean_attention.view(-1,1))*((x[self.edge_index[1]]/torch.pow(self.degy[self.edge_index[1]], self.a)) - (x[self.edge_index[0]]/torch.pow(self.degx[self.edge_index[0]], self.b)))
      grad = torch.sqrt(m_vector.view(-1,1))*((x[self.edge_index[1]]/torch.pow(self.degy[self.edge_index[1]], self.a)) - (x[self.edge_index[0]]/torch.pow(self.degx[self.edge_index[0]], self.b)))
      # __import__('pdb').set_trace()
      return grad

  def get_h(self, x, a_matrix):
      # need to have mulitple m and b vector, so really multi-attention step.
      a_matrix = 1.0
      grad = torch.sqrt(a_matrix.view(-1,1))*((x[self.edge_index[1]]/torch.pow(self.degy[self.edge_index[1]], self.a)) - (x[self.edge_index[0]]/torch.pow(self.degx[self.edge_index[0]], self.b)))
      # __import__('pdb').set_trace()
      return grad

  def get_flow(self, grad):
      if self.opt["g_normalization"] == "N":
        if self.opt["multivariate_type"] == 2:
          f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * (grad), self.edge_index[0], dim=0, reduce = "sum") 
          f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degy[self.edge_index[0]],self.b)) * (grad), self.edge_index[1], dim=0, reduce = "sum") 
        else:
          f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * self.ho11(grad), self.edge_index[0], dim=0, reduce = "sum") 
          f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degy[self.edge_index[0]],self.b)) * self.ho22(grad), self.edge_index[1], dim=0, reduce = "sum") 
      else: 
        if self.opt["multivariate_type"] == 2:
          f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * (grad), self.edge_index[0], dim=0, reduce = "sum") 
          f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.b)) * (grad), self.edge_index[1], dim=0, reduce = "sum") 
        else:
          f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * self.ho11(grad), self.edge_index[0], dim=0, reduce = "sum") 
          f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.b)) * self.ho22(grad), self.edge_index[1], dim=0, reduce = "sum") 
      if self.opt["multivariate_type"] == 2:
        # return 0.5 * (self.ho11[0](f1) - self.ho22[0](f2))
        return (self.ho11[0](f1)),  (self.ho22[0](f2))
      else:
        return 0.5 * (f1 - f2)

  def get_flow2(self, grad,i):
      f1  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degx[self.edge_index[0]],self.a)) * (grad), self.edge_index[0], dim=0, reduce = "sum") 
      f2  = scatter((torch.sqrt(self.mean_attention.view(-1,1))/torch.pow(self.degy[self.edge_index[0]],self.b)) * (grad), self.edge_index[1], dim=0, reduce = "sum") 
      # __import__('pdb').set_trace()
      # print("here - i am")
      return 0.5 * self.ho11[i](f1), 0.5 * self.ho22[i](f2)
      # return 0.5 * f1, 0.5 * f2

  def grad_norm_l1(self, x):
      grad = self.get_grad(x)
      if self.opt["ngbr_axis_sum"]:
          normi = scatter( torch.abs(grad), self.edge_index[0], dim=0, reduce = "sum") #outgoing
      else:
          normi = scatter( torch.abs(grad), self.edge_index[0], dim=0, reduce = "max") #good
      if self.opt["channel_sum"]:
          normi = (torch.mm(normi, torch.abs(self.per1.weight))) # good
          # this is bad: normi = F.relu(torch.mm(normi, (self.per1.weight)))
      else:
          normi = torch.max(normi, dim=1)[0] #good
          normi = normi.view(-1,1)
      norm = normi
      return norm

  def grad_norm_l2(self, x):
      grad = self.get_grad(x)
      if self.opt["channel_sum"]:
          normi = (scatter( torch.pow(grad,2), self.edge_index[0], dim=0, reduce = "sum")+ self.opt["eps"]) #outgoing
          normi = torch.sqrt(torch.mm(normi, torch.abs(self.per1.weight))) # good
      else:
          normi = torch.sqrt(scatter( torch.pow(grad,2), self.edge_index[0], dim=0, reduce = "sum")+self.opt["eps"]) #outgoing
          normi = torch.max(normi, dim=1)[0] #good
          normi = normi.view(-1,1)
        
      norm =  normi
      return norm

  def iso(self, x):
      grad = self.get_grad(x)
      if self.opt["L2_iso"]:
        grad_norm = self.grad_norm_l2(x)
        flow = self.get_flow(grad*torch.pow(grad_norm[self.edge_index[0]], -self.opt["pow"])) # good for l2
      else:
        grad_norm = self.grad_norm_l1(x)
        flow = self.get_flow(grad*torch.pow(self.opt["eps"] + grad_norm[self.edge_index[0]], -self.opt["pow"])) #good for l1
      return flow

  def pseudo_hessian2(self, x, alpha_out=1, alpha_in=1, beta_out=1, beta_in=1, cross_beta_out=1, cross_beta_in=1):
      """
      - clean code later  - user proper names
      """
      # __import__('pdb').set_trace()
      beta_out = beta_out * (1.0 - torch.eye(self.data.k_out, device= self.device))
      beta_in = beta_in * (1.0 - torch.eye(self.data.k_in, device= self.device))
      # this is for learnable betas in the code
      cross_beta_out = cross_beta_out * (1.0 - torch.eye(self.data.k_out, device= self.device))
      cross_beta_in = cross_beta_in * (1.0 - torch.eye(self.data.k_in, device= self.device))

      beta_out = self.sigmoid(beta_out.unsqueeze(-1))
      beta_in = self.sigmoid(beta_in.unsqueeze(-1))
      # cross_beta_out = self.sigmoid(cross_beta_out.unsqueeze(-1))
      # cross_beta_in = self.sigmoid(cross_beta_in.unsqueeze(-1))

      alpha_out = self.sigmoid(alpha_out.unsqueeze(-1))
      alpha_in = self.sigmoid(alpha_in.unsqueeze(-1))

      x_out = (x[self.out_ngbr] - x.unsqueeze(1))
      x_in = (x.unsqueeze(1) - x[self.in_ngbr])

      h_out = x_out
      h_in = x_in

      pair_prod_out = h_out.unsqueeze(2) + h_out.unsqueeze(1)
      pair_prod_in = h_in.unsqueeze(2) + h_in.unsqueeze(1)

      # __import__('pdb').set_trace()

      pair_sum_out = (beta_out * pair_prod_out).sum(dim=2) 
      pair_sum_in = (beta_in * pair_prod_in).sum(dim=2) 

      # pair_sum_out = (pair_prod_out).sum(dim=2) 
      # pair_sum_in = (pair_prod_in).sum(dim=2) 

      outflow = (alpha_out * x_out) + pair_sum_out
      inflow = (alpha_in * x_in) + pair_sum_in

      # outflow = (x_out) + pair_sum_out
      # inflow = (x_in) + pair_sum_in
      # __import__('pdb').set_trace()
      return 0.5 * self.ho11[0](outflow.sum(dim=1)), 0.5 * self.ho22[0](inflow.sum(dim=1))
      # return 0.5 * self.ho11(torch.max(outflow, dim=1).values), 0.5 * self.ho22(torch.max(inflow, dim=1).values)
      # return 0.5 * self.ho11(torch.max(outflow, dim=1).values), 0.5 * self.ho22(torch.min(inflow, dim=1).values)
      # return 0.5 * self.ho11(outflow.max(dim=1)), 0.5 * self.ho22(inflow.max(dim=1))
      # return 0.5 * self.ho11(torch.abs(outflow.sum(dim=1))), 0.5 * self.ho22(torch.abs(inflow.sum(dim=1)))
