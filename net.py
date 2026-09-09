import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from utils import *


# Define the GNN model.
class Morpho(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(Morpho, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.graph = dataset.data
    self.device = device
    self.front_initial = get_front(self.graph.y, self.graph.train_mask, self.opt['p'], self.opt['hidden_dim']).to(device)
    self.mask = torch.where(torch.sum(self.front_initial.T,dim=1)==1, False, True)
    self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor, mask=self.mask).to(device)
    self.run_pde = False

    if opt["use_mlp"]:
      self.reset_parameters()

    self.output_normalization =nn.LayerNorm(opt['hidden_dim'])

  def reset_parameters(self):
    torch.nn.init.xavier_normal_(self.m11.weight, gain=1.414)
    torch.nn.init.xavier_normal_(self.m12.weight, gain=1.414)

  # this is the key here!
  def forward(self, x, pos_encoding=None):
    x = self.m1(x)
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    if self.opt['use_mlp']:
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    else:
      x = F.dropout(self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    if self.opt['layer_norm']:
      x = F.normalize(x, p=2, dim=1)

    if not self.run_pde:
      self.odeblock.donot_run_pde()
      self.odeblock.set_x0(x)
    else:
      self.odeblock.run_pde()
      if self.opt['boundary'] or self.opt['initial']:
        if self.opt['dynamic_p'] == 0:
          fzero = (x.T * self.mask.view(1,-1) + self.front_initial).T
        else:
          self.front_initial = get_front_dynamic(self.graph.y, self.graph.train_mask, self.opt['d_p'], self.opt['hidden_dim'], training=self.training).to(self.device)
          self.mask = torch.where(torch.sum(self.front_initial.T,dim=1)==1, False, True)
          fzero = (x.T * self.mask.view(1,-1) + self.front_initial).T

        self.odeblock.set_x0(fzero)
      else:
        self.odeblock.set_x0(x)

    z = self.odeblock(x)
    z = self.m2(z)

    return z
