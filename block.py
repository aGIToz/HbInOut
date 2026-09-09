import torch
from torch_scatter import scatter
# from function_transformer_convection import SpGraphTransAttentionLayer
import torch.nn as nn
from torch.nn import  Sigmoid
from base_classes import ODEblock
from utils import get_rw_adj
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_undirected


class ODEBlock(ODEblock):
  def __init__(self, odefunc, opt, data, device, t=torch.tensor([0, 1]), gamma=0.5, mask=None):
    super(ODEBlock, self).__init__(odefunc, opt, data, device, t)

    self.mask = mask
    self.device = device
    self.rel = Sigmoid()
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device, mask=self.mask)
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.in_ngbr = data.in_ngbr.to(device)
    self.odefunc.out_ngbr = data.out_ngbr.to(device)
    # __import__('pdb').set_trace()
    self.odefunc.edge_weight = edge_weight.to(device)
    self.odefunc.edge_index_un, self.odefunc.edge_weight_un = to_undirected(self.odefunc.edge_index, self.odefunc.edge_weight)

    # the code moved from the line 60 (after t = self.t.type_as(x)
    self.odefunc.mean_attention = (torch.ones(self.odefunc.edge_index.shape[1])).to(self.device) #E
    self.odefunc.mean_attention_un = nn.Parameter(torch.ones(self.odefunc.edge_index_un.shape[1])).to(self.device) #E
    self.odefunc.degx = scatter(self.odefunc.mean_attention.view(-1,1),self.odefunc.edge_index[0], dim=0, reduce="add")
    self.odefunc.degy = scatter(self.odefunc.mean_attention.view(-1,1),self.odefunc.edge_index[1], dim=0, reduce="add")
    self.odefunc.deg = scatter(self.odefunc.mean_attention_un.view(-1,1),self.odefunc.edge_index_un[0], dim=0, reduce="add")
    if self.opt['symm']:
        self.odefunc.mean_attention = self.odefunc.mean_attention_un
        self.odefunc.degx = self.odefunc.deg
        self.odefunc.degy = self.odefunc.deg
        self.odefunc.edge_index = self.odefunc.edge_index_un

    if self.opt["adjoint"]:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()

  def run_pde(self):
    self.odefunc.run_pde = True

  def donot_run_pde(self):
    self.odefunc.run_pde = False

  def forward(self, x):
    t = self.t.type_as(x)

    integrator = self.train_integrator 

    func = self.odefunc

    state =  x

    if self.opt["method"] == "org":
        state_dt = integrator( func, state, t,  atol=self.atol, rtol=self.rtol)
    else:
        state_dt = integrator( func, state, t, method=self.opt['method'], options={'step_size': self.opt['step_size']}, atol=self.atol, rtol=self.rtol)

    z = state_dt[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) + ')'
