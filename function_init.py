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
from utils import MaxNFEException, MLP, MLPX
from utils import get_rw_adj
from torch_geometric.utils import to_undirected


class ODEFuncInit(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device, mask=None):
    super(ODEFuncInit, self).__init__(opt, data, device)

    self.count = 0 # for mlp list
    self.run_pde = None
    self.mask = mask
    self.rel = ReLU()
    self.sigmoid = Sigmoid()
    self.tan = Tanh()
    self.device = device
    self.a = self.opt['a']
    self.b = self.opt['b']
    self.c = self.opt['c']
    self.data = data

    # this can be done in a module list!
    self.mlp = MLP(self.opt, self.device)
    self.mlp_out = MLP(self.opt, self.device)
    self.mlp_in = MLP(self.opt, self.device)
    self.mlps_out = nn.ModuleList([MLP(self.opt, self.device) for _ in range(self.opt["n_layers"])])
    self.mlps_in = nn.ModuleList([MLP(self.opt, self.device) for _ in range(self.opt["n_layers"])])

    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    edge_index_un, edge_weight_un = to_undirected(edge_index, edge_weight)
    # self.m_vector = torch.nn.Parameter(torch.ones(edge_index.shape[1])).to(self.device) #E
    # self.m_vector_test = Linear(1, edge_index.shape[1], bias=False, weight_initializer='kaiming_uniform').weight.view(-1).to(self.device)
    # __import__('pdb').set_trace()
    self.m_vector = Linear(1, edge_index.shape[1], bias=False, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.m_vectors = nn.ModuleList([Linear(1, edge_index.shape[1], bias=False, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])])

    self.beta_outs = nn.ModuleList([Linear(self.data.k_out * self.data.k_out, self.data.num_nodes, bias=False, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])])
    self.beta_ins = nn.ModuleList([Linear(self.data.k_in * self.data.k_in, self.data.num_nodes, bias=False, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])])

    self.alpha_outs = nn.ModuleList([Linear(self.data.k_out, self.data.num_nodes, bias=False, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])])
    self.alpha_ins = nn.ModuleList([Linear(self.data.k_in, self.data.num_nodes, bias=False, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])])
    # self.m_vector = self.sigmoid(self.m_vector)
    # __import__('pdb').set_trace()

    self.grad_ml1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.grad_ml2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.norm_ml1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.norm_ml2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1

    self.ho1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    # self.ho11 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.ho11 = nn.ModuleList([Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])]) #N, 1
    self.ho2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    # self.ho22 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.ho22 = nn.ModuleList([Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) for _ in range(self.opt["n_layers"])]) #N, 1

    self.per1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per3 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per31 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1

    # x_new, x_new_un
    self.per4 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per5 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per6 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.per7 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1

    self.perf = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1

    # alpha, beta, gamma
    self.pera1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.pera1_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.pera1__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.pera2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.pera2_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.pera2__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb1_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb1__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb2_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perb2__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg1 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg1_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg1__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg2 = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg2_ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
    self.perg2__ = Linear(self.opt['hidden_dim'], self.opt['hidden_dim'], bias=True, weight_initializer="kaiming_uniform").to(self.device) #N, 1
