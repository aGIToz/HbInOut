import argparse
import time
import os

import numpy as np
import torch
import random
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from data import get_dataset
import sys
import json
from net import Morpho
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import is_undirected, to_undirected

from utils import ROOT_DIR


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  model.run_pde = False
  optimizer.zero_grad()
  feat = data.x
  out = model(feat, pos_encoding)
  lf = torch.nn.functional.nll_loss
  loss = lf(out.log_softmax(dim=1)[data.train_mask], data.y.squeeze()[data.train_mask])
  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()

@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  model.run_pde = True
  feat = data.x
  logits_, accs = model(feat, pos_encoding), []
  logits = F.log_softmax(logits_, dim=1)
  lf = torch.nn.functional.nll_loss
  val_loss = lf(logits_.log_softmax(dim=1)[data.val_mask], data.y.squeeze()[data.val_mask])
  if opt['dataset'] in [ 'minesweeper', 'workers', 'questions']:
    # print("using ROC-AUC metric")
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      mask_idx = torch.where(mask)[0]
      acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(), y_score=logits[:, 1][mask_idx].cpu().numpy()).item()
      accs.append(acc)
  else:
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
  accs.append(val_loss.item())
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


def main(cmd_opt,split):
  opt = cmd_opt
  print("code refactored to only consider command args")
  dataset = get_dataset(opt, f'{ROOT_DIR}/data', True,split)
  # Select CUDA if available, otherwise CPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pos_encoding = None
  model = Morpho(opt, dataset, device).to(device)
  data = dataset.data.to(device)

  print("num of train samples: ", len(torch.nonzero(data.train_mask,as_tuple=True)[0]))
  print("num of val samples: ", len(torch.nonzero(data.val_mask,as_tuple=True)[0]))
  print("num of test samples: ", len(torch.nonzero(data.test_mask,as_tuple=True)[0]))

  parameters = [p for p in model.parameters() if p.requires_grad]
  # __import__('pdb').set_trace()
  print_model_params(model)
  # __import__('pdb').set_trace()
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0
  val_loss = np.inf 

  # opt['epoch'] = 12 # CHANGED from 701 during sweep
  opt["a"], opt["b"] = 0.5, 0.5; print("NOTE >>> CHANGED 'A' AND 'B' TO 0.5!!!")
  print(torch.sum(model.odeblock.odefunc.mlps_in[0].fc1.weight))
  for i in range(opt["n_layers"]):
    print(torch.sum(model.odeblock.odefunc.m_vectors[i].weight.view(-1)))
  # Run the requested number of epochs inclusively
  for epoch in range(1, opt['epoch'] + 1):
    start_time = time.time()
    loss = train(model, optimizer, data, pos_encoding)
    tmp_train_acc, tmp_val_acc, tmp_test_acc, tmp_val_loss = test(model, data, pos_encoding, opt)
    best_time = opt['time']
    if opt['val_acc']:
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
        best_time = opt['time']
    else:
      if tmp_val_loss < val_loss:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        val_loss = tmp_val_loss
        test_acc = tmp_test_acc
        best_time = opt['time']

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, tmp_Test{:.4f}, Best time: {:.4f}'
    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc, tmp_test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc, best_epoch, best_time))
  print(torch.sum(model.odeblock.odefunc.mlps_in[0].fc1.weight))
  for i in range(opt["n_layers"]):
    print(torch.sum(model.odeblock.odefunc.m_vectors[i].weight.view(-1)))
  # __import__('pdb').set_trace()
  return train_acc, val_acc, test_acc,opt, model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # redundant use CUDA_VISIBLE_DEVICES instead 
  parser.add_argument('--dataset', type=str, default='roman-empire')
  parser.add_argument('--self_loop_weight',type=float, default=1) #

  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=64)
  parser.add_argument('--dropout', type=float, default=0.2957472568498216)
  parser.add_argument("--batch_norm", type=int, default=1)
  parser.add_argument("--layer_norm", type=int, default=0)
  parser.add_argument('--lr', type=float, default=0.010430328236387338)
  parser.add_argument('--decay', type=float, default=0.0004122406209861293)
  parser.add_argument('--epoch', type=int, default=1500)
  parser.add_argument('--block', type=str, default="attention")
  parser.add_argument('--function', type=str, default="transconv")
  parser.add_argument('--use_mlp', type=int, default = 1) 
  parser.add_argument('--optimizer', type=str, default = "adam")

  # ODE args
  parser.add_argument('--tol_scale', type=float, default=3.694985355776542)
  parser.add_argument('--time', type=float, default=3.5)
  parser.add_argument('--max_nfe', type=int, default=2000)
  parser.add_argument('--method', type=str, default="euler")
  parser.add_argument('--step_size', type=float, default=0.500503540264418)
  parser.add_argument("--symm", type=int, default=0)
  parser.add_argument("--dynamic_p", type=int, default=0)
  parser.add_argument("--d_p", type=float, default=0)
  parser.add_argument("--initial", type=int, default=0)
  parser.add_argument("--p", type=float, default=0)
  parser.add_argument("--boundary", type=int, default=0)
  parser.add_argument("--val_acc", type=int, default=0)
  parser.add_argument("--x_new_mlp", type=int, default=1)
  parser.add_argument("--out_norm", type=str, default='1')
  parser.add_argument("--grad_norm", type=str, default='inf')
  parser.add_argument("--mlp_alpha", type=int, default=1)
  parser.add_argument("--const_alpha_x_new", type=int, default=0)
  parser.add_argument("--model", type=str, default="aniso")
  parser.add_argument("--a", type=float, default=0.5)
  parser.add_argument("--b", type=float, default=0.5)
  parser.add_argument("--c", type=float, default=1)
  parser.add_argument("--inside_dr", type=float, default=0.26922393002277634)
  parser.add_argument("--use_inner_decoder", type=int, default=1)


  parser.add_argument("--heads_", type=int, default=1) # redundant
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--pow", type=float, default=0.9)
  parser.add_argument("--eps", type=float, default=0.1)
  parser.add_argument("--ngbr_axis_sum", type=int, default=0)
  parser.add_argument("--channel_sum", type=int, default=0)
  parser.add_argument("--one_over_g", type=int, default=0)
  parser.add_argument("--L2_iso", type=int, default=1)
  parser.add_argument("--g_normalization", type=str, default="N")
  parser.add_argument("--multivariate_type", type=int, default=2)


  #new
  parser.add_argument("--n_layers", type=int, default=1)
  parser.add_argument("--ST", type=int, default=0)
  parser.add_argument("--lag", type=int, default=8)


  # redundant
  parser.add_argument('--adjoint', type=int, default=0)
  parser.add_argument('--tol_scale_adjoint', type=float, default=1.0)

  args = parser.parse_args()
  opt = vars(args)

  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

  # n_splits = 10
  n_splits = 1
  best = []
  train_acc_list = []
  val_acc_list = []
  timestr = time.strftime("%Y%m%d-%H%M%S")
  filename = "log/" + str(args.dataset) + str(args.function) + str(args.block) + str(args.time) + timestr + ".txt"
  command_args = " ".join(sys.argv)

  for split in range(n_splits):
    train_acc,val_acc, test_acc, opt_final, tmp_model = main(opt,split)
    best.append(test_acc)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
  print('Mean test accuracy: ', np.mean(np.array(best) * 100), 'std: ', np.std(np.array(best) * 100))
  print("test acc: ", best)
