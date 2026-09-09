import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class MyOwnDataset(InMemoryDataset):
  def __init__(self, root, name, transform=None, pre_transform=None):
    super().__init__(None, transform, pre_transform)


def get_dataset(opt: dict, data_dir, use_lcc: bool = False, split=0) -> InMemoryDataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  dataset = MyOwnDataset(path, name=ds)
  data = np.load(os.path.join('./Datasets/data', f'{ds.replace("-", "_")}.npz'))
  node_features = torch.tensor(data['node_features'])
  labels = torch.tensor(data['node_labels'])
  edges = torch.tensor(data['edges'])
  edges = edges.T
  N = data["node_features"].shape[0]

  train_masks = torch.tensor(data['train_masks'])
  val_masks = torch.tensor(data['val_masks'])
  test_masks = torch.tensor(data['test_masks'])

  train_mask = train_masks[split,:]
  val_mask = val_masks[split, :]
  test_mask = test_masks[split, :]
  # __import__('pdb').set_trace()

  src = edges[0].tolist()
  dst = edges[1].tolist()

  out_lists = [[] for _ in range(N)]
  in_lists = [[] for _ in range(N)]


  for s, d in zip(src, dst):
    out_lists[s].append(d)
    in_lists[d].append(s)

  k_out = max((len(neis) for neis in out_lists), default=0)
  k_in  = max((len(neis) for neis in in_lists),  default=0)

  out_neighbors = torch.empty((N, k_out), dtype=torch.long)
  in_neighbors  = torch.empty((N, k_in),  dtype=torch.long)
  
  for i in range(N):
    o = out_lists[i]
    o = o + [i] * (k_out - len(o))  # pad with self
    assert len(o) == k_out
    out_neighbors[i] = torch.tensor(o, dtype=torch.long)
  
    inn = in_lists[i]
    inn = inn + [i] * (k_in - len(inn))  # pad with self
    assert len(inn) == k_in
    in_neighbors[i] = torch.tensor(inn, dtype=torch.long)

  data = Data(
    x=node_features,
    edge_index=torch.LongTensor(edges),
    y=labels,
    train_mask=train_mask,
    test_mask=test_mask,
    val_mask=val_mask,
    in_ngbr = in_neighbors,
    out_ngbr = out_neighbors,
    k_out = k_out,
    k_in = k_in,
    num_nodes = N
  )

  # __import__('pdb').set_trace()

  dataset.data = data
  return dataset
