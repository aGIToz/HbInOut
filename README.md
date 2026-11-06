# HbInOut (Hessian-based Influx Outflux)

Source code for HbInOut model in our published approach: 
Amitoz Azad, Rajjat Dadwal, and Elena Demidova. *"Learning Generalized Hessian on Graphs for Continuous Graph Learning."*
In: Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2026).
LNAI. Springer, 2026. DOI:https://doi.org/10.1007/978-3-032-37657-2_27. In Press. 

## Overview

<!-- TODO: expand this section with a fuller explanation of HbInOut (e.g. abstract, key equations, figure from the paper). -->

HbInOut is a continuous-depth graph neural network for node classification. Node features are encoded, evolved over a time interval by a neural ODE defined on the graph (solved with [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)), and decoded into class scores.

The ODE's vector field is a learnable **generalized Hessian operator on the graph**, split into two flux terms:

- **Outflux** — computed over each node's *out*-neighbours.
- **Influx** — computed over each node's *in*-neighbours.

Each flux combines first-order differences between a node and its neighbours (weighted by learnable per-node coefficients $\alpha$) with second-order terms over pairs of neighbours (weighted by learnable per-node coefficients $\beta$). A map $\Phi$ is applied to each flux, and the node features evolve as

$$\frac{d\mathbf{x}}{dt} = \Phi(\text{outflux}) - \Phi(\text{influx})$$


Two variants are provided:

| Variant | `--model` | $\Phi$ |
|---|---|---|
| HbInOut ($\Phi = \mathrm{MLP}$) | `hessian_mlp` | learnable MLP |
| HbInOut ($\Phi = I$) | `hessian_I` | identity |

```mermaid
flowchart LR
    X["Node features"] --> E["Encoder<br/>(MLP + residual blocks)"]
    E --> ODE["Neural ODE on the graph<br/>dx/dt = Φ(outflux) − Φ(influx)"]
    ODE --> D["Decoder"]
    D --> Y["Class scores"]
```
## Requirements

Key libraries:

| Library | Version |
|---|---|
| `torch` | 2.6.0 |
| `torch-geometric` | 2.6.1 |
| `torch_scatter` | 2.1.2 |
| `torchdiffeq` | 0.2.5 |

The code also imports `torch_sparse`, `ogb`, `scikit-learn`, `scipy` and `numpy`.

## Installation

```bash
git clone https://github.com/aGIToz/HbInOut.git
cd HbInOut

# PyTorch (pick the build matching your CUDA version: https://pytorch.org/get-started/locally/)
pip install torch==2.6.0

# PyTorch Geometric and its compiled extensions
# replace ${CUDA} with cpu, cu118, cu124 or cu126
pip install torch-geometric==2.6.1
pip install torch_scatter==2.1.2 torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+${CUDA}.html

# Remaining dependencies
pip install torchdiffeq==0.2.5 ogb scikit-learn scipy numpy
```

## Instructions
The datasets with their split configurations are too large to be uploaded to this repository. They can be obtained from the source code of references [31, 24, 28] in the paper.

Place each dataset as a `.npz` file under `Datasets/data/`, relative to the directory you run from:

```
python run_.py --model=hessian_mlp --dataset=roman_empire_40_30_30 # \Phi=mlp
python run_.py --model=hessian_I --dataset=roman_empire_40_30_30 # \Phi=I
```

## Disclaimer
This software is published only to support reproducibility in research - not for commercial use or placement on the market.

## Affiliations
The source code is being published in affiliation with the University of Bonn, Germany (https://www.uni-bonn.de/de).

## Acknowledgements
This work was partially funded by the Deutsche Forschungsgemeinschaft (DFG), Germany (“RUSHMORE”, 547610252), 
mFund, the Federal Ministry of Transport (BMV), Germany (“MoToRes”, 01F2271C) and the Lamarr Institute for Machine Learning and 
Artificial Intelligence (https://lamarr-institute.org/).
 
