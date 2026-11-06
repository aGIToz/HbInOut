# HbInOut (Hessian-based Influx Outflux)

Source code for HbInOut model in our published approach: 
Amitoz Azad, Rajjat Dadwal, and Elena Demidova. *"Learning Generalized Hessian on Graphs for Continuous Graph Learning."*
In: Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2026).
LNAI. Springer, 2026. DOI:https://doi.org/10.1007/978-3-032-37657-2_27. In Press. 

## Project Description
- add later
- explain the model hbinout

## Requirements
- Key libraries required torchdiffeq==0.2.5, torch-geometric==2.6.1, torch_scatter==2.1.2, and torch==2.6.0

## Instructions
- The datasets with split configurations are huge - and couldn't be uploaded - but could be obtained from source code of [31,24,28]
Run:
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
 
