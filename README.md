# RWR-GAE
Code for the paper ["Random Walk Regularized Graph Auto Encoder"](https://arxiv.org/pdf/1908.04003.pdf)


The base code is a PyTorch implementation of the Variational Graph Auto-Encoder model described in the paper:
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on or refers to https://github.com/tkipf/gae, https://github.com/tkipf/pygcn and https://github.com/vmasrani/gae_in_pytorch.

### Requirements
- Python 3
- PyTorch 0.4 

### To train a model run the following command
```bash
cd gae
python train.py --model="gcn_ae" --dataset-str="cora" --dw=1 --epochs=200 --walk-length=5 --window-size=3 --number-walks=5 --lr_dw=0.01
```
- Supported models are "gcn_vae" and "gcn_ae"
- Supported datasets are "cora" and "citeseer"
- dw, whether to use regularization or not (0: no regularization, 1: yes)
- if dw = 0, then all the remaining params are useless
- refer to _gae/train.py_ for other program arguments

### Results on CORA test set
Link Prediction results:

Model | ROC | AP
---|---|---
GAE | 0.91 | 0.92
VGAE | 0.914 | 0.926
GAE (our impl) | 0.91430 | 0.92585
VGAE (our impl) | 0.921715 | 0.927751
ARGE | 0.924 | 0.932
ARVGE | 0.924 | 0.926
DW-GAE | 0.924 | 0.918
DW-VGAE | 0.926 | 0.918

Clustering results:

Model | Acc | NMI | F1 | Precision | ARI
---|---|---|---|---|---
GAE | 0.596 | 0.429 | 0.595 | 0.596 | 0.347
VGAE | 0.609 | 0.436 | 0.609 | 0.609 | 0.346
GAE (our impl) | 0.526 | 0.42 | 0.508 | 0.530 | 0.308
VGAE (our impl) | 0.590 | 0.445 | 0.563 | 0.578 | 0.351
ARGE | 0.640 | 0.449 | 0.619 | 0.646 | 0.352
ARVGE | 0.638 | 0.450 | 0.627 | 0.624 | 0.374
DW-GAE | 0.669 | **0.464** | 0.618 | 0.629 | **0.417**
DW-VGAE | **0.685** | 0.455 | **0.668** | **0.685** | **0.417**

### Results on Citeseer test set
Link Prediction results:

Model | ROC | AP
---|---|---
GAE | 0.895 | 0.899
VGAE | 0.908 | 0.92
ARGE | 0.932 | 0.919
ARVGE | 0.924 | 0.93
DW-GAE | 0.921 | 0.915
DW-VGAE | 0.913 | 0.908

Clustering results:

Model | Acc | NMI | F1 | Precision | ARI
---|---|---|---|---|---
GAE | 0.408 | 0.176 | 0.372 | 0.418 | 0.124
VGAE | 0.344 | 0.156 | 0.308 | 0.349 | 0.093
ARGE | 0.573 | **0.350** | 0.546 | 0.573 | 0.341
ARVGE | 0.544 | 0.261 | 0.529 | 0.549 | 0.245
DW-GAE | **0.616** | 0.344 | **0.585** | **0.605** | **0.343**
DW-VGAE | 0.613 | 0.338 | 0.582 | 0.595 | 0.336

### Results on Pubmed test set
Link Prediction results:

Model | ROC | AP
---|---|---
GAE | 0.964 | 0.965
VGAE | 0.944 | 0.947
ARGE | 0.968 | 0.971
ARVGE | 0.965 | 0.968
DW-GAE | 0.947 | 0.947
DW-VGAE | 0.953 | 0.952

Clustering results:

Model | Acc | NMI | F1 | Precision | ARI
---|---|---|---|---|---
GAE | 0.697 | 0.33 | 0.69 | 0.72 | 0.322
VGAE | 0.608 | 0.219 | 0.612 | 0.613 | 0.195
DW-GAE | 0.726 | **0.355** | 0.714 | 0.729 | 0.37
DW-VGAE | **0.736** | 0.346 | **0.725** | **0.736** | **0.381**

Runs in 2-3 mins for cora dataset on cpu. The code currently doesn't support GPU.
