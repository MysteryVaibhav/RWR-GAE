# DW-GAE
Deep Walk Regularized Graph Auto Encoder, implementation using pyTroch.

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
DW-VGAE |  | 

Clustering results:

Model | Acc | NMI | F1 | Precision | ARI
---|---|---|---|---|---
GAE | 0.596 | 0.429 | 0.595 | 0.596 | 0.347
VGAE | 0.609 | 0.436 | 0.609 | 0.609 | 0.346
GAE (our impl) | 0.526 | 0.42 | 0.508 | 0.530 | 0.308
VGAE (our impl) | 0.590 | 0.445 | 0.563 | 0.578 | 0.351
ARGE | 0.640 | 0.449 | 0.619 | **0.646** | 0.352
ARVGE | 0.638 | 0.450 | **0.627** | 0.624 | 0.374
DW-GAE | **0.669** | **0.464** | 0.618 | 0.629 | **0.417**
DW-VGAE | | | | | 

Runs in 2-3 mins for cora dataset on cpu. The code currently doesn't support GPU.
