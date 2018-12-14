# gae-pytorch
Graph Auto-Encoder in PyTorch

This is a PyTorch implementation of the Variational Graph Auto-Encoder model described in the paper:
 
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on or refers to https://github.com/tkipf/gae, https://github.com/tkipf/pygcn and https://github.com/vmasrani/gae_in_pytorch.

### Requirements
- Python 3
- PyTorch 0.4 

### To train a model run the following command
```bash
cd gae
python train.py --model="gcn_vae" --dataset-str="cora" --dw=0 --walk-length=5 --window-size=5 --number-walks=30
```
- Supported models are "gcn_vae" and "gcn_ae"
- Supported datasets are "cora" and "citeseer"
- dw, whether to use regularization or not (0: no regularization, 1: yes)
- if dw = 0, then all the remaining params are useless

### Results on CORA test set
Model | ROC | AP
---|---|---
GAE | 0.91430 | 0.92585
VGAE | 0.921715 | 0.927751

Runs in 2-3 mins for cora dataset on cpu. The code currently doesn't support GPU.
