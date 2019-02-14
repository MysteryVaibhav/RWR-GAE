from __future__ import division
from __future__ import print_function
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import numpy as np
import scipy.sparse as sp
import torch

np.random.seed(SEED)
torch.manual_seed(SEED)
from gae.utils import load_data
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_ae', help="models used")
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--emb', type=str,
                    default='G:\\My Drive\\CMU\\Research\\Code\\sem4\\gae-pytorch\\gae\\logs\\50_30_30_gae\\emb_epoch_90.npy',
                    help='saved embeddings path')
parser.add_argument('--n-clusters', default=7, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
parser.add_argument('--plot', type=int, default=1, help="whether to plot the clusters using tsne")
args = parser.parse_args()


def plot(X, fig, col, size, true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]])


def plotClusters(tqdm, hidden_emb, true_labels):
    tqdm.write('Start plotting using TSNE...')
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 4, true_labels)
    fig.savefig("plot.png")
    tqdm.write("Finished plotting")


def visualize(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    plotClusters(tqdm, np.load(args.emb), true_labels)


if __name__ == '__main__':
    visualize(args)
