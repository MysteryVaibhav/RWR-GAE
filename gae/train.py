from __future__ import division
from __future__ import print_function

import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from gae.model import GCNModelVAE, GCNModelAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter
from deepWalk.skipGram import SkipGram

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--dw', type=bool, default=True, help="whether to use deepWalk regularization")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=5, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=30, type=int, help='Number of random walks to start at each node')
parser.add_argument('--lr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Before proceeding further, make the structure for doing deepWalk
    if args.dw:
        print('Using deepWalk regularization...')
        G = load_edgelist_from_csr_matrix(adj_orig, undirected=True)
        print("Number of nodes: {}".format(len(G.nodes())))
        num_walks = len(G.nodes()) * args.number_walks
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * args.walk_length
        print("Data size (walks*length): {}".format(data_size))

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    if args.model == 'gcn_vae':
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.dw:
        sg = SkipGram(args.hidden2, adj.shape[0])
        optimizer_dw = optim.Adam(sg.parameters(), lr=args.lr_dw)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        if args.dw:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        # After back-propagating gae loss, now do the deepWalk regularization
        if args.dw:
            sg.train()
            for walk in build_deepwalk_corpus_iter(G, num_paths=args.number_walks,
                                                   path_length=args.walk_length, alpha=0,
                                                   rand=random.Random(args.seed)):

                # Construct the pairs for predicting context node
                idx_pairs = []
                # for each node, treated as center word
                for center_node_pos in range(len(walk)):
                    # for each window position
                    for w in range(-args.window_size, args.window_size + 1):
                        context_node_pos = center_node_pos + w
                        # make soure not jump out sentence
                        if context_node_pos < 0 or context_node_pos >= len(walk) or center_node_pos == context_node_pos:
                            continue
                        context_node_idx = walk[context_node_pos]
                        idx_pairs.append((int(walk[center_node_pos]), int(context_node_idx)))

                # Do actual prediction
                for src_node, tgt_node in idx_pairs:
                    optimizer_dw.zero_grad()
                    log_softmax = sg(src_node, mu)
                    y_true = torch.from_numpy(np.array([tgt_node])).long()
                    loss_dw = F.nll_loss(log_softmax.view(1, -1), y_true)
                    loss_dw.backward(retain_graph=True)
                    cur_dw_loss = loss_dw.item()
                    optimizer_dw.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        if args.dw:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss_gae=", "{:.5f}".format(cur_loss),
                  "train_loss_dw=", "{:.5f}".format(cur_dw_loss), "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )
        else:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
