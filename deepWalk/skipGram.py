import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, input_feat_dim, num_nodes):
        super(SkipGram, self).__init__()
        self.num_nodes = num_nodes
        #self.context_embds = torch.Tensor(torch.randn(input_feat_dim, num_nodes).float())
        #self.context_embds.requires_grad = True
        self.context_embds = torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.randn(input_feat_dim, num_nodes)))

    def get_input_layer(self, word_idx):
        x = torch.zeros(self.num_nodes).float()
        x[word_idx] = 1.0
        return x

    def forward(self, node, z):
        x = self.get_input_layer(node)
        z1 = torch.matmul(self.context_embds, x)
        z2 = torch.matmul(z, z1)
        log_softmax = F.log_softmax(z2, dim=0)
        return log_softmax
