import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

class GraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm

        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs)

        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs)

        self.fc = nn.Linear(num_inputs, num_outputs)

    def forward(self, graph, x):
        batch_size = graph.batch_size
        if self.batch_norm == "pre":
            x = x.view(batch_size, -1, self.num_inputs)
            x = x.transpose(1, 2)
            x = x.contiguous()
            x = self.bn(x)
            x = x.transpose(1, 2)
            x = x.contiguous()
            x = x.view(-1, self.num_inputs)
        x = self.fc(x)
        if self.batch_norm == "post":
            x = x.view(batch_size, -1, self.num_outputs)
            x = x.transpose(1, 2)
            x = x.contiguous()
            x = self.bn(x)
            x = x.transpose(1, 2)
            x = x.contiguous()
            x = x.view(-1, self.num_outputs)

        return x

class LapDeepModel(nn.Module):
    def __init__(self, in_features=3, out_features=1, num_hidden=128, layers=15, bnmode='', only_lap = False, **useless):
        super(LapDeepModel, self).__init__()

        self.conv1 = GraphConv1x1(in_features, num_hidden, batch_norm='')
        self.layer_num = layers
        self.bottleneck = [num_hidden]* (layers+1)
        for i in range(self.layer_num):
            if i % 2 == 0 or only_lap:
                module = _LapResNet2(self.bottleneck[i], self.bottleneck[i+1], bnmode)
            else:
                module = _AvgResNet2(self.bottleneck[i], self.bottleneck[i+1], bnmode)
            self.add_module('rn{}'.format(i), module)

        if bnmode is not None:
            bnmode += 'pre'
        self.conv2 = GraphConv1x1(num_hidden, out_features, batch_norm=bnmode)

    def forward(self, graph, inputs):
        x = self.conv1(graph, inputs)

        for i in range(self.layer_num):
            x = self._modules['rn{}'.format(i)](graph, x)

        x = F.elu(x)
        x = self.conv2(graph, x)

        return x + repeating_expand(inputs, x.size(1))


class _AvgResNet2(nn.Module):
    def __init__(self, num_inputs, num_outputs=None, bnmode='', inner_layers = 2):
        super().__init__()
        if num_outputs is None:
            num_outputs = num_inputs
        self.num_outputs = num_outputs

        if bnmode is not None:
            bnmode = bnmode + 'pre'
        self.add_module(f'bn_fc{0}', GraphConv1x1(2 * num_inputs, num_outputs, batch_norm=bnmode))
        self.layer = inner_layers
        for i in range(1, self.layer):
            bn_fc = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm=bnmode)
            self.add_module(f'bn_fc{i}', bn_fc)

    def global_average(self, graph, feat):
        mask = graph.ndata['mask']
        global_sum = SumPooling()(graph, feat * mask)
        global_norm = SumPooling()(graph, mask)
        return global_sum / global_norm

    def forward(self, graph, inputs):
        x = inputs
        for i in range(self.layer):
            x = F.elu(x)
            glob_avg = self.global_average(graph, x)
            x = x.view(graph.batch_size, -1, x.shape[-1])
            x = torch.cat([x, glob_avg.unsqueeze(1).expand_as(x)], 2)
            x = x.view(-1, x.shape[-1])
            x = self._modules[f'bn_fc{i}'](graph, x)

        return x + inputs


class _LapResNet2(nn.Module):
    def __init__(self, num_inputs, num_outputs=None, bnmode='', inner_layers = 2):
        super().__init__()
        if num_outputs is None:
            num_outputs = num_inputs
        self.num_outputs = num_outputs

        if bnmode is not None:
            bnmode = bnmode + 'pre'
        self.add_module(f'bn_fc{0}', GraphConv1x1(2 * num_inputs, num_outputs, batch_norm=bnmode))
        self.layer = inner_layers
        for i in range(1, self.layer):
            bn_fc = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm=bnmode)
            self.add_module(f'bn_fc{i}', bn_fc)

    def forward(self, graph, inputs):
        feat = inputs
        for i in range(self.layer):
            feat = F.elu(feat)
            graph.ndata['feat'] = feat
            print(graph.ndata['feat'].shape, graph.ndata['feat'].dtype)
            print(graph.edata['L'].shape, graph.edata['L'].dtype)
            graph.update_all(fn.u_mul_e('feat', 'L', 'msg'), fn.sum('msg', 'feat'))
            feat = torch.cat([feat, graph.ndata['feat']], 1)
            feat = self._modules[f'bn_fc{i}'](graph, feat)

        # residual
        return feat + inputs


def repeating_expand(inputs, out_features):
    _, in_features = inputs.size()
    times = out_features // in_features
    remin = out_features % in_features
    expanded_input = torch.cat([inputs]*times + [inputs[:,:remin]],dim=1)
    return expanded_input
