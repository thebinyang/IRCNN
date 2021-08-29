import torch
import math
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from typing import Union, Sequence
from functools import partial
from .functional import AutogradConvRNN, _conv_cell_helper, ConvNdWithSamePadding
from .utils import _single, _pair, _triple


class IRCNN_LSTMCell(nn.Module):

    def __init__(self, mode: str,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True,
                 groups: int = 1
                 ):
        super(IRCNN_LSTMCell, self).__init__()
        self.mode = mode
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        gate_size = 4 * out_channels
        Tgate_size = out_channels

        self.weight_ih = Parameter(torch.Tensor(gate_size, in_channels // groups))
        self.weight_hh = Parameter(torch.Tensor(gate_size, out_channels // groups))
        self.weight_pi = Parameter(torch.Tensor(out_channels, out_channels // groups))
        self.weight_pf = Parameter(torch.Tensor(out_channels, out_channels // groups))
        self.weight_po = Parameter(torch.Tensor(out_channels, out_channels // groups))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(gate_size))
            self.bias_hh = Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

        self.a = nn.Parameter(torch.Tensor(1))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.a.data, mean=0.5, std=0.01)
        nn.init.normal_(self.b.data, mean=0.5, std=0.01)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_input(self, input):
        if input.size(1) != self.in_channels:
            raise RuntimeError(
                "input has inconsistent channels: got {}, expected {}".format(
                    input.size(1), self.in_channels))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.out_channels:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.out_channels))

    def forward(self, input, time_dis, time_dis2, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            batch_size = input.size(0)
            insize = input.shape[2:]
            hidden = input.new_zeros(batch_size, self.out_channels, *insize, requires_grad=False)
            hidden = (hidden, hidden)
        self.check_forward_hidden(input, hidden[0])
        self.check_forward_hidden(input, hidden[1])
        
        if mode in ['LSTM', 'DisLSTM', 'DisLSTM2', 'DisLSTM3', 'DisLSTM4']:
            cell = partial(LSTMdisCell, linear_func=linear_func, mode=mode)
        else:
            raise Exception('Unknown mode: {}'.format(mode))
        
        return cell(
            input, time_dis, hidden, self.a, self.b,
            self.weight_ih, self.weight_hh, self.weight_pi, self.weight_pf, self.weight_po,
            self.bias_ih, self.bias_hh)


def LSTMdisCell(input, time_dis, hidden, a, b, w_ih, w_hh, w_pi, w_pf, w_po,
                    b_ih=None, b_hh=None, linear_func=None, mode='DisLSTM1'):

        if linear_func is None:
            linear_func = F.linear
        hx, cx = hidden
        hx = torch.squeeze(hx)
        cx = torch.squeeze(cx)
        input = torch.squeeze(input)
        gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = ingate + linear_func(cx, w_pi)
        forgetgate = forgetgate + linear_func(cx, w_pf)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)

        if mode == 'LSTM':  # The LSTM is PeepholeLSTM
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = outgate + linear_func(cy, w_po)
            outgate = F.sigmoid(outgate)
            hy = outgate * F.tanh(cy)
            return hy, cy

        if mode == 'DisLSTM':
            assert len(time_dis) == 2
            forcoff = a.cuda() * torch.exp(- c.cuda() * time_dis[0])
            incoff = a.cuda() * torch.exp(- c.cuda() * time_dis[1])

            forcoff = forcoff.unsqueeze(1).expand_as(forgetgate)
            incoff = incoff.unsqueeze(1).expand_as(ingate)

            forgetgate = forcoff * forgetgate
            ingate = incoff * ingate
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = outgate + linear_func(cy, w_po)
            outgate = F.sigmoid(outgate)
            hy = outgate * F.tanh(cy)
            return hy, cy

        if mode == 'DisLSTM2':
            forcoff = a.cuda() * torch.max(torch.tensor(1 - c.cuda() * time_dis[0]), torch.tensor([0.1]).cuda())
            incoff = a.cuda() * torch.max(torch.tensor(1 - c.cuda() * time_dis[1]), torch.tensor([0.1]).cuda())
            forcoff = forcoff.unsqueeze(1).expand_as(forgetgate)
            incoff = incoff.unsqueeze(1).expand_as(ingate)

            forgetgate = forcoff * forgetgate
            ingate = incoff * ingate
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = outgate + linear_func(cy, w_po)
            outgate = F.sigmoid(outgate)
            hy = outgate * F.tanh(cy)
            return hy, cy

        if mode == 'DisLSTM3':
            forcoff = a.cuda() * torch.max(torch.tensor(1 - c.cuda() * time_dis[0] * time_dis[0]),
                                           torch.tensor([0.1]).cuda())
            incoff = a.cuda() * torch.max(torch.tensor(1 - c.cuda() * time_dis[1] * time_dis[1]),
                                          torch.tensor([0.1]).cuda())
            forcoff = forcoff.unsqueeze(1).expand_as(forgetgate)
            incoff = incoff.unsqueeze(1).expand_as(ingate)
            forgetgate = forcoff * forgetgate
            ingate = incoff * ingate
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = outgate + linear_func(cy, w_po)
            outgate = F.sigmoid(outgate)
            hy = outgate * F.tanh(cy)
            return hy, cy

        if mode == 'DisLSTM4':
            forcoff = a.cuda() * torch.log(1 + torch.exp(- c.cuda() * time_dis[0]))
            incoff = a.cuda() * torch.log(1 + torch.exp(- c.cuda() * time_dis[1]))
            forcoff = forcoff.unsqueeze(1).expand_as(forgetgate)
            incoff = incoff.unsqueeze(1).expand_as(ingate)

            forgetgate = forcoff * forgetgate
            ingate = incoff * ingate
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = outgate + linear_func(cy, w_po)
            outgate = F.sigmoid(outgate)
            hy = outgate * F.tanh(cy)
            return hy, cy