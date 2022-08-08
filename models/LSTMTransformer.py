#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:23:53 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.Transformer import TransformerBlock


class LSTM_ATT(nn.Module):
    def __init__(self, feature_channel,
                 output_channel,
                 hidden_size,
                 num_lstm_layers,
                 embed_size,
                 num_attention_layers,
                 heads,
                 forward_expansion,
                 dropout):
        super(LSTM_ATT, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_layers = num_attention_layers
        self.output_channel = output_channel
        self.lstm = nn.LSTM(feature_channel, hidden_size,
                            num_lstm_layers, batch_first=True, bidirectional=True)

        self.final = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = torch.permute(x, [0, 2, 1])
        # Set initial hidden and cell states

        h0 = torch.zeros(2*self.num_lstm_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)
        c0 = torch.zeros(2*self.num_lstm_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)

        return out


class ATT_LSTM(nn.Module):
    def __init__(self, feature_channel,
                 output_channel,
                 hidden_size,
                 num_lstm_layers,
                 embed_size,
                 num_attention_layers,
                 heads,
                 forward_expansion,
                 dropout):
        super(ATT_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_layers = num_attention_layers
        self.output_channel = output_channel

        self.first = nn.Linear(feature_channel, embed_size)
        self.first_act = nn.ReLU()

        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_lstm_layers, batch_first=True, bidirectional=True)

        self.final = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x, mask=None):
        # print(x.shape)
        x = torch.permute(x, [0, 2, 1])
        # Set initial hidden and cell states

        x = self.first_act(self.first(x))
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, x, x, mask)

        # print(x.shape)

        h0 = torch.zeros(2*self.num_lstm_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)
        c0 = torch.zeros(2*self.num_lstm_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden
        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )
        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)

        return out


def test():
    hidden_size = 128
    num_lstm_layers = 2
    batch_size = 100
    embed_size = 256
    num_attention_layers = 3
    heads = 1
    forward_expansion = 1
    dropout = 0.0

    net = LSTM_ATT(feature_channel=34, output_channel=4, hidden_size=hidden_size,
                   num_lstm_layers=num_lstm_layers,
                   embed_size=embed_size,
                   num_attention_layers=num_attention_layers,
                   heads=heads,
                   forward_expansion=forward_expansion,
                   dropout=dropout)

    y = net(torch.randn(batch_size, 34, 57))
    print(y.size())


def test2():
    hidden_size = 128
    num_lstm_layers = 2
    batch_size = 100
    embed_size = 128
    num_attention_layers = 3
    heads = 1
    forward_expansion = 1
    dropout = 0.0

    net = ATT_LSTM(feature_channel=34, output_channel=4, hidden_size=hidden_size,
                   num_lstm_layers=num_lstm_layers,
                   embed_size=embed_size,
                   num_attention_layers=num_attention_layers,
                   heads=heads,
                   forward_expansion=forward_expansion,
                   dropout=dropout)

    y = net(torch.randn(batch_size, 34, 57))
    print(y.size())


if __name__ == "__main__":
    test2()
