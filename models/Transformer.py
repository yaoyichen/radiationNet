#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 00:29:16 2021

@author: yaoyichen
"""


"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my youtube channel!


"""

import torch
import torch.nn as nn
import sys
# sys.path.append("..")
# from utils.model_helper import ModelUtils


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        # x = self.dropout(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        # out = self.dropout(forward + x)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        feature_channel,
        output_channel,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        seq_length,
        dropout
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(feature_channel, embed_size)
        self.position_embedding = nn.Embedding(seq_length, embed_size)

        self.first = nn.Linear(feature_channel, embed_size)
        self.first_act = nn.ReLU()

        self.seq_length = seq_length

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.final = nn.Conv1d(
            embed_size, output_channel, kernel_size=1, padding=0, bias=True)

        # self.final = nn.Linear(embed_size, output_channel)

    def forward(self, x, mask=None):

        x = torch.permute(x, (0, 2, 1))
        N = x.shape[0]

        positions = torch.arange(0, self.seq_length).expand(
            N, self.seq_length).to(x.device)
        positions = self.position_embedding(positions)

        out = self.first_act(self.first(x))

        out = out + positions
        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = torch.permute(out, (0, 2, 1))
        out = self.final(out)
        # out = torch.permute(out, (0, 2, 1))

        return out


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def testSF():
    net = SelfAttention(embed_size=128, heads=2)

    values = torch.randn(100, 57, 128)
    keys = torch.randn(100, 57, 128)
    query = torch.randn(100, 57, 128)

    out = net(values, keys, query, mask=None)

    print(out.shape)


def testTFB():
    net = TransformerBlock(embed_size=128, heads=2,
                           dropout=0.1, forward_expansion=2)

    values = torch.randn(100, 57, 128)
    keys = torch.randn(100, 57, 128)
    query = torch.randn(100, 57, 128)

    out = net(values, keys, query, mask=None)
    print(out.shape)


def testEncoder():
    model = Encoder(feature_channel=34,
                    output_channel=4,
                    embed_size=128,
                    num_layers=6,
                    heads=1,
                    forward_expansion=1,
                    seq_length=57,
                    dropout=0.1)

    x = torch.randn(100, 34, 57)
    import time
    time1 = time.time()
    result = model(x)
    print(time.time() - time1)
    print(result.shape)
    model_info = get_parameter_number(model)

    print(model_info)


if __name__ == "__main__":
    testEncoder()
