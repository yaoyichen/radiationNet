
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""


""" helper function for nn.Module """
import torch
import torch.nn as nn


class ModelUtils(object):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel()
                            for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    @staticmethod
    def get_memory_usage(model, input_shape):
        from torchstat import stat
        result = stat(model, input_shape)
        print(result)

    @staticmethod
    def print_model_layer(model):
        for name, value in model.named_parameters():
            print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    @staticmethod
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer):
        print("=> Loading checkpoint")

        # tt = torch.load(checkpoint, map_location=torch.device("cpu"))
        # model.load_state_dict(tt["state_dict"])
        # optimizer.load_state_dict(tt["optimizer"])

        # optimizer.load_state_dict(
        #     torch.load(checkpoint["optimizer"], map_location=torch.device("cpu")))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
