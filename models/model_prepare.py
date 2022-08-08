

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.append("..")
from models.FCNet1D import FCNet1D
from models.ResNet1D import ResNet
from models.UNet1D import UNET
from models.ConvLSTM1D import RNN_LSTM, RNN_GRU
from models.FNO_H import FNO1d
from models.FNO_F import FNO1d_Feature
from models.FNO_2D import FNO2d
from models.SVD_2D import SVD2D
from models.Transformer import Encoder
from models.LSTMTransformer import LSTM_ATT, ATT_LSTM


def load_model(model_name, device, feature_channel, signal_length):

    if model_name == "FC":
        model = FCNet1D(feature_channel=feature_channel,
                        output_channel=4,
                        hidden_number=10,
                        hidden_size=200,
                        # hidden_number=3,
                        # hidden_size=50,
                        signal_length=signal_length,
                        dim_add=0
                        )
    elif model_name == "FC_M":
        model = FCNet1D(feature_channel=feature_channel,
                        output_channel=4,
                        hidden_number=5,
                        hidden_size=30,
                        # hidden_number=3,
                        # hidden_size=50,
                        signal_length=signal_length,
                        dim_add=0
                        )

    elif model_name == "RES":
        model = ResNet([10], feature_channel=feature_channel,
                       output_channel=4, intermediate_channel=128)
        #     signal_length=signal_length,
        #    dim_add=0)

    elif model_name == "UNET":
        model = UNET(feature_channel=feature_channel, output_channel=4,
                     features=[24, 48, 96, 192])


    elif model_name == "LSTM":

        model = RNN_LSTM(feature_channel=feature_channel, output_channel=4, hidden_size=96,
                         num_layers=5)

    elif model_name == "LSTM_32_5":
        model = RNN_LSTM(feature_channel=feature_channel, output_channel=4, hidden_size=32,
                         num_layers=5)

    elif model_name == "LSTM_32_3":
        model = RNN_LSTM(feature_channel=feature_channel, output_channel=4, hidden_size=32,
                         num_layers=3)

    elif model_name == "LSTM_16_1":
        model = RNN_LSTM(feature_channel=feature_channel, output_channel=4, hidden_size=16,
                         num_layers=1)

    elif model_name == "GRU":
        model = RNN_GRU(feature_channel=feature_channel, output_channel=4, hidden_size=128,
                        num_layers=5)

    elif model_name == "FNO_H":
        model = FNO1d(feature_channel=feature_channel,
                      output_channel=4, modes=16, width=96)

    elif model_name == "WRT":
        model = WRT1d(feature_channel=feature_channel,
                      output_channel=4, width=48)

    elif model_name == "FNO_F":
        model = FNO1d(feature_channel=feature_channel,
                      output_channel=4, modes=24, width=128, signal_length=signal_length,
                      dim_add=0)

    elif model_name == "FNO_2D":
        model = FNO2d(modes1=12, modes2=16,  width=24, in_channels=feature_channel,
                      out_channels=4)
        #   , signal_length=signal_length, dim_add=0)

    elif model_name == "SVD_2D":
        model = SVD2D(in_modes=7, out_modes=7, width=5,
                      in_channels=feature_channel, out_channels=4, signal_length=signal_length, dim_add=0)

    elif model_name == "ATT":
        # model = Encoder(feature_channel=feature_channel,
        #                 output_channel=4,
        #                 embed_size=128,
        #                 num_layers=6,
        #                 heads=1,
        #                 forward_expansion=1,
        #                 seq_length=57,
        #                 dropout=0.0)
        model = Encoder(feature_channel=feature_channel,
                        output_channel=4,
                        embed_size=128,
                        num_layers=7,
                        heads=1,
                        forward_expansion=1,
                        seq_length=57,
                        dropout=0.0)

    elif model_name == "LSTM_ATT":
        model = LSTM_ATT(feature_channel=feature_channel, output_channel=4, hidden_size=96,
                         num_lstm_layers=3,
                         embed_size=192,
                         num_attention_layers=3,
                         heads=1,
                         forward_expansion=1,
                         dropout=0.0)

    elif model_name == "ATT_LSTM":
        model = ATT_LSTM(feature_channel=feature_channel, output_channel=4, hidden_size=96,
                         num_lstm_layers=3,
                         embed_size=96,
                         num_attention_layers=3,
                         heads=1,
                         forward_expansion=1,
                         dropout=0.0)

    else:
        raise Exception('not implemented model : ' + model_name)

    return model





