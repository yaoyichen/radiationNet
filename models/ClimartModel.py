
import torch.nn as nn
from typing import Sequence, Optional, Dict, Union,Callable
import numpy as np
import torch
from torch import Tensor
from einops import rearrange, repeat



def get_activation_function(name: str, functional: bool = False, num: int = 1):
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "identity": nn.Identity(),
                None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu, 'prelu': nn.PReLU(),
                }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU(),
                }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_normalization_layer(name, dims, num_groups=None, device='cpu'):
    if not isinstance(name, str) or name.lower() == 'none':
        return None
    elif 'batch' in name:
        return nn.BatchNorm1d(num_features=dims).to(device)
    elif 'layer' in name:
        return nn.LayerNorm(dims).to(device)
    elif 'inst' in name:
        return nn.InstanceNorm1d(num_features=dims).to(device)
    elif 'group' in name:
        if num_groups is None:
            num_groups = int(dims / 10)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)




class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'gelu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 output_normalization: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 out_layer_bias_init: Tensor = None,
                 name: str = ""
                 ):
        """
        Args:
            input_dim (int): the expected 1D input tensor dim
            output_activation_function (str, bool, optional): By default no output activation function is used (None).
                If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
                If True, the same activation function is used as defined by the arg `activation_function`.
        """

        super().__init__()
        self.name = name
        hidden_layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            hidden_layers += [MLP_Block(
                in_dim=dims[i - 1],
                out_dim=dims[i],
                net_norm=net_normalization.lower() if isinstance(net_normalization, str) else 'none',
                activation_function=activation_function,
                dropout=dropout,
                residual=residual
            )]
        self.hidden_layers = nn.ModuleList(hidden_layers)

        out_weight = nn.Linear(dims[-1], output_dim, bias=True)
        if out_layer_bias_init is not None:
            # log.info(' Pre-initializing the MLP final/output layer bias.')
            out_weight.bias.data = out_layer_bias_init
        out_layer = [out_weight]
        if output_normalization and net_normalization != 'none':
            out_layer += [get_normalization_layer(net_normalization, output_dim)]
        if output_activation_function is not None and output_activation_function:
            if isinstance(output_activation_function, bool):
                output_activation_function = activation_function

            out_layer += [get_activation_function(output_activation_function, functional=False)]
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            X = layer(X)

        Y = self.out_layer(X)
        return Y.squeeze(1)


class MLP_Block(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 net_norm: str = 'none',
                 activation_function: str = 'Gelu',
                 dropout: float = 0.0,
                 residual: bool = False
                 ):
        super().__init__()
        layer = [nn.Linear(in_dim, out_dim, bias=net_norm != 'batch_norm')]
        if net_norm != 'none':
            layer += [get_normalization_layer(net_norm, out_dim)]
        layer += [get_activation_function(activation_function, functional=False)]
        if dropout > 0:
            layer += [nn.Dropout(dropout)]
        self.layer = nn.Sequential(*layer)
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
        elif residual:
            print('MLP block with residual!')

    def forward(self, X: Tensor) -> Tensor:
        X_out = self.layer(X)
        if self.residual:
            X_out += X
        return X_out




class ClimartMLP(nn.Module):
    def __init__(self,
                 raw_input_dim,
                 output_dim,
                 hidden_dims: Sequence[int],
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 output_normalization: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 *args, **kwargs):
        """
        Args:
            input_dim must either be an int, i.e. the expected 1D input tensor dim, or a dict s.t.
                input_dim and spatial_dim have the same keys to compute the flattened input shape.
            output_activation_function (str, bool, optional): By default no output activation function is used (None).
                If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
                If True, the same activation function is used as defined by the arg `activation_function`.
        """
        # super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        # self.save_hyperparameters()
        super().__init__()

        # if isinstance(raw_input_dim, dict):
        #     assert all([k in self.raw_spatial_dim.keys() for k in self.raw_input_dim.keys()])
        #     self.input_dim = sum([self.raw_input_dim[k] * max(1, self.raw_spatial_dim[k]) for k in self.raw_input_dim.keys()])  # flattened
        #     # self.log_text.info(f' Inferred a flattened input dim = {self.input_dim}')
        # else:
        self.input_dim = raw_input_dim
        
        self.output_dim = output_dim
        self.out_layer_bias_init = None

        # self.output_dim = self.raw_output_dim

        self.mlp = MLP(
            self.input_dim, hidden_dims, self.output_dim,
            net_normalization=net_normalization, activation_function=activation_function, dropout=dropout,
            residual=residual, output_normalization=output_normalization,
            output_activation_function=output_activation_function, 
            out_layer_bias_init=self.out_layer_bias_init
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.mlp(X) 
        return  torch.reshape(X, (X.shape[0], 4, 50))
    
    
#%%


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = GAP()
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


import torch.nn.functional as F

class GAP():
    def __init__(self):
        pass

    def __call__(self, x):
        x = F.adaptive_avg_pool1d(x, 1)
        return x
    
    
class Climart_CNN_Net(nn.Module):
    def __init__(self,
                 raw_input_dim,
                 output_dim,
                 hidden_dims: Sequence[int],
                 dilation: int = 1,
                 net_normalization: str = 'none',
                 gap: bool = False,
                 se_block: bool = False,
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 *args, **kwargs):
        super().__init__()
        input_dim = raw_input_dim
        output_dim  = output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        
        self.channel_list = list(hidden_dims)
        # input_dim = self.input_transform.output_dim
        self.channel_list = [input_dim] + self.channel_list

        self.use_linear = gap
        self.ratio = 16
        self.kernel_list = [20, 10, 5]
        self.stride_list = [2, 1, 1]  # 221
        self.global_average = GAP()
        
        self.net_normalization = "batch"
        
        
        feat_cnn_modules = []
        for i in range(len(self.channel_list) - 1):
            out = self.channel_list[i + 1]
            feat_cnn_modules += [nn.Conv1d(in_channels=self.channel_list[i],
                                           out_channels=out, kernel_size=self.kernel_list[i],
                                           stride=self.stride_list[i],
                                           bias=self.net_normalization != 'batch_norm') ]
                                
            if se_block:
                feat_cnn_modules.append(SE_Block(out, self.ratio))
            if self.net_normalization != 'none':
                feat_cnn_modules += [get_normalization_layer(self.net_normalization, out)]
            feat_cnn_modules += [get_activation_function(activation_function, functional=False)]
            # TODO: Need to add adaptive pooling with arguments
            feat_cnn_modules += [nn.Dropout(dropout)]

        self.feat_cnn = nn.Sequential(*feat_cnn_modules)

    def forward(self, X: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """
        input:
            Dict with key-values {GLOBALS: x_glob, LEVELS: x_lev, LAYERS: x_lay},
             where x_*** are the corresponding features.
        """
        print(X.shape)
        X = self.feat_cnn(X)
        print(X.shape)
        self.use_linear = True
        if not self.use_linear:
            print(f"self.use_linear:{self.use_linear}")
            X = rearrange(X, 'b f c -> b (f c)')
            X = self.ll(X)
        else:
            X = self.global_average(X)
            X = torch.reshape(X,(X.shape[0],4,-1))

        return X.squeeze(2)



def test():
    """
    用于测试 Climart上自带的MLP和CNN模型
    torch.Size([100, 4, 50])
    torch.Size([100, 96, 50])
    torch.Size([100, 200, 3])
    torch.Size([100, 4, 50])
    """
    model_mpl = ClimartMLP(raw_input_dim =782, 
                    output_dim = 200, 
                    hidden_dims = [512,256,256],
                    activation_function = 'gelu')

    input_ = torch.randn(100, 14*50+82)

    output_ = model_mpl(input_) 
    print(output_.shape)



    input_ = torch.randn(100, 96,50)

    model_cnn = Climart_CNN_Net(raw_input_dim = 96, 
                output_dim = 4,
                hidden_dims = [200, 400, 200],
                activation_function = 'gelu'
                )

    output_ = model_cnn(input_) 

    print(output_.shape)


if __name__ == "__main__":
    test()
