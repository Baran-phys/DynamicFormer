import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class PsiSuffix(nn.Module):
    def __init__(self, features, predict_diagonal):
        super().__init__()
        layers = []
        for i in range(len(features) - 2):
            layers.append(DiagOffdiagMLP(features[i], features[i + 1], predict_diagonal))
            layers.append(nn.ReLU())
        layers.append(DiagOffdiagMLP(features[-2], features[-1], predict_diagonal))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiagOffdiagMLP(nn.Module):
    def __init__(self, in_features, out_features, seperate_diag):
        super(DiagOffdiagMLP, self).__init__()

        self.seperate_diag = seperate_diag
        self.conv_offdiag = nn.Conv2d(in_features, out_features, 1)
        if self.seperate_diag:
            self.conv_diag = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        # Assume x.shape == (B, C, N, N)
        if self.seperate_diag:
            return self.conv_offdiag(x) + (self.conv_diag(x.diagonal(dim1=2, dim2=3))).diag_embed(dim1=2, dim2=3)
        return self.conv_offdiag(x)


class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        self.query = nn.Sequential(
            nn.Linear(in_features, small_in_features),
            nn.Tanh(),
        )
        self.key = nn.Linear(in_features, small_in_features)

    def forward(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)  # B,N,C/10

        x = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # B,N,N

        x = x.transpose(1, 2)  # (B,N,N)
        x = x.softmax(dim=2)  # over rows
        x = torch.matmul(x, inp)  # (B, N, C)
        return x

    
class DeepSet(nn.Module):
    def __init__(self, in_features, feats, attention, normalization, second_bias):
        """
        DeepSets implementation
        :param in_features: input's number of features
        :param feats: list of features for each deepsets layer
        :param attention: True/False to use attention
        :param cfg: configurations of second_bias and normalization method
        """
        super(DeepSet, self).__init__()
        if cfg is None:
            cfg = {}

        layers = []
        normalization = cfg.get('normalization', 'fro')
        second_bias = cfg.get('second_bias', True)

        layers.append(DeepSetLayer(in_features, feats[0], attention, normalization, second_bias))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(DeepSetLayer(feats[i-1], feats[i], attention, normalization, second_bias))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features, attention, normalization, second_bias):
        """
        DeepSets single layer
        :param in_features: input's number of features
        :param out_features: output's number of features
        :param attention: Whether to use attention
        :param normalization: normalization method - 'fro' or 'batchnorm'
        :param second_bias: use a bias in second conv1d layer
        """
        super(DeepSetLayer, self).__init__()

        self.attention = None
        if attention:
            self.attention = Attention(in_features)
        self.layer1 = nn.Conv1d(in_features, out_features, 1)
        self.layer2 = nn.Conv1d(in_features, out_features, 1, bias=second_bias)

        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x.shape = (B,C,N)

        # attention
        if self.attention:
            x_T = x.transpose(2, 1)  # B,C,N -> B,N,C
            x = self.layer1(x) + self.layer2(self.attention(x_T).transpose(1, 2))
        else:
            x = self.layer1(x) + self.layer2(x - x.mean(dim=2, keepdim=True))

        # normalization
        if self.normalization == 'batchnorm':
            x = self.bn(x)
        else:
            x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN

        return x    
    
    
class PNAAggregator(nn.Module):
    def __init__(self, average_n = None):
        super().__init__()
        average_n = 1 if average_n is None else average_n
        self.average_n = torch.Tensor([average_n])
        #self.average_n = nn.Parameter(torch.Tensor([average_n]))
        self.name = "PNA"
        self.dim_multiplier = 12    # The output has 12x more channels than the input

    def forward(self, x):
        """ x: batch_size x n x channels"""
        x = x.unsqueeze(-1)         # bs, n, c, 1
        n = torch.Tensor([x.shape[1]])
        # n = x.shape[1]
        scalers = torch.Tensor([torch.log(n + 1) / torch.log(self.average_n + 1),
                   torch.log(self.average_n + 1) / torch.log(n + 1)]).to(x.device)

        x = torch.cat((x, x * scalers[0], x * scalers[1]), dim=-1)

        aggregators = [torch.sum(x, dim=1), torch.max(x, dim=1)[0], torch.mean(x, dim=1),
                       torch.std(x, dim=1)]
        aggregators = [agg.unsqueeze(2) for agg in aggregators]
        z = torch.cat(aggregators, dim=2)       # bs, channels, 4, 3
        z = torch.reshape(z, (z.shape[0], -1))
        return z
    
#class LePNAAggregator(nn.Module):
#    def __init__(self, average_n = None):
#        super().__init__()
#        average_n = 1 if average_n is None else average_n
#        #self.average_n = torch.Tensor([average_n])
#        self.average_n = nn.Parameter(torch.Tensor([average_n]))
#        self.name = "PNA"
#        self.dim_multiplier = 12    # The output has 12x more channels than the input
#
#    def forward(self, x):
#        """ x: batch_size x n x channels"""
#        x = x.unsqueeze(-1)         # bs, n, c, 1
#        n = torch.tensor([x.shape[1]], dtype=torch.float32, device=x.device)
#        # n = x.shape[1]
#        scalers = torch.Tensor([torch.log(n + 1) / torch.log(self.average_n + 1),
#                   torch.log(self.average_n + 1) / torch.log(n + 1)]).to(x.device)
#
#        x = torch.cat((x, x * scalers[0], x * scalers[1]), dim=-1)
#
#        aggregators = [torch.sum(x, dim=1), torch.max(x, dim=1)[0], torch.mean(x, dim=1),
#                       torch.std(x, dim=1)]
#        aggregators = [agg.unsqueeze(2) for agg in aggregators]
#        z = torch.cat(aggregators, dim=2)       # bs, channels, 4, 3
#        z = torch.reshape(z, (z.shape[0], -1))
#        return z


class LePNAAggregator(nn.Module):
    def __init__(self, average_n=None):
        super().__init__()
        average_n = 1 if average_n is None else average_n
        self.average_n = nn.Parameter(torch.tensor([average_n], dtype=torch.float32))
        self.name = "PNA"
        self.dim_multiplier = 12    # The output has 12x more channels than the input

    def forward(self, x):
        """ x: batch_size x n x channels"""
        x = x.unsqueeze(-1)         # bs, n, c, 1
        n = torch.tensor([x.shape[1]], dtype=torch.float32, device=x.device)
        #print(self.average_n)
        
        # Ensure scalers depend on self.average_n in a way that requires gradients
        scalers = torch.stack([torch.log(n + 1) / torch.log(self.average_n + 1),
                               torch.log(self.average_n + 1) / torch.log(n + 1)]).to(x.device)

        x = torch.cat((x, x * scalers[0], x * scalers[1]), dim=-1)

        aggregators = [torch.sum(x, dim=1), torch.max(x, dim=1)[0], torch.mean(x, dim=1),
                       torch.std(x, dim=1)]
        aggregators = [agg.unsqueeze(2) for agg in aggregators]
        z = torch.cat(aggregators, dim=2)       # bs, channels, 4, 3
        z = torch.reshape(z, (z.shape[0], -1))
        return z

    

class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.dim_multiplier = 2
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.softmax = nn.Softmax(dim=0)
        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x):
        """x: bs x n x channels. """
        # TODO: check the implementation
        batch_size = x.shape[0]

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(1, batch_size, self.out_channels)

        x = x.transpose(0, 1)           # n, bs, hidden

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star, h)    # q: 1, bs, hidden
            a = self.softmax(x * q)          # n, bs, hidden
            r = torch.sum(a * x, dim=0)     # bs, hidden
            q_star = torch.cat([q, r.unsqueeze(0)], dim=-1)
        return q_star.squeeze(0)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class DRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)
        self.d = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        term1 = x + (self.a * torch.square(torch.sin(self.a * x)) / self.b)
        term2 = self.c * torch.cos(self.b * x)
        term3 = self.d * torch.tanh(self.b * x)
        return term1 + term2 + term3

    
class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=True,
                 dim_in_2: int=None, modulation: str = '+'):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
            modulation (str): "+", "*" or "film". Used only if  dim_in_2 is not None (2 inputs to MLP)
        """
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.modulation = modulation
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        if dim_in_2 is not None:
            self.lin2 = nn.Linear(dim_in_2, width)
            if modulation == 'film':
                self.lin3 = nn.Linear(dim_in_2, width)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor, y: Tensor=None, mask: Tensor=None):
        if mask is not None:
            mask = mask.unsqueeze(-1) 
            x = x * mask  # Apply the mask to nullify the effect of some inputs

            out = self.lin1(x)
            if y is not None:
                out2 = self.lin2(y)
                if self.modulation == '+':
                    out = out + out2
                elif self.modulation == '*':
                    out = out * out2
                elif self.modulation == 'film':
                    out3 = self.lin3(y)
                    out = out * torch.sigmoid(out2) + out3
                else:
                    raise ValueError(f"Unknown modulation parameter: {self.modulation}")
            out = F.relu(out) * mask + (x if self.residual_start else 0)
            for layer in self.hidden:
                out = out + layer(F.relu(out))
            out = self.lin_final(out) + (out if self.residual_end else 0)
            return out * mask
        else:
            out = self.lin1(x)
            if y is not None:
                out2 = self.lin2(y)
                if self.modulation == '+':
                    out = out + out2
                elif self.modulation == '*':
                    out = out * out2
                elif self.modulation == 'film':
                    out3 = self.lin3(y)
                    out = out * torch.sigmoid(out2) + out3
                else:
                    raise ValueError(f"Unknown modulation parameter: {self.modulation}")
            out = F.relu(out) + (x if self.residual_start else 0)
            #out = self.nonlin_1(out) + (x if self.residual_start else 0)
            for layer in self.hidden:
                out = out + layer(F.relu(out))
            out = self.lin_final(F.relu(out)) + (out if self.residual_end else 0)
            return out


class MLP_dra(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, 
                 bias=True, dim_in_2: int=None, modulation: str = '+', 
                 dropout=0.0, BN=False, return_inter = False):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
            modulation (str): "+", "*" or "film". Used only if  dim_in_2 is not None (2 inputs to MLP)
        """
        super(MLP_dra, self).__init__()
        self.dim_in = dim_in
        self.bn = BN
        self.return_inter = return_inter
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.modulation = modulation
        self.hidden = nn.ModuleList()
        self.dra_1 = DRA()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        if dim_in_2 is not None:
            self.lin2 = nn.Linear(dim_in_2, width)
            if modulation == 'film':
                self.lin3 = nn.Linear(dim_in_2, width)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.final_dropout = nn.Dropout(dropout) 
        if self.bn:
            self.last_norm = nn.BatchNorm1d(width, affine=True)
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor, y: Tensor=None, mask: Tensor=None):
        if mask is not None:
            mask = mask.unsqueeze(-1) 
            x = x * mask  # Apply the mask to nullify the effect of some inputs

            out = self.lin1(x)
            if y is not None:
                out2 = self.lin2(y)
                if self.modulation == '+':
                    out = out + out2
                elif self.modulation == '*':
                    out = out * out2
                elif self.modulation == 'film':
                    out3 = self.lin3(y)
                    out = out * torch.sigmoid(out2) + out3
                else:
                    raise ValueError(f"Unknown modulation parameter: {self.modulation}")
            out = self.dra_1(out) * mask + (x if self.residual_start else 0)
            out = self.lin_final(out) + (out if self.residual_end else 0)
            return out * mask
        else:
            out = self.lin1(x)
            if y is not None:
                out2 = self.lin2(y)
                if self.modulation == '+':
                    out = out + out2
                elif self.modulation == '*':
                    out = out * out2
                elif self.modulation == 'film':
                    out3 = self.lin3(y)
                    out = out * torch.sigmoid(out2) + out3
                else:
                    raise ValueError(f"Unknown modulation parameter: {self.modulation}")
            pre_out = self.dra_1(out) + (x if self.residual_start else 0)
            pre_out = self.final_dropout(pre_out)
            if self.bn:
                pre_out = self.last_norm(pre_out)
            out = self.lin_final(pre_out) + (out if self.residual_end else 0)
            if self.return_inter:
                return out, pre_out
            else:
                return out
    

class MLP_dr(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=True,
                 dim_in_2: int=None, modulation: str = '+', cat=False, dropout=0.0, post_act_ln = False):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
            modulation (str): "+", "*" or "film". Used only if  dim_in_2 is not None (2 inputs to MLP)
        """
        super(MLP_dr, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.modulation = modulation
        self.cat = cat
        self.nonlin1 = LearnedSnake()
        self.nonlin2 = LearnedSnake()
        self.nonlin3 = LearnedSnake()
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Sequential(nn.Linear(self.dim_in, width, bias), 
                                  self.nonlin1,
                                  nn.Dropout(dropout) 
                                 )
        
        if dim_in_2 is not None:
            self.lin2 = nn.Sequential(nn.Linear(dim_in_2, width), 
                                      self.nonlin2,
                                      nn.Dropout(dropout)
                                     )
                    
            if modulation == 'film':
                self.lin3 = nn.Sequential(nn.Linear(dim_in_2, width), 
                                          LearnedSnake(),
                                          nn.Dropout(dropout)
                                         )

        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        #self.lin_final = nn.Sequential(nn.Linear(width, int(width//2), bias), 
        #                                  LearnedSnake(),
        #                                  #nn.LayerNorm(width),
        #                                  nn.Dropout(dropout),
        #                                  nn.Linear(int(width//2), dim_out, bias)
        #                                 )
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor, y: Tensor=None):
        """
        MLP is overloaded to be able to take two arguments.
        This is used in the first layer of the decoder to merge the set and the latent vector
        Args:
            x: a tensor with last dimension equals to dim_in
        """
        out = self.lin1(x)
        if y is not None:
            if self.cat:
                out2 = y
            else:
                out2 = self.lin2(y)
            if self.modulation == '+':
                out = out + out2
            elif self.modulation == '*':
                out = out * out2
            elif self.modulation == 'film':
                out3 = self.lin3(y)
                out = out * torch.sigmoid(out2) + out3
            else:
                raise ValueError(f"Unknown modulation parameter: {self.modulation}")
        out = self.nonlin3(out) + (x if self.residual_start else 0)
        for layer in self.hidden:
            out = out + layer(F.relu(out))
        out = self.lin_final(out) + (out if self.residual_end else 0)
        return out