import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from einops import rearrange
from .tem_mixf import Temporal_MixFormer
from .spa_mixf import Spatial_MixFormer

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class unit_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Ske_MixF(nn.Module):
    def __init__(self, in_channels, out_channels, A, Frames, stride=1, residual=True):
        super(Ske_MixF, self).__init__()
        self.spa_mixf = Spatial_MixFormer(in_channels, out_channels, A)
        self.tem_mixf = Temporal_MixFormer(out_channels, out_channels, Frames, kernel_size=5, stride=stride, dilations=[1,2],residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_skip(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tem_mixf(self.spa_mixf(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=2):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, 2)
        self.num_point = num_point        
        self.data_bn = nn.BatchNorm1d(num_person * 80 * num_point)        
        self.to_joint_embedding = nn.Linear(in_channels, 80)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))
        
        self.l1 = Ske_MixF(80, 80, A, 64, residual=False)
        self.l2 = Ske_MixF(80, 80, A, 64)
        self.l3 = Ske_MixF(80, 80, A, 64)
        self.l4 = Ske_MixF(80, 80, A, 64)
        self.l5 = Ske_MixF(80, 160, A, 32, stride=2)
        self.l6 = Ske_MixF(160, 160, A, 32)
        self.l7 = Ske_MixF(160, 160, A, 32)
        self.l8 = Ske_MixF(160, 320, A, 16, stride=2)
        self.l9 = Ske_MixF(320, 320, A, 16)
        self.l10= Ske_MixF(320, 320, A, 16)

        self.fc = nn.Linear(320, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        # Retrospect Model
        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(80, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(160, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        self.num_class=num_class
        
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))        

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        p = self.A_vector
        p = torch.tensor(p,dtype=torch.float)
        x = p.to(x.device).expand(N*M*T, -1, -1) @ x
        
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2=x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3=x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
                
        x2 = self.first_tram(x2)#x2(N*M,64,75,25)
        x3 = self.second_tram(x3)#x3(N*M,128,75,25)
        x =x + x2 + x3
        
        x = x.reshape(N, M, 320, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)





