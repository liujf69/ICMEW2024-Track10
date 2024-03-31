import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class Tem_Seq_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Tem_Seq_h, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.AvpTemSeq = nn.AdaptiveAvgPool2d(1)
        self.MaxTemSeq = nn.AdaptiveMaxPool2d(1)
        self.combine_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V = x.size()
        x = x.permute(0,2,1,3).contiguous()
        Q_Tem_Seq = self.AvpTemSeq(x)
        K_Tem_Seq = self.MaxTemSeq(x)
        Combine = torch.cat([Q_Tem_Seq,K_Tem_Seq],dim=2)
        Combine = self.combine_conv(Combine.permute(0,2,1,3).contiguous()).permute(0,2,1,3).contiguous()
        Tem_Seq_out = (x * self.sigmoid(Combine).expand_as(x)).permute(0,2,1,3).contiguous()      
        return Tem_Seq_out
                             
class Tem_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size, stride=1):
        super(Tem_Trans, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))     
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.AvpTemTrans = nn.AdaptiveAvgPool2d(1)
        self.MaxTemTrans = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()        
        self.soft = nn.Softmax(-1)
        self.linear = nn.Linear(Frames,Frames)
        
    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V=x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]                
        Q_Tem_Trans = self.AvpTemTrans(x1.permute(0,2,1,3).contiguous())
        K_Tem_Trans = self.MaxTemTrans(x2.permute(0,2,1,3).contiguous())
        Q_Tem_Trans = self.relu(self.linear(Q_Tem_Trans.squeeze(-1).squeeze(-1)))
        K_Tem_Trans = self.relu(self.linear(K_Tem_Trans.squeeze(-1).squeeze(-1)))       
        Tem_atten = self.sigmoid(torch.einsum('nt,nm->ntm', (Q_Tem_Trans, K_Tem_Trans)))                   
        Tem_Trans_out = self.bn(torch.einsum('nctv,ntm->ncmv', (x, Tem_atten)))      
        return Tem_Trans_out
        
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):               
        x = self.conv(x)
        x = self.bn(x)
        return x

class Temporal_MixFormer(nn.Module):
    def __init__(self,in_channels,out_channels,Frames,kernel_size=3,stride=1,dilations=[1,2,3,4],residual=True,residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 3) == 0
        self.num_branches = len(dilations) + 3
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(in_channels,branch_channels,kernel_size=1,padding=0),nn.BatchNorm2d(branch_channels),nn.ReLU(inplace=True),
            TemporalConv(branch_channels,branch_channels,kernel_size=ks,stride=stride,dilation=dilation),)
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            Tem_Trans(in_channels, branch_channels, Frames, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))        
        self.branches.append(nn.Sequential(
            Tem_Seq_h(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)        
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out



