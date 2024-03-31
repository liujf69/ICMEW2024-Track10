import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)
        
class Spa_Atten(nn.Module):
    def __init__(self, out_channels):
        super(Spa_Atten, self).__init__()
        self.out_channels=out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-1) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(17,17)
#        self.linear2 = nn.Linear(25,25)

    def forward(self, x): 
        N, C, T, V = x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]
        Q_o = Q_Spa_Trans = self.avg_pool(x1.permute(0,3,1,2).contiguous())
        K_o = K_Spa_Trans = self.avg_pool(x2.permute(0,3,1,2).contiguous())
        Q_Spa_Trans = self.relu(self.linear(Q_Spa_Trans.squeeze(-1).squeeze(-1)))
        K_Spa_Trans = self.relu(self.linear(K_Spa_Trans.squeeze(-1).squeeze(-1)))
        Spa_atten = self.soft(torch.einsum('nv,nw->nvw', (Q_Spa_Trans, K_Spa_Trans))).unsqueeze(1).repeat(1,self.out_channels,1,1)  
        return Spa_atten, Q_o, K_o

class Spatial_MixFormer(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4, num_subset=3,t_stride=1,t_padding=0,t_dilation=1,bias=True,first=False,residual=True):
        super(Spatial_MixFormer, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.groups=groups
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.num_subset = 3
        self.alpha = nn.Parameter(torch.ones(1))
        self.A_GEME = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,17,17]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.A_SE = Variable(torch.from_numpy(np.reshape(A.astype(np.float32),[3,1,17,17]).repeat(groups,axis=1)), requires_grad=False) 
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(17,17)
        self.Spa_Att = Spa_Atten(out_channels//4)
        self.AvpChaRef = nn.AdaptiveAvgPool2d(1) 
        self.ChaRef_conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_subset,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        self.fc = nn.Linear(34, 17)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / 17))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-1)
        self.relu = nn.ReLU()         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x0):
        
        N, C, T, V = x0.size()
        A = self.A_SE.cuda(x0.get_device()) + self.A_GEME
        norm_learn_A = A.repeat(1,self.out_channels//self.groups,1,1)      
        A_final=torch.zeros([N,self.num_subset,self.out_channels,17,17],dtype=torch.float,device='cuda').detach().cuda(x0.get_device())    
        m = x0         
        m = self.conv(m)
        n, kc, t, v = m.size()
        m = m.view(n, self.num_subset, kc// self.num_subset, t, v)
        for i in range(self.num_subset):  
            m1,Q1,K1 = self.Spa_Att(m[:,i,:(kc// self.num_subset)//4,:,:])             
            m2,Q2,K2 = self.Spa_Att(m[:,i,(kc// self.num_subset)//4:((kc// self.num_subset)//4)*2,:,:])  
            m3,Q3,K3 = self.Spa_Att(m[:,i,((kc// self.num_subset)//4)*2:((kc// self.num_subset)//4)*3,:,:])             
            m4,Q4,K4 = self.Spa_Att(m[:,i,((kc// self.num_subset)//4)*3:((kc// self.num_subset)//4)*4,:,:])
            m1_2 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K1.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q2.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            m2_3 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K2.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q3.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            m3_4 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K3.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q4.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)

            m1 = m1/2 + m1_2/4
            m2 = m2/2 + m1_2/4 + m2_3/4 
            m3 = m3/2 + m2_3/4 + m3_4/4 
            m4 = m4/2 + m3_4/4
            atten = torch.cat([m1,m2,m3,m4],dim=1)
            A_final[:,i,:,:,:] = atten * 0.5 + norm_learn_A[i]
        m = torch.einsum('nkctv,nkcvw->nctw', (m, A_final))   
        
        # Channel Reforming       
        CR_in = self.AvpChaRef(m)
        CR_in = self.ChaRef_conv(CR_in.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        CR_out = m + m * self.sigmoid(CR_in).expand_as(m)               
        
        out = self.bn(CR_out)        
        out += self.down(x0) 
        out = self.relu(out)
        return out

