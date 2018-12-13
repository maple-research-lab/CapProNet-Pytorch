import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter

class Caps_BN(nn.Module):
    '''
    Input variable N*CD*H*W
    First perform normal BN without learnable affine parameters, then apply a C group convolution to perform per-capsule
    linear transformation
    '''
    def __init__(self, num_C, num_D):
        super(Caps_BN, self).__init__()
        self.BN = nn.BatchNorm2d(num_C*num_D, affine=False)
        self.conv = nn.Conv2d(num_C*num_D, num_C*num_D, 1, groups=num_C)
        
        eye = torch.FloatTensor(num_C, num_D, num_D).copy_(torch.eye(num_D), broadcast = True).view(num_C*num_D, num_D, 1, 1)
        self.conv.weight.data.copy_(eye)
        self.conv.bias.data.zero_()
        
    def forward(self, x):
        output = self.BN(x)
        output = self.conv(output)
        
        return output
        
class Caps_MaxPool(nn.Module):
    '''
    Input variable N*CD*H*W
    First get the argmax indices of capsule lengths, then tile the indices D time and apply the tiled indices to capsules
    '''
    def __init__(self, num_C, num_D, kernel_size, stride=None, padding=0, dilation=1):
        super(Caps_MaxPool, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)
        
    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[2:]
        x_caps = x.view(B, self.num_C, self.num_D, H, W)
        x_length = torch.sum(x_caps * x_caps, dim=2)
        x_length_pool, indices = self.maxpool(x_length)
        H_pool, W_pool = x_length_pool.shape[2:]
        indices_tile = torch.unsqueeze(indices, 2).expand(-1, -1, self.num_D, -1, -1).contiguous()
        indices_tile = indices_tile.view(B, self.num_C * self.num_D, -1)
        x_flatten = x.view(B, self.num_C*self.num_D, -1)
        output = torch.gather(x_flatten, 2, indices_tile).view(B, self.num_C*self.num_D, H_pool, W_pool)
        
        return output
        
class Caps_Conv(nn.Module):    
    def __init__(self, in_C, in_D, out_C, out_D, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Caps_Conv, self).__init__()
        self.in_C = in_C
        self.in_D = in_D
        self.out_C = out_C
        self.out_D = out_D
        self.conv_D = nn.Conv2d(in_C*in_D, in_C*out_D, 1, groups=in_C, bias=False)
        self.conv_C = nn.Conv2d(in_C, out_C, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        m = self.conv_D.kernel_size[0] * self.conv_D.kernel_size[1] * self.conv_D.out_channels
        self.conv_D.weight.data.normal_(0, math.sqrt(2. / m))
        n = self.conv_C.kernel_size[0] * self.conv_C.kernel_size[1] * self.conv_C.out_channels
        self.conv_C.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.conv_C.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv_D(x)
        x = x.view(x.shape[0], self.in_C, self.out_D, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.in_C, x.shape[3], x.shape[4])
        x = self.conv_C(x)
        x = x.view(-1, self.out_D, self.out_C, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.out_C*self.out_D, x.shape[3], x.shape[4])
        
        return x
        
class Squash(nn.Module):
    def __init__(self, num_C, num_D, eps=0.0001):
        super(Squash, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        
    def forward(self, x):
        x_caps = x.view(x.shape[0], self.num_C, self.num_D, x.shape[2], x.shape[3])
        x_length = torch.sqrt(torch.sum(x_caps * x_caps, dim=2))
        x_length = torch.unsqueeze(x_length, 2)
        x_caps = x_caps * x_length / (1+self.eps+x_length*x_length)
        x = x_caps.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return x

class Relu_Caps(nn.Module):
    def __init__(self, num_C, num_D, theta=0.2, eps=0.0001):
        super(Relu_Caps, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.theta = theta
        self.eps = eps
        
    def forward(self, x):
        x_caps = x.view(x.shape[0], self.num_C, self.num_D, x.shape[2], x.shape[3])
        x_length = torch.sqrt(torch.sum(x_caps * x_caps, dim=2))
        x_length = torch.unsqueeze(x_length, 2)
        x_caps = F.relu(x_length - self.theta) * x_caps / (x_length + self.eps)
        x = x_caps.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return x

class Relu_Adpt(nn.Module):
    def __init__(self, num_C, num_D, eps=0.0001):
        super(Relu_Adpt, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        
        self.theta = Parameter(torch.Tensor(1, self.num_C, 1, 1, 1))
        self.theta.data.fill_(0.)
        
    def forward(self, x):
        x_caps = x.view(x.shape[0], self.num_C, self.num_D, x.shape[2], x.shape[3])
        x_length = torch.sqrt(torch.sum(x_caps * x_caps, dim=2))
        x_length = torch.unsqueeze(x_length, 2)
        x_caps = F.relu(x_length - self.theta) * x_caps / (x_length + self.eps)
        x = x_caps.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return x  

class LinearCaps(nn.Module):
    def __init__(self, in_features, num_C, num_D, bias = False, eps=0.0001):
        super(LinearCaps, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_C*num_D, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_C*num_D))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
#        weights_reduce = torch.sqrt(torch.sum(self.weight*self.weight, dim=1))
#        weights_reduce = torch.reciprocal(weights_reduce + self.eps)
#        weights_reduce = torch.unsqueeze(weights_reduce, dim=0)
#        
#        self.scalar.data.copy_(weights_reduce.data)
#        del weights_reduce
        
    def forward(self, x):
        scalar = torch.sqrt(torch.sum(self.weight * self.weight, dim=1))
        scalar = torch.reciprocal(scalar + self.eps)
        scalar = torch.unsqueeze(scalar, dim=1)        
    
        output = F.linear(x, scalar * self.weight, self.bias)
        
        return output
        
class LinearCapsPro(nn.Module):
    def __init__(self, in_features, num_C, num_D, eps=0.0001):
        super(LinearCapsPro, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps 
        self.weight = Parameter(torch.Tensor(num_C*num_D, in_features))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x, eye):
        weight_caps = self.weight[:self.num_D]
        sigma = torch.inverse(torch.mm(weight_caps, torch.t(weight_caps))+self.eps*eye)
        sigma = torch.unsqueeze(sigma, dim=0)
        for ii in range(1, self.num_C):
            weight_caps = self.weight[ii*self.num_D:(ii+1)*self.num_D]
            sigma_ = torch.inverse(torch.mm(weight_caps, torch.t(weight_caps))+self.eps*eye)
            sigma_ = torch.unsqueeze(sigma_, dim=0)
            sigma = torch.cat((sigma, sigma_))
        
        out = torch.matmul(x, torch.t(self.weight))
        out = out.view(out.shape[0], self.num_C, 1, self.num_D)
        out = torch.matmul(out, sigma)
        out = torch.matmul(out, self.weight.view(self.num_C, self.num_D, self.in_features))
        out = torch.squeeze(out, dim=2)
        out = torch.matmul(out, torch.unsqueeze(x, dim=2))
        out = torch.squeeze(out, dim=2)
        
        return torch.sqrt(out)              