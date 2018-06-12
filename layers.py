import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import Im2Col , Col2Im
import pdb

def xavier(m):
    if isinstance(m, (nn.Conv2d, Similarity)):
        init.xavier_normal(m.weight.data)
        if m.bias is not None:
            init.constant(m.bias.data,0)
    if isinstance(m, Similarity):
        init.xavier_normal(m.tamplate)

class Mex(nn.Module):
    def __init__(self, kernel_size, 
    stride=1, padding=0, dilation=1, groups=1,beta=3, eps=1e-2):
        super(Mex, self).__init__()
        helper = lambda x: x if isinstance(x, (list, tuple)) else (x, x)
        self.kernel_size = helper(kernel_size)
        self.stride = helper(stride)
        self.padding = helper(padding)
        self.dilation = helper(dilation)
        self.groups = groups
        self.beta = nn.Parameter(torch.Tensor(1).fill_(beta))
        self.eps=eps
    def forward(self, x):
        N, C, H, W = x.shape
        HH, WW = self.kernel_size[0], self.kernel_size[1]
        beta = self.beta + self.eps
        H_out = 1 + (H + 2 * self.padding[0] - self.dilation[0]*(HH - 1) - 1 ) / self.stride[0] # formula from pytorch doc
        W_out = 1 + (W + 2 * self.padding[1] - self.dilation[1]*(WW - 1) - 1) / self.stride[1]
        x_max = x.max()
        x = x.sub(x_max).mul(beta)
        x = Im2Col.apply(x, self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.view(N, C, HH*WW, H_out*W_out)
        x = x.mean(2)
        # x.shape = N, C,  H_out*W_out
        x = x.log().div(beta) + x_max
        return x.view(N,C, H_out, W_out)

class Similarity(nn.Conv2d):
    ''' Definition in _ConvNd
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.transposed = transposed
    self.output_padding = output_padding
    self.groups = groups
    '''
    def __init__(self, in_channels, out_channels, kernel_size, 
    stride=1, padding=0, dilation=1, groups=1, bias=True, sim_type='linear'):
        '''
        sim_type = linear/conv/l1/l2
        '''
        super(Similarity, self).__init__(in_channels, out_channels, kernel_size, 
        stride, padding, dilation, groups, bias=bias) # note that is will have bias para but none forever
        self.sim_type = sim_type
        self.tamplate = nn.Parameter(torch.Tensor(self.weight.size())) if sim_type!='conv' else None

    def forward(self,x):
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        H_out = 1 + (H + 2 * self.padding[0] - self.dilation[0]*(HH - 1) - 1 ) / self.stride[0] # formula from pytorch doc
        W_out = 1 + (W + 2 * self.padding[1] - self.dilation[1]*(WW - 1) - 1) / self.stride[1]
        x = Im2Col.apply(x, self.kernel_size, self.dilation, self.padding, self.stride)
        w = self.weight.view(1, F, C*HH*WW, 1) # w.shape = 1, F, C*HH*WW, 1
        x = x.unsqueeze(1) # x.shape = B, 1, C*HH*WW, H_out*W_out
        if self.sim_type == 'conv':
            x = w.mul(x).sum(-2) # outshape = B, F, H_out*W_out 
        elif self.sim_type == 'linear':
            t = self.tamplate.view_as(w)
            x = w.mul(t).mul(x).sum(-2)
        elif self.sim_type == 'l1':
            t = self.tamplate.view_as(w)
            x = x.sub(t).abs().mul(w).sum(-2)
        elif self.sim_type == 'l2':
            t = self.tamplate.view_as(w)
            x = x.sub(t).pow(2).mul(w).sum(-2)
        else:
            raise Exception("unknow sim_type: {}".format(self.sim_type)) 
        if self.bias is not None :
            x = x + self.bias.view(1, F, 1)
        x = x.view(N,F,H_out, W_out)
        return x

if __name__ == '__main__':
    sim = Similarity(3, 4, (3,3), 2,1, bias =True,sim_type='l1')
    mex = Mex(2,2)
    conv = torch.nn.Conv2d(3, 4,(3,3),2,1, bias=True)
    conv.weight = sim.weight
    x = torch.rand(2,3,300,300).mul(1e-4)
    print x.shape
    #print conv(x) - sim(x)
    x = sim(x)
    print x.shape
    x = mex(x)
    print x.shape
