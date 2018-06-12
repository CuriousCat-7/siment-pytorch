import torch
from torch import nn
from torch.nn.functional import Im2Col , Col2Im
import pdb
class Mex(nn.Module):
    def __init__(self,  bias=True):
        super(Mex, self).__init__()
        if bias:
            pass
        else:
            self.register_parameter('bias', None)
        

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
    stride=1, padding=0, dilation=1, groups=1, sim_type='linear'):
        super(Similarity, self).__init__(in_channels, out_channels, kernel_size, 
        stride, padding, dilation, groups, bias=False) # note that is will have bias para but none forever
        self.sim_type = sim_type
        self.tamplate = nn.Parameter(torch.Tensor(self.weight.size()))

    def forward(self,x):
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        H_out = 1 + (H + 2 * self.padding[0] - self.dilation[0]*(HH - 1) - 1 ) / self.stride[0]
        W_out = 1 + (W + 2 * self.padding[1] - self.dilation[1]*(WW - 1) - 1) / self.stride[1]
        x = Im2Col.apply(x, self.kernel_size, self.dilation, self.padding, self.stride)
        w = self.weight.view(1, F, C*HH*WW, 1) # w.shape = 1, F, C*HH*WW, 1
        x = x.unsqueeze(1) # x.shape = B, 1, C*HH*WW, H_out*W_out
        x = w.mul(x).sum(-2) # outshape = B, F, H_out*W_out 
        x = x.view(N,F,H_out, W_out)
        if self.sim_type == 'linear':
            pass
        elif self.sim_type == 'l1':
            pass
        elif self.sim_type == 'l2':
            pass
        return x

if __name__ == '__main__':
    sim = Similarity(64,4, (3,3), 2,1)
    conv = torch.nn.Conv2d(64,4,(3,3),2,1, bias=False)
    conv.weight = sim.weight
    x = torch.rand(2,64,300,300)
    print x.shape
    print (sim(x) - conv(x)).sum()