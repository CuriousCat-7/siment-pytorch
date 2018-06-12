import torch
from torch import nn
from layers import Similarity, Mex
import pdb
class Replace(object):
    def __init__(self, sim_type='l1'):
        self.sim_type = sim_type
        
    def __call__(self, net):
        self.conv2sim(net)
        self.pool2mex(net)
    def conv2sim(self,net):
        atts = dir(net)
        for att in atts:
            try:
                attob = net.__getattr__(att)
            except:
                continue
            if isinstance(attob, nn.Conv2d):
                sim = Similarity(attob.in_channels, attob.out_channels,attob.kernel_size,
                        attob.stride, attob.padding, attob.dilation, attob.groups, attob.bias is not None, 
                        sim_type=self.sim_type)
                net.__setattr__(att, sim)
            elif isinstance(attob, nn.Sequential):
                ms = attob.children()
                newms=[]
                for m in ms:
                    if isinstance(m, nn.Conv2d):
                        sim = Similarity(m.in_channels, m.out_channels, m.kernel_size,
                                m.stride, m.padding, m.dilation, m.groups, m.bias is not None, 
                                sim_type=self.sim_type)
                        newms.append(sim)
                    else:
                        self.conv2sim(m)
                        newms.append(m)
                net.__setattr__(att, nn.Sequential(*newms))
            elif isinstance(attob, nn.Module):
                self.conv2sim( attob)

    def pool2mex(self, net):
        atts = dir(net)
        for att in atts:
            try:
                attob = net.__getattr__(att)
            except:
                continue
            if isinstance(attob, nn.MaxPool2d):
                mex = Mex(attob.kernel_size, attob.stride, attob.padding, attob.dilation)
                net.__setattr__(att, mex)
            elif isinstance(attob, nn.Sequential):
                ms = attob.children()
                newms=[]
                for m in ms:
                    if isinstance(m, nn.MaxPool2d):
                        mex = Mex(m.kernel_size, m.stride, m.padding, m.dilation)
                        newms.append(mex)
                    else:
                        self.pool2mex(m)
                        newms.append(m)
                net.__setattr__(att, nn.Sequential(*newms))
            elif isinstance(attob, nn.Module):
                self.pool2mex(attob)

if __name__ == '__main__':
    from pytorch_cifar.models import *
    net = ResNet101()
    Replace()(net)
    print net