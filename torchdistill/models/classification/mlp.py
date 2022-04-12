import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,in_channel,out_channel,hidden_channel=256):
        super(MLP, self).__init__()
        self.adaptivepool2d=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        self.fc1=nn.Linear(in_channel,hidden_channel)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden_channel,out_channel)
    def forward(self,x):
        if x.ndim==4:
            x=self.adaptivepool2d(x)
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return x
