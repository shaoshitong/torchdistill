from typing import Any

import torch,math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchdistill.models.registry import register_model_func

"""
Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
for CIFAR datasets, referring to https://github.com/szagoruyko/wide-residual-networks
"""
SECOND_URL='https://github.com/shaoshitong/torchdistill/releases/download'
ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar10-wide_resnet40_4.pt',
    'cifar10-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar10-wide_resnet28_10.pt',
    'cifar10-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar10-wide_resnet16_8.pt',
    'cifar100-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar100-wide_resnet40_4.pt',
    'cifar100-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar100-wide_resnet28_10.pt',
    'cifar100-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar100-wide_resnet16_8.pt',
    'cifar100-wide_resnet40_2': SECOND_URL+'/v0.3.2/wrn_40_2.pth',
}



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,**kwargs):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f1, f2, f3], out
        else:
            return out


@register_model_func
def wide_resnet(
        depth: int,
        k: int,
        dropout_p: float,
        num_classes: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> WideResNet:
    assert (depth - 4) % 6 == 0, 'depth of Wide ResNet (WRN) should be 6n + 4'
    model = WideResNet(depth,num_classes,k, dropout_p,**kwargs)
    model_key = 'cifar{}-wide_resnet{}_{}'.format(num_classes, depth, k)
    if pretrained and model_key in MODEL_URL_DICT:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL_DICT[model_key], progress=progress)['state_dict']
        model.load_state_dict(state_dict)
    return model


@register_model_func
def wide_resnet40_4(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-40-4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p (float): p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(40, 4, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet40_2(dropout_p=0.0, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-40-4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p (float): p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(40, 2, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet28_10(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-28-10 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(28, 10, dropout_p, num_classes, pretrained, progress, **kwargs)


@register_model_func
def wide_resnet16_8(dropout_p=0.3, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-16-8 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p: p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(16, 8, dropout_p, num_classes, pretrained, progress, **kwargs)
@register_model_func
def wide_resnet16_2(dropout_p=0.0, num_classes=10, pretrained=False, progress=True, **kwargs: Any) -> WideResNet:
    r"""WRN-40-4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        dropout_p (float): p in Dropout
        num_classes (int): 10 and 100 for CIFAR-10 and CIFAR-100, respectively
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10/100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return wide_resnet(16, 2, dropout_p, num_classes, pretrained, progress, **kwargs)