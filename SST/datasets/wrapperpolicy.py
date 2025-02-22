import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
import torch.nn.functional as F
import random
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchdistill.datasets.wrapper import register_dataset_wrapper,BaseDatasetWrapper


def rotate_with_fill(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)
def shearX(img,magnitude,fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),Image.BICUBIC, fillcolor=fillcolor)
def shearY(img,magnitude,fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),Image.BICUBIC, fillcolor=fillcolor)
def translateX(img,magnitude,fillcolor):
    return img.transform( img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),fillcolor=fillcolor)
def translateY(img,magnitude,fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),fillcolor=fillcolor)
def rotate(img,magnitude,fillcolor):
    return rotate_with_fill(img, magnitude)
def color(img,magnitude,fillcolor):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))
def posterize(img,magnitude,fillcolor):
    return ImageOps.posterize(img, magnitude)
def solarize(img,magnitude,fillcolor):
    return ImageOps.solarize(img, magnitude)
def contrast(img,magnitude,fillcolor):
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))
def sharpness(img,magnitude,fillcolor):
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))
def brightness(img,magnitude,fillcolor):
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))
def autocontrast(img,magnitude,fillcolor):
    return ImageOps.autocontrast(img)
def equalize(img,magnitude,fillcolor):
    return ImageOps.equalize(img)
def invert(img,magnitude,fillcolor):
    return ImageOps.invert(img)

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2




class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(128, 128, 128)):
        self.fillcolor=fillcolor
        ranges = {
            'shearX': np.linspace(0, 0.3, 10),
            'shearY': np.linspace(0, 0.3, 10),
            'translateX': np.linspace(0, 150 / 331, 10),
            'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10),
            'color': np.linspace(0.0, 0.9, 10),
            'posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0.0, 0.9, 10),
            'sharpness': np.linspace(0.0, 0.9, 10),
            'brightness': np.linspace(0.0, 0.9, 10),
            'autocontrast': [0] * 10,
            'equalize': [0] * 10,
            'invert': [0] * 10
        }


        func = {
            'shearX': shearX,
            'shearY': shearY,
            'translateX': translateX,
            'translateY': translateY,
            'rotate': rotate,
            'color': color,
            'posterize': posterize,
            'solarize': solarize,
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'autocontrast': autocontrast,
            'equalize': equalize,
            'invert': invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        label=0
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1,self.fillcolor)
            label=1
        return img,label




@register_dataset_wrapper
class PolicyDataset(BaseDatasetWrapper):
    def __init__(self,org_dataset,mixcut=False,mixcut_prob=0.1,beta=0.3):
        super(PolicyDataset, self).__init__(org_dataset)
        self.transform=org_dataset.transform
        org_dataset.transform=None
        self.policies = [
            SubPolicy(0.5, 'invert', 7),
            SubPolicy(0.5, 'rotate', 2),
            SubPolicy(0.5, 'sharpness', 1),
            SubPolicy(0.5, 'shearY', 8),
            SubPolicy(0.5, 'autocontrast', 8),
            SubPolicy(0.5, 'color', 3),
            SubPolicy(0.5, 'sharpness', 9),
            SubPolicy(0.5, 'equalize', 5),
            SubPolicy(0.5, 'contrast', 7),
            SubPolicy(0.5, 'translateY', 3),
            SubPolicy(0.5, 'brightness',6),
            SubPolicy(0.5, 'solarize', 2),
            SubPolicy(0.5, 'translateX',3),
            SubPolicy(0.5, 'shearX', 8),
        ]
        self.policies_len=len(self.policies)
        self.beta=beta
        self.mixcut_prob=mixcut_prob
        self.mixcut=mixcut
    def __getitem__(self, index):
        sample, target_a, supp_dict = super(PolicyDataset, self).__getitem__(index)
        sample=self.transform(sample).detach()
        r = np.random.rand(1)
        if self.mixcut and self.beta > 0 and r < self.mixcut_prob:
            lam = np.random.beta(self.beta, self.beta)
            rand_index=random.randint(0,len(self)-1)
            rsample,target_b, supp_dict = super(PolicyDataset, self).__getitem__(rand_index)
            rsample = self.transform(rsample)
            bbx1, bby1, bbx2, bby2 = rand_bbox(sample.size(), lam)
            sample[ :, bbx1:bbx2, bby1:bby2] = rsample[ :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample.size()[-1] * sample.size()[-2]))
            target=F.one_hot(torch.LongTensor([target_a]),10)*lam+ F.one_hot(torch.LongTensor([target_b]),10)*(1.-lam)
        else:
            target=target_a
        new_sample=transforms.ToPILImage()(sample)
        policy_index = torch.zeros(self.policies_len).float()
        for i in range(self.policies_len):
            new_sample,label=self.policies[i](new_sample)
            policy_index[i]=label
        new_sample=self.transform(new_sample).detach()
        if not isinstance(target,torch.Tensor):
            target=F.one_hot(torch.LongTensor([target]),10)
        target=target.expand(2,-1) # 2,1
        policy_target=torch.stack([torch.zeros(self.policies_len).float(),policy_index],0) # 2, policy_len
        target=torch.cat([target,policy_target],1) # 2,num_classes+policy_len
        sample=torch.stack([
            sample,
            new_sample,
        ])
        return sample,target,supp_dict




@register_dataset_wrapper
class PolicyDatasetC100(BaseDatasetWrapper):
    def __init__(self,org_dataset,mixcut=False,mixcut_prob=0.1,beta=0.3):
        super(PolicyDatasetC100, self).__init__(org_dataset)
        self.transform=org_dataset.transform
        org_dataset.transform=None
        self.policies = [
            SubPolicy(0.5,'autocontrast', 2),
            SubPolicy(0.5, 'contrast', 3),
            SubPolicy(0.5,  'posterize', 0),
            SubPolicy(0.5,  'solarize', 4),

            SubPolicy(0.5, 'translateY', 8),
            SubPolicy(0.5, 'shearX', 5),
            SubPolicy(0.5, 'color', 3),
            SubPolicy(0.5, 'shearY', 0),
            SubPolicy(0.5, 'translateX', 1),

            SubPolicy(0.5, 'sharpness', 5),
            SubPolicy(0.5, 'invert', 4),
            SubPolicy(0.5, 'color', 4),
            SubPolicy(0.5, 'equalize', 8),
            SubPolicy(0.5, 'rotate', 3),

        ]
        self.beta=beta
        self.mixcut_prob=mixcut_prob
        self.mixcut=mixcut
        self.policies_len=len(self.policies)

    def __getitem__(self, index):
        sample, target_a, supp_dict = super(PolicyDatasetC100, self).__getitem__(index)
        sample=self.transform(sample).detach()
        r = np.random.rand(1)
        if self.mixcut and self.beta > 0 and r < self.mixcut_prob:
            lam = np.random.beta(self.beta, self.beta)
            rand_index=random.randint(0,len(self)-1)
            rsample,target_b, supp_dict = super(PolicyDatasetC100, self).__getitem__(rand_index)
            rsample = self.transform(rsample)
            bbx1, bby1, bbx2, bby2 = rand_bbox(sample.size(), lam)
            sample[ :, bbx1:bbx2, bby1:bby2] = rsample[ :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample.size()[-1] * sample.size()[-2]))
            target=F.one_hot(torch.LongTensor([target_a]),100)*lam+ F.one_hot(torch.LongTensor([target_b]),100)*(1.-lam)
        else:
            target=target_a
        new_sample=transforms.ToPILImage()(sample)
        policy_index = torch.zeros(self.policies_len).float()
        for i in range(self.policies_len):
            new_sample,label=self.policies[i](new_sample)
            policy_index[i]=label
        new_sample=self.transform(new_sample).detach()
        if not isinstance(target,torch.Tensor):
            target=F.one_hot(torch.LongTensor([target]),100)
        target=target.expand(2,-1) # 2,1
        policy_target=torch.stack([torch.zeros(self.policies_len).float(),policy_index],0) # 2, policy_len
        target=torch.cat([target,policy_target],1) # 2,num_classes+policy_len
        sample=torch.stack([
            sample,
            new_sample,
        ])
        return sample,target,supp_dict





def policy_classes_compute(hot):
    l=hot.shape[0]
    exp=torch.arange(0,l)
    weight=2**exp
    return (hot*weight).sum().long()

@register_dataset_wrapper
class ICPDataset(BaseDatasetWrapper):
    def __init__(self,org_dataset):
        super(ICPDataset, self).__init__(org_dataset)
        self.transform=org_dataset.transform
        org_dataset.transform=None
        self.policies = [
            SubPolicy(0.5, 'invert', 7),
            SubPolicy(0.5, 'rotate', 2),
            SubPolicy(0.5, 'sharpness', 1),
            SubPolicy(0.5, 'shearY', 8),
            SubPolicy(0.5, 'autocontrast', 8),
            SubPolicy(0.5, 'color', 3),
            SubPolicy(0.5, 'sharpness', 9),
            SubPolicy(0.5, 'equalize', 5),
            SubPolicy(0.5, 'contrast', 7),
            SubPolicy(0.5, 'translateY', 3),
            SubPolicy(0.5, 'brightness',6),
            SubPolicy(0.5, 'solarize', 2),
            SubPolicy(0.5, 'translateX',3),
            SubPolicy(0.5, 'shearX', 8),
        ]
        self.policies_len=len(self.policies)

    def __getitem__(self, index):
        sample,target,supp_dict=super(ICPDataset, self).__getitem__(index)
        policy_index=torch.zeros(self.policies_len).float()
        new_sample=sample
        for i in range(self.policies_len):
            new_sample,label=self.policies[i](new_sample)
            policy_index[i]=label
        new_sample=self.transform(new_sample).detach()
        sample=self.transform(sample).detach()
        if isinstance(target,torch.Tensor) and target.ndim==2 and target.shape[-1]!=1:
            target=target.argmax(1)
        elif not isinstance(target,torch.Tensor):
            target=torch.LongTensor([target])
        identity_target=torch.LongTensor([index]).unsqueeze(0).expand(2,-1)
        classes_target=target.unsqueeze(0).expand(2,-1) # 2,1
        policy_target = torch.stack([torch.zeros(self.policies_len).int(), policy_index.int()], 0)  # 2, policy_len
        target=torch.cat([identity_target,classes_target,policy_target],1) # 2,3
        sample=torch.stack([
            sample,
            new_sample,
        ])
        return sample,target,supp_dict
