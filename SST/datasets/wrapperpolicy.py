import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
import random
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchdistill.datasets.wrapper import register_dataset_wrapper,BaseDatasetWrapper
class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(128, 128, 128)):
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

        def rotate_with_fill(img, magnitude):
            rot = img.convert('RGBA').rotate(magnitude)
            return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            'shearX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            'shearY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            'translateX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            'translateY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),
            'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            'posterize': lambda img, magnitude: ImageOps.posterize(img, magnitude),
            'solarize': lambda img, magnitude: ImageOps.solarize(img, magnitude),
            'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),
            'equalize': lambda img, magnitude: ImageOps.equalize(img),
            'invert': lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        label=0
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
            label=1
        return img,label




@register_dataset_wrapper
class FKDDatasetWarpper(BaseDatasetWrapper):
    def __init__(self,org_dataset):
        super(FKDDatasetWarpper, self).__init__(org_dataset)
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
        sample,target,supp_dict=super(FKDDatasetWarpper, self).__getitem__(index)
        policy_index=torch.zeros(self.policies_len).float()
        new_sample=sample
        for i in range(self.policies_len):
            new_sample,label=self.policies[i](new_sample)
            policy_index[i]=label
        supp_dict['policy_index']=torch.stack([
            torch.zeros(self.policies_len).float(),
            policy_index
        ])
        sample=torch.stack([
            sample,
            new_sample,
        ])
        return sample,target,supp_dict

