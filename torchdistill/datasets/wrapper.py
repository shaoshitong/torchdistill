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

from torchdistill.common import file_util
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)
WRAPPER_CLASS_DICT = dict()


def default_idx2subpath(index):
    digits_str = '{:04d}'.format(index)
    return os.path.join(digits_str[-4:], digits_str)


def register_dataset_wrapper(cls):
    WRAPPER_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseDatasetWrapper(Dataset):
    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target = self.org_dataset.__getitem__(index)
        return sample, target, dict()

    def __len__(self):
        return len(self.org_dataset)


class CacheableDataset(BaseDatasetWrapper):
    def __init__(self, org_dataset, cache_dir_path, idx2subpath_func=None, ext='.pt'):
        super().__init__(org_dataset)
        self.cache_dir_path = cache_dir_path
        self.idx2subath_func = str if idx2subpath_func is None else idx2subpath_func
        self.ext = ext

    def __getitem__(self, index):
        sample, target, supp_dict = super().__getitem__(index)
        cache_file_path = os.path.join(self.cache_dir_path, self.idx2subath_func(index) + self.ext)
        if file_util.check_if_exists(cache_file_path):
            cached_data = torch.load(cache_file_path)
            supp_dict['cached_data'] = cached_data

        supp_dict['cache_file_path'] = cache_file_path
        return sample, target, supp_dict


@register_dataset_wrapper
class ContrastiveDataset(BaseDatasetWrapper):
    def __init__(self, org_dataset, num_negative_samples, mode, ratio):
        super().__init__(org_dataset)
        self.num_negative_samples = num_negative_samples
        self.mode = mode
        num_classes = len(org_dataset.classes)
        num_samples = len(org_dataset)
        labels = org_dataset.targets
        self.cls_positives = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positives[labels[i]].append(i)

        self.cls_negatives = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negatives[i].extend(self.cls_positives[j])

        self.cls_positives = [np.asarray(self.cls_positives[i]) for i in range(num_classes)]
        self.cls_negatives = [np.asarray(self.cls_negatives[i]) for i in range(num_classes)]
        if 0 < ratio < 1:
            n = int(len(self.cls_negatives[0]) * ratio)
            self.cls_negatives = [np.random.permutation(self.cls_negatives[i])[0:n] for i in range(num_classes)]

        self.cls_positives = np.asarray(self.cls_positives)
        self.cls_negatives = np.asarray(self.cls_negatives)

    def __getitem__(self, index):
        sample, target, supp_dict = super().__getitem__(index)
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positives[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)

        replace = True if self.num_negative_samples > len(self.cls_negatives[target]) else False
        neg_idx = np.random.choice(self.cls_negatives[target], self.num_negative_samples, replace=replace)
        contrast_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        supp_dict['pos_idx'] = index
        supp_dict['contrast_idx'] = contrast_idx
        return sample, target, supp_dict


@register_dataset_wrapper
class SSKDDatasetWrapper(BaseDatasetWrapper):
    def __init__(self, org_dataset):
        super().__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None

    def __getitem__(self, index):
        # Assume sample is a PIL Image
        sample, target, supp_dict = super().__getitem__(index)
        sample = torch.stack([self.transform(sample).detach(),
                              self.transform(sample.rotate(90, expand=True)).detach(),
                              self.transform(sample.rotate(180, expand=True)).detach(),
                              self.transform(sample.rotate(270, expand=True)).detach()])
        return sample, target, supp_dict


def get_dataset_wrapper(class_name, *args, **kwargs):
    if class_name not in WRAPPER_CLASS_DICT:
        logger.info('No dataset wrapper called `{}` is registered.'.format(class_name))
        return None

    instance = WRAPPER_CLASS_DICT[class_name](*args, **kwargs)
    return instance


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





