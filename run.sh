#!/bin/bash
source activate pytorch
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_0.0_3.txt --p 0.0
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_0.2_3.txt --p 0.2
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_0.4_3.txt --p 0.4
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_0.6_3.txt --p 0.6
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_0.8_3.txt --p 0.8
python ./SST/image_classification_policy_p.py --log log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_1.0_3.txt --p 1.0
