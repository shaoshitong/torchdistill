import os, sys
import numpy as np
if __name__ == "__main__":
    negative_identity_weight = [0, 0, 0]
    positive_identity_weight = [0, 0, 0]
    negative_classes_weight = [0, 0, 0]
    positive_classes_weight = [0, 0, 0]
    negative_policy_weight = [0, 0, 0]
    positive_policy_weight = [0, 0, 0]
    iter_list=np.linspace(0,1,6)
    for i,num in enumerate(iter_list):
        log = f'log/cifar100/kd/policy/wrn16_2_from_wrn40_2_2倍_policy_{num}_2.txt'
        os.system(f"python ./SST/image_classification_policy_p.py --log {log} --p {num}")
        print(f"============================end {i}================================")