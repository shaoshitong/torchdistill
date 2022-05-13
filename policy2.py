import os, sys
import numpy as np
if __name__ == "__main__":
    iter_list=np.linspace(0,1,6)
    f=open("run.sh","w")
    f.write("#!/bin/bash\n")
    f.write("source activate pytorch\n")
    for i,num in enumerate(iter_list):
        log = f'log/cifar10/kd/policy/resnet18_from_resnet50_2倍_policy_{num}_3.txt'
        cmd=f"python ./SST/image_classification_policy_p.py --log {log} --p {num}"
        f.write(cmd+"\n")
        print(f"============================end {i}================================")
    f.close()