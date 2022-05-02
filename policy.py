import os, sys

if __name__ == "__main__":
    negative_identity_weight = [0, 0, 0]
    positive_identity_weight = [0, 0, 0]
    negative_classes_weight = [0, 0, 0]
    positive_classes_weight = [0, 0, 0]
    negative_policy_weight = [0, 0, 0]
    positive_policy_weight = [0, 0, 0]

    for i in range(len(negative_identity_weight)):
        log = f'log/cifar10/icpkd/resnet18_from_resnet50_policy_kd_{i}_1倍.txt'
        os.system("rm -rf ./resource")
        os.system(f"python ./SST/image_classification_policy.py --log {log}")
        print(f"============================end {i}================================")