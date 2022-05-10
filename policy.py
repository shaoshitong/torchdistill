import os, sys

if __name__ == "__main__":
    negative_identity_weight = [0, 0, 0]
    positive_identity_weight = [0, 0, 0]
    negative_classes_weight = [0, 0, 0]
    positive_classes_weight = [0, 0, 0]
    negative_policy_weight = [0, 0, 0]
    positive_policy_weight = [0, 0, 0]

    for i in range(len(negative_identity_weight)):
        log = f'log/cifar100/kd/icpkd/wrn16_2_from_wrn40_2_option_3_2倍_{i}.txt'
        os.system("rm -rf ./resource")
        os.system(f"python ./SST/image_classification_policy.py --log {log}")
        print(f"============================end {i}================================")