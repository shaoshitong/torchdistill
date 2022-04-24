import os,sys

if __name__=="__main__":
    negative_identity_weight=[0.1,0,0,0,0,0]
    positive_identity_weight=[0,0.1,0,0,0,0]
    negative_classes_weight=[0,0,0.1,0,0,0]
    positive_classes_weight=[0,0,0,0.1,0,0]
    negative_policy_weight=[0,0,0,0,0.1,0]
    positive_policy_weight=[0,0,0,0,0,0.1]
    
    for i in range(len(negative_identity_weight)):
        os.system(f"python ./SST/image_classification_policy.py --negative_weight_loss {negative_identity_weight[i]} {negative_classes_weight[i]} {negative_policy_weight[i]} --positive_weight_loss {positive_identity_weight[i]} {positive_classes_weight[i]} {positive_policy_weight[i]}")
        print(f"============================end {i}================================")
