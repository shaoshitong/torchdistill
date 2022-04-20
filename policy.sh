#!/bin/bash
id_weight=(1 10 100)
kd_weight=(1 10 100)
policy_weight=(1 5 10)
a=0.1
for k in ${id_weight[*]};
do
  for j in ${kd_weight[*]};
  do
    for i in ${policy_weight[*]};
    do
      idweight=`echo "scale=2;$k*$a"|bc`
      useweight=`echo "scale=2;$i*$a"|bc`
      kdweight=`echo "scale=2;$j*$a"|bc`
      python ./SST/image_classification_policy.py --p_weight_loss $idweight $kdweight $useweight
    done
  done
done
