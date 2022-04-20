#!/bin/bash
id_weight=(10 9 8 7 6 5 4 3 2 1)
kd_weight=(0)
policy_weight=(0)
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
