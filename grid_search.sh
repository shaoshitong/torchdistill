#!/bin/bash
kd_weight=(1 2)
policy_weight=(0 1 2 3)
a=0.25
b=0.5
for j in $kd_weight
do
  for i in $policy_weight
  do
    useweight=`echo "scale=2;$i*$a"|bc`
    kdweight=`echo "scale=2;$j*$b"|bc`
    python ./SST/image_classification_policy_grid_search.py --weight_loss 1.0 $kdweight $useweight
  done
done
