#!/bin/bash
# (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21),
for i in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}
do
    cls_id=$i
    ckpt=train_log/ycb/checkpoints/
    n_gpu=2  # number of gpu to use
    python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu -state=train  -dataset_name=ycbv -cls_id=$cls_id -checkpoint=$ckpt
done