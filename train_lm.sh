#!/bin/bash

for i in {1,}   ###{1,5,6,8,9,10,11,12}#{5,10,11,12}
do
    cls_id=$i
    ckpt=train_log/lm/checkpoints/
    n_gpu=2  # number of gpu to use
    python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu -state=train  -dataset_name=lmo -cls_id=$cls_id -checkpoint=$ckpt
done