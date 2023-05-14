#!/bin/bash

cls_id=1
ckpt=train_log/lm/checkpoints/
n_gpu=2  # number of gpu to use
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu -state=train -dataset_name=lmo -cls_id=$cls_id -checkpoint=$ckpt
