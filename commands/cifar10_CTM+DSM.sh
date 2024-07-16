#!/bin/bash
MODEL_FLAGS="--data_name=cifar10 --microbatch=18 --global_batch_size=1440 --lr=0.0004 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=10 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1"
CKPT_FLAGS="--out_dir ./ctm-runs/cifar10_ctm_bs_1440 --dm_sample_path_seed_42=./ctm-sample-paths/samples_ver2 --ref_path=./ref-statistics/cifar10-32x32.npz --teacher_model_path=./ckpts-cifar10/edm-cifar10-32x32-uncond-vp.pkl --data_dir=/data/datasets/ctm-cifar10-32x32/"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

CUDA_LAUNCH_BLOCKING=1 mpiexec -n 8 --hostfile ./commands/hostfile --allow-run-as-root python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS

# torchrun --nnodes=1 --nproc_per_node=8 ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS

# rlaunch --group=midjourney --cpu=64 --gpu=8 --memory=256000 -- bash

# torchrun --nnodes=1 --nproc_per_node=8 python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS