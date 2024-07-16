#!/bin/bash
MODEL_FLAGS="--data_name=imagenet64 --microbatch=18 --global_batch_size=180 --lr=0.0004 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1"
CKPT_FLAGS="--out_dir ./ctm-runs/ctm_bs_1440 --dm_sample_path_seed_42=./ctm-sample-paths/samples_ver2_debug2/ --ref_path=./ckpts/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=./ckpts/edm_imagenet64_ema.pt --data_dir=/data/datasets/ILSVRC2012/raw/ILSVRC2012_img_train/"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

# mpiexec -n 1 --hostfile ./commands/hostfile --allow-run-as-root python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS

python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS


# rlaunch --group=midjourney --cpu=64 --gpu=8 --memory=256000 -- bash

# torchrun --nnodes=1 --nproc_per_node=8 ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS