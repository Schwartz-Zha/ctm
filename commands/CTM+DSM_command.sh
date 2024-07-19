#!/bin/bash
MODEL_FLAGS="--data_name=imagenet64 --microbatch=11 --global_batch_size=2048 --lr=0.0004 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
CKPT_FLAGS="--out_dir ./ctm-runs/ctm_bs_1440 --dm_sample_path_seed_42=./ctm-sample-paths/samples_ver2 --ref_path=./ckpts/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=./ckpts/edm_imagenet64_ema.pt --data_dir=/data/datasets/ILSVRC2012/raw/ILSVRC2012_img_train/"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS