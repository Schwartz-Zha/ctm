#!/bin/bash
MODEL_FLAGS="--use_fp16=True --use_d_fp16=False --num_heun_step=20 --start_scales=40 --data_name=imagenet64 --image_size=64 --microbatch=16 --global_batch_size=2048 --lr=0.000008 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1"
CKPT_FLAGS="--out_dir ./ctm-runs/imagenet_CTMDSMGAN_cond/ --dm_sample_path_seed_42=./ctm-sample-paths/samples_ver2 --ref_path=./ckpts/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=./ckpts/edm_imagenet64_ema.pt --data_dir=/data/datasets/ILSVRC2012/raw/ILSVRC2012_img_train/ "
RESUME_FLAGS="--resume_checkpoint=./ctm-runs/imagenet_CTMDSM_cond/model010000.pt"
# RESUME_FLAGS=" "
GAN_FLAGS="--d_lr=0.002 --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2"
# GAN_FLAGS=" "
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --hostfile ./commands/hostfile --allow-run-as-root python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS $RESUME_FLAGS $GAN_FLAGS


# rlaunch --group=midjourney --cpu=64 --gpu=8 --memory=256000 --max-wait-duration=72h -- bash
# bash commands/CTM+DSM+GAN_paper_command.sh