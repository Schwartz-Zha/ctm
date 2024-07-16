#!/bin/bash
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8
MODEL_FLAGS="--num_heun_step=20 --discriminator_weight=1.0 --gan_specific_time=True --microbatch=11 --global_batch_size=4224 --lr=0.000008 --data_name=imagenet64 --class_cond=True --start_ema=0.999 --gan_different_augment=True --eval_interval=500 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=10 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --num_workers=32 --log_interval=1"
CKPT_FLAGS="--out_dir ./ctm-runs/M20_w1_v2 --ref_path=./ckpts/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=./ckpts/edm_imagenet64_ema.pt --data_dir=/data/datasets/ILSVRC2012/raw/ILSVRC2012_img_train/"

mpiexec -n 8 --hostfile ./commands/hostfile python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS 