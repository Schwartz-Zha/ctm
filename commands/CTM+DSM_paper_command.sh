#!/bin/bash
MODEL_FLAGS="--use_fp16=True --use_d_fp16=True --num_heun_step=20 --start_scales=40 --data_name=imagenet64 --image_size=64 --microbatch=16 --global_batch_size=2048 --lr=0.000008 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1"
CKPT_FLAGS="--out_dir ./ctm-runs/imagenet_CTMDSM_cond/ --dm_sample_path_seed_42=./ctm-sample-paths/samples_ver2 --ref_path=./ckpts/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=./ckpts/edm_imagenet64_ema.pt --data_dir=/data/datasets/ILSVRC2012/raw/ILSVRC2012_img_train/ "
CKPT_FLAGS="${CKPT_FLAGS} --resume_checkpoint=./ctm-runs/imagenet_CTMDSM_cond/model008000.pt"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --hostfile ./commands/hostfile --allow-run-as-root python ./code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS