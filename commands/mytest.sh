# rlaunch --group=midjourney --cpu=8 --gpu=1 --memory=40000 -- bash 


### Imagenet

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440/ctm_exact_sampler_1_steps_006000_itrs_0.999_ema_/

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440_author/ctm_exact_sampler_1_steps_049000_itrs_0.999_ema_/


CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_authorckpt_50K/ctm_exact_sampler_1_steps_049000_itrs_0.999_ema_
# Inception Score: 70.66731262207031
# FID: 1.9493514261867517
# sFID: 3.8694287498137783
# Precision: 0.79404
# Recall: 0.5662
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_authorckpt_50K/ctm_exact_sampler_2_steps_049000_itrs_0.999_ema_
# Inception Score: 63.636817932128906
# FID: 1.7392019630183881
# sFID: 3.8073791193312445
# Precision: 0.78232
# Recall: 0.5851
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_001000_50K/ctm_exact_sampler_1_steps_001000_itrs_0.999_ema_
# Inception Score: 1.0486156940460205
# FID: 460.39508526589015
# sFID: 199.17943438178293
# Precision: 0.0
# Recall: 0.0
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_001000_50K/ctm_exact_sampler_2_steps_001000_itrs_0.999_ema_
# Inception Score: 1.0731167793273926
# FID: 437.90950938942626
# sFID: 177.10879391901278
# Precision: 8e-05
# Recall: 0.0
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_005000_50K/ctm_exact_sampler_1_steps_005000_itrs_0.999_ema_
# Inception Score: 1.0817410945892334
# FID: 409.4246387945607
# sFID: 246.20075454233927
# Precision: 0.0
# Recall: 0.0
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_005000_50K/ctm_exact_sampler_2_steps_005000_itrs_0.999_ema_
# Inception Score: 1.0917046070098877
# FID: 409.2992508772535
# sFID: 247.0386076025752
# Precision: 0.0
# Recall: 0.0
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_007000_50K/ctm_exact_sampler_1_steps_007000_itrs_0.999_ema_
# Inception Score: 1.0937525033950806
# FID: 411.9978353025722
# sFID: 250.17546683403634
# Precision: 0.0
# Recall: 0.0
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/imagenet_CTMDSM_condckpt_007000_50K/ctm_exact_sampler_2_steps_007000_itrs_0.999_ema_
# Inception Score: 1.0975966453552246
# FID: 411.7133088193041
# sFID: 251.39565722559348
# Precision: 0.0
# Recall: 0.0


### CIFAR10
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_0030000/ctm_exact_sampler_1_steps_003000_itrs_0.999_ema_/
# FID: 42.251732485206446

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_uncondckpt_0030000/ctm_exact_sampler_1_steps_003000_itrs_0.999_ema_/
# 3000 step FID: 57.744621466267006

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_006000/ctm_exact_sampler_1_steps_006000_itrs_0.999_ema_/
# 6000 step FID: 24.004894470190777

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_012000/ctm_exact_sampler_1_steps_012000_itrs_0.999_ema_/
# FID: 15.329679956336463

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_020000/ctm_exact_sampler_1_steps_020000_itrs_0.999_ema_/
# FID: 12.067576916177359

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_050000/ctm_exact_sampler_1_steps_050000_itrs_0.999_ema_/
# FID: 9.450132538849857
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_050000/ctm_exact_sampler_2_steps_050000_itrs_0.999_ema_/
# FID 8.413614174509405

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSMGAN_condckpt_100000/ctm_exact_sampler_1_steps_100000_itrs_0.999_ema_
# FID: 5.5145640836782945
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSMGAN_condckpt_100000/ctm_exact_sampler_2_steps_100000_itrs_0.999_ema_
# FID: 5.686800981166812

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSMGAN_condckpt_100000_50K/ctm_exact_sampler_1_steps_100000_itrs_0.999_ema_
# FID: 2.000095483598784
CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSMGAN_condckpt_100000_50K/ctm_exact_sampler_2_steps_100000_itrs_0.999_ema_
# FID: 2.245542972791668   ---> ???