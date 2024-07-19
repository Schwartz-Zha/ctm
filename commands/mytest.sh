CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440/ctm_exact_sampler_1_steps_006000_itrs_0.999_ema_/

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440_author/ctm_exact_sampler_1_steps_049000_itrs_0.999_ema_/


CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_0030000/ctm_exact_sampler_1_steps_003000_itrs_0.999_ema_/
# FID: 42.251732485206446

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_uncondckpt_0030000/ctm_exact_sampler_1_steps_003000_itrs_0.999_ema_/
# FID: 57.744621466267006

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator_cifar10.py \
    ref-statistics/cifar10-32x32-new.npz \
    ctm-sample-paths/cifar10_CTMDSM_condckpt_006000/ctm_exact_sampler_1_steps_006000_itrs_0.999_ema_/
# 6000 step FID: 24.004894470190777

