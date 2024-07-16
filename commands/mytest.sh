CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440/ctm_exact_sampler_1_steps_006000_itrs_0.999_ema_/

CUDA_VISIBLE_DEVICES=0 python code/evaluations/evaluator.py \
    ref-statistics/VIRTUAL_imagenet64_labeled.npz \
    ctm-sample-paths/ctm_bs_1440_author/ctm_exact_sampler_1_steps_049000_itrs_0.999_ema_/