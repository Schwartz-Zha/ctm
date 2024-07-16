from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
import pickle
import torch
import argparse

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name])

def create_argparser():
    defaults = dict(data_name='cifar10')
    # defaults = dict(data_name='imagenet64')
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    defaults.update()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():

    args = create_argparser().parse_args()
    
    teacher_model, _ = create_model_and_diffusion(args, teacher=True)

    # print(args.model_type)
    print(args.data_name)

    # breakpoint()

    teacher_model_path='./ckpts-cifar10/edm-cifar10-32x32-uncond-vp.pkl'

    with open(teacher_model_path, 'rb') as f:
        data = pickle.load(f)
    
    tm_dict = dict(teacher_model.model.named_parameters())

    for name, para in data['ema'].named_parameters():
        tm_param = tm_dict[name]
        if (para == tm_param).all():
            print(name) 

    # f = open("data.txt", "x")
    # for name, _ in data['ema'].named_parameters():
    #     f.write(f"{name}" + '\n')
    # f.close()
    
    # f = open("data_tm.txt", "x")
    # for name, _ in teacher_model.model.named_parameters():
    #     f.write(f"{name}"+ '\n')
    # f.close()
    
    # breakpoint() 

    # print(data['ema'].model.enc['32x32_block0'].conv1.weight)

    # breakpoint()

    # print(teacher_model.model.model.enc['32x32_block0'].conv1.weight)

    # breakpoint()

    # copy_params_and_buffers(src_module=data['ema'], dst_module=teacher_model.model, require_all=False)

    # print(teacher_model.model.model.enc.32x32_block0.conv1.weight)

    # breakpoint()


if __name__ == "__main__":
    main()
