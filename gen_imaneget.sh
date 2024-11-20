export LD_LIBRARY_PATH=/data/ctm/venv_ctm/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python code/image_gen.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir gen_imgs/imagenet/ --model_path=ctm_pretrained/ema_0.999_049000.pt --training_mode=ctm --class_cond=True --eval_num_samples=50000 --batch_size=200 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=1
# CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python code/image_gen.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir gen_imgs/imagenet/ --model_path=ctm_pretrained/ema_0.999_049000.pt --training_mode=ctm --class_cond=True --eval_num_samples=50000 --batch_size=200 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=2

# CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python code/image_gen.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir gen_imgs/imagenet/ --model_path=ctm_pretrained/ema_0.999_049000.pt --training_mode=ctm --class_cond=True --eval_num_samples=50000 --batch_size=200 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=4