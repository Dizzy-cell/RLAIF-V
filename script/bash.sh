export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=1,2
export MASTER_ADDR=localhost
export MASTER_PORT=123456

export WANDB_MODE=offilne
torchrun --nproc_per_node=2 train_llava15_lora.py
torchrun --nproc_per_node=1 --standalone --nnnodes=1 train_llava15_lora.py
