export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
dataset=alpaca_en_demo
# export WANDB_DISABLED=True 
export ALLLO_EXTRA_ARGS=True

export MASTER_ADDR=localhost
export MASTER_PORT=23556

ds_config_path=./examples/deepspeed/ds_z3_config.json

model_name_or_path=/home/yyx/models/Qwen3-30B-A3B

finetuning_type=full

seq_length=1024

output_dir=./output/qwen3-30b-alpaca-en-demo

torchrun --nproc_per_node=8 src/train.py \
    --num_train_epochs 1 \
    --overwrite_cache True\
    --report_to tensorboard \
    --logging_dir ./logs \
    --deepspeed $ds_config_path \
    --gradient_checkpointing yes \
    --trust_remote_code \
    --disable_shuffling \
    --stage pt \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
    --template qwen3 \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 20000000\
    --save_steps 20000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss true \
    --seed 42 \
    --bf16 \
    --ddp_timeout 180000000 \
    --max_steps 200 2>&1 | tee cu_log
