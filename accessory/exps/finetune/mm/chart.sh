#!/bin/bash
#proxy_on

pretrained_path=./epoch0
pretrained_type=consolidated
llama_config="./params.json"
tokenizer_path="./tokenizer.model"
data_config=./chart_multitask_mixed_othertypebasetype.yaml

data_parallel=sdp
model_parallel=2
GPU=16
time=$(date "+%Y%m%d-%H%M%S")
# port=$((((RANDOM<<15)|RANDOM) % 49152 + 1000 )) 
echo $time
exp_name=finetune/mm/chart_multitask_instruction_tuning_gpu16_mixed_othertypebasetype
echo "exp name: $exp_name"
mkdir -p /SPHINX/LLaMA2-Accessory/shpinx_log/output/"$exp_name"

#srun -p Gveval-P1 --gres=gpu:8 --job-name=Chart_load_pretrain --cpus-per-task 12 -n8 --ntasks-per-node=8  --quotatype=spot torchrun --nproc_per_node 8 main_finetune.py \

srun -p Gveval-P1 --gres=gpu:8 --job-name=Chart_load_pretrain --cpus-per-task 12 -n$GPU --ntasks-per-node=8  --quotatype=reserved python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.03 \
--batch_size 4 --accum_iter 8 --num_workers 4 \
--max_words 2048 \
--input_size 448 \
--lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_ens5 --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
--image_transform padded_resize \
2>&1 | tee -a /SPHINX/LLaMA2-Accessory/shpinx_log/output/"$exp_name"/output_$time.log 2>&1 & \
#2>&1 | tee -a /mnt/petrelfs/share_data/shaowenqi/shpinx_log/output/"$exp_name"/output.log &

echo "exp name: $exp_name"