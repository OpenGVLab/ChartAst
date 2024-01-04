#!/bin/bash
#proxy_on

# pretrained_path=/mnt/petrelfs/share_data/gaopeng/shared_env/load_pdf_pretrained/pdf_only_epoch0-iter9999
# pretrained_path=/mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/accessory/exps/finetune/mm/output/finetune/mm/chart_ocr_gpu24/epoch0-iter24999
pretrained_path=/mnt/petrelfs/share_data/gaopeng/dialog_data2_llamaEns5_13B_loadLongSphinxPre10K/epoch0
# pretrained_path=/mnt/petrelfs/share_data/gaopeng/shared_env/dialog_llava1.5noboxRef4v2Cwbflicker30k_llamaEns5_13B
pretrained_type=consolidated
llama_config="/mnt/petrelfs/share_data/llm_llama2/llama2_raw/llama-2-13b/params.json"
tokenizer_path="/mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/tokenizer.model"
data_config=/mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/accessory/configs/data/finetune/mm/chart_multitask_mixed_othertypebasetype.yaml

data_parallel=sdp
model_parallel=2
GPU=16
time=$(date "+%Y%m%d-%H%M%S")
# port=$((((RANDOM<<15)|RANDOM) % 49152 + 1000 )) 
echo $time
exp_name=finetune/mm/chart_multitask_instruction_tuning_gpu16_mixed_othertypebasetype
echo "exp name: $exp_name"
mkdir -p /mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/shpinx_log/output/"$exp_name"

#srun -p Gveval-P1 --gres=gpu:8 --job-name=Chart_load_pretrain --cpus-per-task 12 -n8 --ntasks-per-node=8  --quotatype=spot torchrun --nproc_per_node 8 main_finetune.py \

srun -p Gveval-P1 --gres=gpu:8 --job-name=Chart_load_pretrain --cpus-per-task 12 -n$GPU --ntasks-per-node=8  --quotatype=reserved --async -o /mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/shpinx_log/output/"$exp_name"/output_$time.log python -u /mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/accessory/main_finetune2.py \
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
2>&1 | tee -a /mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/shpinx_log/output/"$exp_name"/output_$time.log 2>&1 & \
#2>&1 | tee -a /mnt/petrelfs/share_data/shaowenqi/shpinx_log/output/"$exp_name"/output.log &

echo "exp name: $exp_name"