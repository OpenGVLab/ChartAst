torchrun --nproc-per-node=1 --master-port=12345 /root/cloud_disk2/LLaMA2-Accessory/accessory/demos/single_turn_mm.py --llama_config /root/cloud_disk2/LLaMA2-Accessory/params.json \
--tokenizer_path /root/cloud_disk2/LLaMA2-Accessory/tokenizer.model \
--pretrained_path /root/cloud_disk2/LLaMA2-Accessory/accessory/exps/finetune/mm/output/finetune/mm/chart_multitask_instruction_tuning_gpu8_ft_stock/epoch0 $QUANT --llama_type llama_ens5 > /root/cloud_disk2/LLaMA2-Accessory/accessory/demos/start1.log 2>&1 & \
