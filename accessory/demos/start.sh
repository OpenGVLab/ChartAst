srun -p gvembodied --gres=gpu:1 --job-name=bf_Chart_debug --cpus-per-task 12 -n1 --ntasks-per-node=1 --quotatype=spot python -u single_turn_mm.py --llama_config params.json \
--tokenizer_path tokenizer.model \
--pretrained_path model_path $QUANT --llama_type llama_ens5
