MODEL=llama_ens5
llama_config="params.json"
tokenizer_path="tokenizer.model"
# MASTER_PORT=$((RANDOM % 101 + 20000))
#export OMP_NUM_THREADS=8
#export NCCL_LL_THRESHOLD=0
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
pretrained_path=pretrained_path
exp_name=ChartOCR_448_f16_t0.9_p0.5
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
srun -p Gveval-P1 --gres=gpu:1 --job-name=sam_bf_Chart_debug --cpus-per-task 12 -n1 --ntasks-per-node=1  --quotatype=auto python -u accessory/single_turn_eval_multitask.py \
--llama_type ${MODEL} \
--world_size 1 \
--llama_config ${llama_config} \
--tokenizer_path ${tokenizer_path} \
--pretrained_path ${pretrained_path} \
--batch_size 10 \
--input_size 448 \
--model_parallel_size 1 \
2>&1 | tee -a test.log 2>&1 & \

echo "exp name: $exp_name"
