export CUDA_VISIBLE_DEVICES=2,3
# export MODEL_NAME="/workspace/home/czhuo/rl_ckpts/Qwen2.5-Coder-3B/grpo/cold_start_multi_turn_sft_data_20250521-20_final/ckpt180"
# export MODEL_NAME='/workspace/home/czhuo/sft_ckpts/Qwen2.5-Coder-3B-Instruct/lora/cold_start/0623_debug_fa2'
export MODEL_NAME='/workspace/home/czhuo/sft_ckpts/Qwen2.5-Coder-3B/lora/cold_start/multi_turn_sft_data_20250521-20/final'

vllm serve $MODEL_NAME --served-model-name agent --port 18000 --gpu-memory-utilization 0.7