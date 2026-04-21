export CHECKPOINT_DIR="/workspace/home/czhuo/Agent-R1/checkpoints/kbqa_qwen2.5_coder_3B_cold_start_final_ckpt_0604_02/grpo-qwen2.5-coder-7b-cold-start/global_step_180/actor"
export HF_MODEL_PATH="/workspace/home/czhuo/sft_ckpts/Qwen2.5-Coder-3B/lora/cold_start/multi_turn_sft_data_20250521-20/final"
export TARGET_DIR="/workspace/home/czhuo/rl_ckpts/Qwen2.5-Coder-3B/grpo/cold_start_multi_turn_sft_data_20250521-20_final/ckpt180"

python3 verl/scripts/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CHECKPOINT_DIR --target_dir $TARGET_DIR