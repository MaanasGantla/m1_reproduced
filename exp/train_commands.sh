# qwen2.5-7b-instruct
# m1k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-1k \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False

# gradient_checkpoint=True, 40GB per GPU with 8 GPUs
# gradient_checkpoint=False, 50GB per GPU with 8 GPUs

# m23k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m23k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-23k \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False


# qwen2.5-32b-instruct
# 1k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-1k \
    --exp_name 32b \
    --gradient_checkpointing True \
    --use_flash_attention_2 False \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --nnodes 2 \
    --head_node_ip ???

# gradient_checkpointing=True, no cpu offload, 50-60GB per GPU with 16 GPUs
