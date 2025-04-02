#!/bin/bash

# Set training parameters
lr=1e-5
epochs=5
global_batch_size=16
per_device_batch_size=1
weight_decay=1e-4
train_dataset_name="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-tokenized-120325"
uid="$(date +%Y%m%d_%H%M%S)"
model_name="Qwen/Qwen2.5-7B-Instruct"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --global_batch_size) global_batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        --per_device_batch_size) per_device_batch_size="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Get node information
node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600


# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((global_batch_size/(gpu_count * nnodes)))

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node IP: $head_node_ip"

# Launch distributed training using srun
run_name="qwen_${train_dataset_name}_bs${global_batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"

# NOTE: if we start the job with srun, no srun in the script is needed. If we start the job with sbatch, we need to use srun.
torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    src/train/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=${per_device_batch_size} \
    --per_device_eval_batch_size=${per_device_batch_size} \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --train_file_path="${train_dataset_name}" \
    --model_name=$model_name \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="src/train/fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="outputs/${run_name}" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}' \
    --report_to='wandb'

