set -x

# conda activate YOUR_ENV

MODEL_PATH="../../models/Qwen2.5-Math-7B"

DATA_PATH="../data/BBEH-CFT-CausalUnderstanding-p0.jsonl"

OUTPUT_DIR="../output_models/"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ../ms-swift

torchrun \
    --nproc_per_node 8 \
    --standalone \
    swift/cli/sft.py\
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0 \
    --dataset_num_proc 8 \
    --streaming False \
    --strict False \
    --deepspeed zero3 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    \
    --truncation_strategy delete \
    \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --weight_decay 0.05 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --report_to none \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 64 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_only_model True \
    --warmup_ratio 0.05 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False\
    --attn_impl flash_attn \

    # --attn_impl flash_attn \

    # --save_strategy "epoch" \
    # --save_strategy "steps" \
    # --save_steps 109 \
    # --save_total_limit 5 \
    # --deepspeed zero3 \
    # --max_steps -1 \
    # --device_map auto \
    # --override_exist_file True \
    # --eval_strategy None \
    # --custom_dataset_in
