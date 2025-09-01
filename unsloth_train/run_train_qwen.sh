export HF_HOME="HF"
export HF_DATASETS_CACHE="${HF_HOME}/cache"
export WANDB_PROJECT=llm_summarization

CKPT_ROOT=checkpoints/textllm/summarization

model_id="Qwen/Qwen2.5-3B-Instruct"
RUN_NAME=qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 4 --lora_alpha 8 \
    --lr 2e-4 \
    --batch_size 8 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME

python avg.py $CKPT_ROOT/$RUN_NAME
python eval.py $CKPT_ROOT/$RUN_NAME/avg_last_3