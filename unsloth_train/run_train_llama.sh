export HF_HOME="HF"
export HF_DATASETS_CACHE="${HF_HOME}/cache"
export WANDB_PROJECT=llm_summarization

CKPT_ROOT=checkpoints/textllm/summarization

model_id="meta-llama/Llama-3.2-3B-Instruct"

RUN_NAME=llama32_3b_it_cnn_dailymail_10k_r4_a8_lr2e-4_1ep

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

# ===

RUN_NAME=llama32_3b_it_4bit_cnn_dailymail_10k_r16_a32_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 16 --lora_alpha 32 \
    --lr 2e-4 \
    --batch_size 8 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME

python avg.py $CKPT_ROOT/$RUN_NAME

RUN_NAME=llama32_3b_it_4bit_cnn_dailymail_10k_r16_a16_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 16 --lora_alpha 16 \
    --lr 2e-4 \
    --batch_size 8 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME

python avg.py $CKPT_ROOT/$RUN_NAME

RUN_NAME=llama32_3b_it_4bit_cnn_dailymail_10k_r16_a8_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 16 --lora_alpha 8 \
    --lr 2e-4 \
    --batch_size 8 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME

python avg.py $CKPT_ROOT/$RUN_NAME

python eval.py $CKPT_ROOT/llama32_3b_it_4bit_cnn_dailymail_10k_r16_a32_lr2e-4_1ep/avg_last_3
python eval.py $CKPT_ROOT/llama32_3b_it_4bit_cnn_dailymail_10k_r16_a16_lr2e-4_1ep/avg_last_3
python eval.py $CKPT_ROOT/llama32_3b_it_4bit_cnn_dailymail_10k_r16_a8_lr2e-4_1ep/avg_last_3

# # ======

RUN_NAME=llama32_3b_it_4bit_cnn_dailymail_5k_r4_a8_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 4 --lora_alpha 8 \
    --lr 2e-4 \
    --batch_size 8 \
    --data_size 5000 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAMEB

python avg.py $CKPT_ROOT/$RUN_NAME

RUN_NAME=llama32_3b_it_4bit_cnn_dailymail_20k_r4_a8_lr2e-4_1ep

python train.py \
    --is_log \
    --qlora \
    --model_id $model_id \
    --lora_rank 4 --lora_alpha 8 \
    --lr 2e-4 \
    --batch_size 8 \
    --data_size 20000 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME

python avg.py $CKPT_ROOT/$RUN_NAME

python eval.py $CKPT_ROOT/llama32_3b_it_4bit_cnn_dailymail_5k_r4_a8_lr2e-4_1ep/avg_last_2
python eval.py $CKPT_ROOT/llama32_3b_it_4bit_cnn_dailymail_20k_r4_a8_lr2e-4_1ep/avg_last_3