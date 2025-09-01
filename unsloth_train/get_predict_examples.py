import json
from datasets import load_dataset
import sys
import random

path1 = "output/checkpoints_textllm_summarization_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_avg_last_3.json"
path2 = "output/checkpoints_textllm_summarization_qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_avg_last_3.json"

data_llama = json.load(open(path1))
data_qwen = json.load(open(path2))

cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='test').shuffle(1211)

idx = random.randint(0, 1000)
# idx = 532

print(idx)
sample = cnn_ds[idx]

truth = data_llama['references'][idx]
predict = data_llama['hypothesis'][idx]

print("Article:", sample['article'])
print('=' * 20)
print("Truth:", truth)
print('=' * 20)
print("Predict:\n", predict)
print("=====")
print("Predict qwen\n", data_qwen['hypothesis'][idx])