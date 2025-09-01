import json
from mlx_lm import load, stream_generate
import evaluate
import os
from tqdm import tqdm

cal_rouge_metric = evaluate.load('rouge')

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ===

data_to_test = json.load(open('data_to_test.json'))

idx = 4

# sample = data_to_test[idx]

# ===

# repo = "mlx-community/Llama-3.2-3B-Instruct"
# repo = "mlx_converted/Llama-3.2-3B-Instruct-4bit"

model_list = [
    "mlx_converted/Llama-3.2-3B-Instruct-4bit",
    "./mlx_converted/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit",
    
    "mlx_converted/Qwen2.5-3B-Instruct-4bit",
    "./mlx_converted/converted_qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit"
]

list_time_to_first_token = []
list_token_per_secs = []
list_r1 = []
list_r2 = []
list_rl = []
list_rlsum = []

repo = model_list[1]

# for repo in model_list:

for sample in tqdm(data_to_test):
    
    model, tokenizer = load(repo)

    fixed_system_prompt = "You are a helpful assistant."
    fixed_prompt = "Summarize the following paragraph into a concise highlight in one paragraph, capturing only the key idea."

    article = sample['article']
    hlight = sample['highlights']

    prompt = f"{fixed_prompt}\n{article}"

    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    print("Start:")
    import time
    end = None
    cnt = 0

    output_decoded = ''
    start = time.perf_counter()
    
    for response in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        print(response.text, end="", flush=True)
        cnt += 1
        if not end:
            end = time.perf_counter() - start
            time_to_first_token = end
            
        output_decoded += response.text
        
    end = time.perf_counter() - start        

    eval_rouge = cal_rouge_metric.compute(predictions=[output_decoded], references=[hlight])

    # print()
    # print("Model:", repo)
    # print("Total tokens:", cnt)
    # print("Total seconds:", round(end, 2))
    # print("Time to first token:", round(time_to_first_token, 2))
    # print("Token per seconds:", round(cnt / end, 2))
    # print("ROUGE:", eval_rouge)

    list_time_to_first_token.append(time_to_first_token)
    list_token_per_secs.append(cnt / end)
    list_r1.append(eval_rouge['rouge1'])
    list_r2.append(eval_rouge['rouge2'])
    list_rl.append(eval_rouge['rougeL'])
    list_rlsum.append(eval_rouge['rougeLsum'])
    
import numpy as np

print("Model:", repo)
print("Time to first token", np.mean(list_time_to_first_token), np.std(list_time_to_first_token))
print("Tokens/Seconds", np.mean(list_token_per_secs), np.std(list_token_per_secs))
print("ROUGE-1", np.mean(list_r1), np.std(list_r1))
print("ROUGE-2", np.mean(list_r2), np.std(list_r2))
print("ROUGE-L", np.mean(list_rl), np.std(list_rl))
print("ROUGE-L-Sum", np.mean(list_rlsum), np.std(list_rlsum))