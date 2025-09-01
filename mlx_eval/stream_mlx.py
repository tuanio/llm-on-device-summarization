import json
from mlx_lm import load, stream_generate
import evaluate
import os
from tqdm import tqdm
import argparse

# model_list = [
#     "mlx_converted/Llama-3.2-3B-Instruct-4bit",
#     "tuanio/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit",
#     # "./../../mlx_converted/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit",
    
#     "mlx_converted/Qwen2.5-3B-Instruct-4bit",
#     "tuanio/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit"
#     # "./mlx_converted/converted_qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit"
# ]

# repo = model_list[1]

def main(args):
    repo = args.model
    
    print("Load model: ", repo)
    model, tokenizer = load(repo)

    fixed_system_prompt = "You are a helpful assistant."
    fixed_prompt = "Summarize the following paragraph into a concise highlight in one paragraph, capturing only the key idea."

    print("Input file that you want to summarize:\n")
    file = input("=> Input File: ")

    article = open(file).read().split('\n')
    article = ' '.join(article)

    prompt = f"{fixed_prompt}\n{article}"

    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    import time
    end = None
    cnt = 0

    output_decoded = ''
    start = time.perf_counter()

    print("=" * 10 + " Summary " + "=" * 10)
    for response in stream_generate(model, tokenizer, prompt, max_tokens=4096):
        print(response.text, end="", flush=True)
        cnt += 1
        if not end:
            end = time.perf_counter() - start
            time_to_first_token = end
            
        output_decoded += response.text
        
    end = time.perf_counter() - start        

    print("\n" + "=" * 29)
    print()
    # print("Model:", repo)
    print("Total tokens:", cnt)
    print("Total seconds:", round(end, 2))
    print("Time to first token:", round(time_to_first_token, 2))
    print("Token per seconds:", round(cnt / end, 2))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with a model argument")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the model to use"
    )
    args = parser.parse_args()
    
    main(args)