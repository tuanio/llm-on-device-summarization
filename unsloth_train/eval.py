import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
from datasets import load_dataset
import random
import evaluate
import numpy as np
import sys
import json
from transformers import BitsAndBytesConfig

torch.set_float32_matmul_precision('high')

cal_rouge_metric = evaluate.load('rouge')

model_id = sys.argv[1]

do_load_4bit = False
if len(sys.argv) > 2:
    q4 = sys.argv[2]
    
    if q4 == 'true':
        do_load_4bit = True

fixed_system_prompt = "You are a helpful assistant."
fixed_prompt = "Summarize the following paragraph into a concise highlight in one paragraph, capturing only the key idea."

def load_model(model_id):
    load_in_4bit = '4bit' in model_id or do_load_4bit
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    add_dict = dict()
    if load_in_4bit:
        add_dict = dict(
            quantization_config=quantization_config
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        trust_remote_code=True,
        **add_dict
    ).eval()
    
    print(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, tokenizer

def load_text(model_id, paragraph):
    prompt = f"{fixed_prompt}\n{paragraph}"

    if 'gemma' in model_id:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": fixed_system_prompt},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt},]
            },
        ]
    else:
        messages = [
            {"role": "system", "content": fixed_system_prompt},
            {"role": "user", "content": prompt}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def generate_response(text):
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    total_seconds = time.perf_counter() - start_time
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # tps = len(generated_ids[0]) / total_seconds

    return response, len(generated_ids[0]), total_seconds

cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

subset = 'test'
out_cnn_ds = cnn_ds[subset].shuffle(1211)
# ====

model, tokenizer = load_model(model_id)
tokenizer.pad_token = tokenizer.eos_token


list_truth = []
list_predict = []
list_gen_len = []
list_gen_secs = []

subset = 'test'

total_samples = len(out_cnn_ds)
total_samples = 1000
batch_size = 16

for idx in tqdm(range(0, total_samples, batch_size), total=total_samples // batch_size):
    paragraphs = out_cnn_ds['article'][idx:idx+batch_size]
    highlights = out_cnn_ds['highlights'][idx:idx+batch_size]
    
    texts = [load_text(model_id, para) for para in paragraphs]
    
    model_inputs = tokenizer(texts, padding=True, padding_side='left', return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    if 'gpt' in model_id:
        responses = [i.split("assistantfinal", 1)[-1] for i in responses]

    list_truth.extend(highlights)
    list_predict.extend(responses)
    
    del model_inputs

eval_rouge = cal_rouge_metric.compute(predictions=list_predict, references=list_truth)

model_id = model_id.replace('/', '_')

save_file = f'output/{model_id}.json'

if do_load_4bit:
    save_file = f'output/{model_id}_bnb_4bit.json'

with open(save_file, 'w') as f:
    json.dump(
        dict(
            references=list_truth,
            hypothesis=list_predict,
            metrics=eval_rouge
        ),
        f,
        indent=2,
        ensure_ascii=False
    )

print("ROUGE Scores:", eval_rouge)