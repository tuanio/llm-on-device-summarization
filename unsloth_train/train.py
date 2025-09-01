from unsloth import FastLanguageModel
import torch
import sys
import os
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only

import glob
    
def main(args):

    max_seq_length = 4096
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    seed = 1211

    model_id = args.model_id

    load_in_4bit = args.qlora
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    gemma_config = dict()
    if 'gemma' in model_id:
        gemma_config = dict(
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # SHould leave on always!
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        **gemma_config
    )
    
    if 'llama' in model_id:
        chat_template = "llama-3.1"
    elif 'gemma' in model_id:
        chat_template = "gemma-3"
    else:
        chat_template = "chatml"

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    
    if args.use_flash_attn:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    fixed_system_prompt = "You are a helpful assistant."
    fixed_prompt = "Summarize the following paragraph into a concise highlight in one paragraph, capturing only the key idea."

    def formatting_prompts_func(examples):
        
        paragraph = examples['article']
        highlights = examples['highlights']
        
        texts = []
        
        for para, hlight in zip(paragraph, highlights):
            prompt = f"{fixed_prompt}\n{para}"
            
            if 'gemma' in model_id:
                messages = [
                    {
                        "role": "system",
                        "content": fixed_system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": hlight
                    }
                ]
            else:
                messages = [
                    {"role": "system", "content": fixed_system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": hlight}
                ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            if 'gemma' in model_id:
                text = text.removeprefix('<bos>')
            
            texts.append(text)

        return { "text" : texts, }

    print("LOad dataset from start")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    
    # data_size = 50_000
    data_size = args.data_size
    dataset = dataset.shuffle(seed).select(range(data_size))
    
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    lr = args.lr

    batch_size = args.batch_size
    
    out_dir = os.path.join(args.ckpt_root, args.run_name)

    os.system('mkdir -p ' + out_dir)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        packing = True, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 1,
            # warmup_steps = 5,
            warmup_ratio=0.1,
            # num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 60,
            num_train_epochs=1,
            learning_rate = lr,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 1e-5,
            lr_scheduler_type = "linear",
            seed = seed,
            output_dir = out_dir,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=2,
            run_name=args.run_name,
            save_total_limit=3,
            report_to = "wandb" if args.is_log else "none", # Use this for WandB etc
        ),
    )

    if 'llama' in model_id:
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'gemma' in model_id:
        instruction_part = "<start_of_turn>user\n"
        response_part = "<start_of_turn>model\n"
    else:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
        
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part
    )
    
    resume_from_checkpoint = len(glob.glob(out_dir + '/checkpoint-*')) > 0

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model(out_dir)  # saves tokenizer + model
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True,
                        help="Model identifier (e.g., 'facebook/opt-1.3b')")
    parser.add_argument("--run_name", type=str, default="experiment_01",
                        help="Unique name for this run (used in logging/checkpointing)")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints",
                        help="Root directory to save checkpoints")

    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Root directory to save checkpoints")
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument("--data_size", type=int, default=10_000)
    
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--is_log', action='store_true')
    parser.add_argument('--qlora', action='store_true')

    args = parser.parse_args()

    main(args)
