from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch


base_id = "meta-llama/Llama-3.2-3B-Instruct" 
lora_id = "checkpoints/textllm/summarization/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep" 
save_path = "checkpoints/textllm/summarization/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit"


base_id = "Qwen/Qwen2.5-3B-Instruct"  
lora_id = "checkpoints/textllm/summarization/qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep"
save_path = "checkpoints/textllm/summarization/qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit"

# load model without quantization config
model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_id)

model = model.merge_and_unload() 

model.save_pretrained(save_path)
tok = AutoTokenizer.from_pretrained(base_id)
tok.save_pretrained(save_path)