import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import glob
from tqdm import tqdm
import sys
import os
from transformers import BitsAndBytesConfig

ckpt_root = sys.argv[1]

load_in_4bit = '4bit' in ckpt_root

# Danh sách các checkpoint cùng loại
checkpoints = glob.glob(ckpt_root + '/checkpoint-*')

# Load model đầu tiên làm "khung"
model = AutoModelForCausalLM.from_pretrained(checkpoints[0])
state_dict = model.state_dict()

# Khởi tạo accumulator
avg_state_dict = {k: torch.zeros_like(v, dtype=torch.bfloat16) for k, v in state_dict.items()}

# Cộng dồn weights từ từng checkpoint
for ckpt in tqdm(checkpoints):
    m = AutoModelForCausalLM.from_pretrained(ckpt)
    sd = m.state_dict()
    for k in avg_state_dict:
        avg_state_dict[k] += sd[k].to(dtype=torch.bfloat16)

# Chia trung bình
for k in avg_state_dict:
    avg_state_dict[k] /= len(checkpoints)

# Load lại vào model
model.load_state_dict(avg_state_dict, strict=False)

# Save checkpoint trung bình
save_dir = ckpt_root + f"/avg_last_{len(checkpoints)}"

os.system('mkdir -p ' + save_dir)

model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
tokenizer.save_pretrained(save_dir)

print(f"Averaged checkpoint saved to {save_dir}")
