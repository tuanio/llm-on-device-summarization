from mlx_lm import convert

import torch
import json
import numpy as np
from safetensors import safe_open

def convert_hf_lora_to_mlx(hf_adapter_path, hf_config_path, mlx_output_path):
    # 1. Load Hugging Face LoRA Adapters
    with safe_open(hf_adapter_path, framework="pt") as f:
        hf_lora_weights = {key: f.get_tensor(key) for key in f.keys()}

    with open(hf_config_path, 'r') as f:
        hf_lora_config = json.load(f)

    mlx_lora_data = {}

    # 2. Map Weights and Parameters
    # This is a simplified example; actual mapping depends on MLX's internal structure
    for key, value in hf_lora_weights.items():
        if "lora_A" in key:
            # Example: MLX might expect A transposed or in a different shape
            mlx_lora_data[key.replace("lora_A", "lora_a")] = value.cpu().numpy()
        elif "lora_B" in key:
            mlx_lora_data[key.replace("lora_B", "lora_b")] = value.cpu().numpy()

    # Add other necessary config parameters if MLX expects them in the saved file
    mlx_lora_data["lora_alpha"] = hf_lora_config.get("lora_alpha")
    mlx_lora_data["r"] = hf_lora_config.get("r")

    # 3. Save in MLX Format
    np.savez(mlx_output_path, **mlx_lora_data)
    print(f"Converted LoRA adapters saved to {mlx_output_path}")

# Usage example (replace with your actual paths)
# hf_adapter_path = "path/to/your/hf_lora_adapter/adapter_model.safetensors"
# hf_config_path = "path/to/your/hf_lora_adapter/adapter_config.json"
# mlx_output_path = "path/to/save/mlx_lora_adapters.npz"
# convert_hf_lora_to_mlx(hf_adapter_path, hf_config_path, mlx_output_path)

# repo = "mlx-community/Llama-3.2-3B-Instruct"
# mlx_path = "mlx_converted/Llama-3.2-3B-Instruct-4bit"

# repo = "./local_repo/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep"
# mlx_path = "./mlx_converted/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_4bit"

# convert(repo, mlx_path=mlx_path, quantize=True)

# repo = "./local_repo/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep"
# mlx_path = "./mlx_converted/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_4bit"

# convert(repo, mlx_path=mlx_path, quantize=True)

# repo = "./local_repo/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora"
# mlx_path = "./mlx_converted/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora"


# repo = "./local_repo/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit"
# mlx_path = "./mlx_converted/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit"

# convert(repo, mlx_path=mlx_path, quantize=True)

# repo = "mlx-community/Qwen2.5-3B-Instruct-bf16"
# mlx_path = "mlx_converted/Qwen2.5-3B-Instruct-4bit"
# convert(repo, mlx_path=mlx_path, quantize=True)


repo = "./local_repo/qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit"
mlx_path = "./mlx_converted/converted_qwen_qwen2_5_3b_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit"

convert(repo, mlx_path=mlx_path, quantize=True)


# root_lora_path = 'local_repo/llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep'
# convert_hf_lora_to_mlx(
#     hf_adapter_path=root_lora_path + '/adapter_model.safetensors',
#     hf_config_path=root_lora_path + '/adapter_config.json',
#     mlx_output_path=root_lora_path + '/mlx_adapter_converted'
# )