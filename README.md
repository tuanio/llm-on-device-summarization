# On-Device Summarization with LLMs

This repository contains an implementation of **text summarization using lightweight Large Language Models (LLMs)** on Apple devices such as the **MacBook Pro M1 (16GB)**.  
The project demonstrates how to fine-tune, evaluate, and deploy compact LLMs for efficient **on-device summarization**.

---

## 📌 Overview

Text summarization is the task of producing a shorter version of a given text while retaining its essential meaning.  
This project explores both **extractive** and **abstractive** summarization methods using small LLMs (≤ 4B parameters) optimized for on-device performance.

---

## 🎯 Objectives

- Implement text summarization with small LLMs.
- Fine-tune models efficiently with **QLoRA**.
- Evaluate models using **ROUGE metrics**.
- Deploy the models with **MLX** for on-device inference.

---

## ⚙️ Implementation Plan

1. **Modeling**: Choose community-proven LLMs ≤ 3B parameters.  
   - [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
   - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

2. **Dataset**:  
   - [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)  
   - ~313k news articles with human-written highlights  
   - Example format:
     ```json
     {
       "article": "Full news article text",
       "highlights": "Concise human-written summary"
     }
     ```

3. **Training**:  
   - Fine-tune with **QLoRA (4-bit quantization + LoRA adapters)**  
   - Default config:
     - Learning rate: `2e-4`  
     - Batch size: `8`  
     - Epochs: `1` (10k samples)  
     - Rank/Alpha tuning (ablation study included)

4. **Evaluation**:  
   - Metrics: **ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-Sum**  
   - Chosen: **ROUGE-L-Sum** (best represents multi-sentence quality)

5. **Deployment**:  
   - Fuse base LLM with QLoRA adapters  
   - Convert to **MLX format**  
   - Quantize to **4.5-bit** for efficient inference on MacBook Pro M1

---

## 🔬 Experiments

### 📊 Model Comparison
| Model | Params | ROUGE-L-Sum (Base) | ROUGE-L-Sum (Finetuned) | Rel. Improvement |
|-------|--------|---------------------|-------------------------|------------------|
| Llama-3.2-3B-Instruct | 3B | 28.54 | 37.66 | +31.96% |
| Qwen2.5-3B-Instruct   | 3B | 26.66 | 36.10 | +35.41% |

- Llama-3.2-3B performed slightly better overall.
- Both models showed significant gains after fine-tuning.

### ⚖️ Ablation Studies
- **Rank vs Alpha**: Alpha has a stronger effect; best when `alpha ≈ 2 × rank`.
- **Data size**:  
  - 5k samples → weak learning  
  - 10k samples → stable performance  
  - 20k samples → minimal additional gain

### 💻 On-Device Inference (MacBook Pro M1, 16GB)
- Precision: **mlx-4.5bit**  
- Time to first token: ~2.8s  
- Tokens/sec: ~20–27  
- Memory usage: ~3–4 GB  
- Human evaluation: Smooth streaming summaries within 2–4s

---

## 📝 Example Summarization

**Input (news article excerpt):**
> Ryan Bertrand believes he deserves to be higher up the England pecking order and has pledged...  

**Ground Truth**
> Ryan Bertrand was fifth left back to be called up by England last month .
> Bertrand only included after injuries to Leighton Baines, Luke Shaw and Danny Rose .
> He replaced Kieran Gibbs in second half away in Italy .
> Bertrand belives he should be higher up the England pecking order .

**Model Output (Llama-3.2-3B, finetuned):**
> Ryan Bertrand was called up to Roy Hodgson's England squad last month.
> The 25-year-old was included in the squad for the matches against Lithuania and Italy.
> Bertrand has been in fine form for Southampton and hopes to prove himself.
> He is currently fourth in the pecking order for left-back spot.


---

## ⚡ Challenges

- Qwen2.5 occasionally produced **hallucinations** after quantization.
- Limited dataset size may cause bias when generalizing across domains.

---

## ✅ Conclusion

- Small LLMs (≤ 3B) can achieve **strong summarization performance on-device**.  
- Fine-tuned models outperform base models significantly.  
- Deployment via **MLX** makes them practical for real-world use on MacBooks.

---

## 🚀 Getting Started

```bash
# Clone repo
git clone https://github.com/tuanio/llm-on-device-summarization.git
cd llm-on-device-summarization

# Install dependencies
pip install -r requirements.txt
```

Use `train.py` to fine-tune a model with QLoRA:

Example of `unsloth_train/run_train_llama.sh`

```bash
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
    --lora_rank 4 \
    --lora_alpha 8 \
    --lr 2e-4 \
    --batch_size 8 \
    --data_size 10000 \
    --ckpt_root $CKPT_ROOT \
    --run_name $RUN_NAME
```

# Evaluate
```bash
bash run_eval.sh
```

# Run inference
```bash
cd mlx_eval
python stream_mlx.py --model tuanio/converted_llama32_3b_it_4bit_cnn_dailymail_10k_r4_a8_lr2e-4_1ep_fused_lora_dequant4bit_mlx4bit
```