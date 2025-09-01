export HF_HOME="HF"
export HF_DATASETS_CACHE="${HF_HOME}/cache"

python eval.py "meta-llama/Llama-3.2-3B-Instruct" true
python eval.py "Qwen/Qwen2.5-3B-Instruct" true