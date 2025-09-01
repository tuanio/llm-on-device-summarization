import json
from datasets import load_dataset

cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0", subset='test')

out_cnn_ds = cnn_ds.shuffle().select(range(10))

save_data = []
for sample in out_cnn_ds:
    save_data.append(sample)
    
with open('data_to_test_speed.json', 'w') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)