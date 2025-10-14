import json, jsonlines, os
from datasets import Dataset


TAG = "1.5B-after-Iter1"
# TAG = "7B-after-Iter1"

data_path = "data/M-Thinker-1.5B-RL-Iter1-data/train.parquet"
# data_path = "data/M-Thinker-7B-RL-Iter1-data/train.parquet"
have_dataset_RL = Dataset.from_parquet(data_path)
print(len(have_dataset_RL))

RL_data_ids = {}
for i_data in have_dataset_RL:
    cur_id = i_data["extra_info"]["id"]
    cur_lang = i_data["extra_info"]["id"].split("-")[0]
    if cur_lang not in RL_data_ids:
        RL_data_ids[cur_lang] = [cur_id]
    else:
        RL_data_ids[cur_lang].append(cur_id)


for lang in ["ja", "ko", "fr", "pt", "th"]:
    current_data = json.load(open(f"data/Light-R1-SFTData-question-translated-76K/{lang}-temp_0.6-seed_10.json"))
    print(len(current_data))

    train_data_path = f"data/M-Thinker-SFT-data/{lang}-ft-temp_0.6-seed_10.jsonl"

    train_ids = RL_data_ids[lang]
    with jsonlines.open(train_data_path, "r") as fr:
        for item in fr:
            train_ids.append(item["id"])

    other_data = []
    train_data = []
    for item in current_data:
        if item["id"] not in train_ids:
            other_data.append(item)
        else:
            train_data.append(item)
    
    print(lang, len(train_data), len(other_data))

    save_path = f"data/predict-other_data-{TAG}"
    os.makedirs(save_path, exist_ok=True)
    json.dump(other_data, open(f"{save_path}/{lang}-temp_0.6-seed_10.json", 'w', encoding="utf-8"), indent=4, ensure_ascii=False)

    save_path = f"data/predict-train_data-{TAG}" 
    os.makedirs(save_path, exist_ok=True)
    json.dump(train_data, open(f"{save_path}/{lang}-temp_0.6-seed_10.json", 'w', encoding="utf-8"), indent=4, ensure_ascii=False)


