import json, jsonlines, os


for lang in ["ja", "ko", "fr", "pt", "th"]:
    current_data = json.load(open(f"data/Light-R1-SFTData-question-translated-76K/{lang}-temp_0.6-seed_10.json"))

    train_data_path = f"data/M-Thinker-SFT-data/{lang}-ft-temp_0.6-seed_10.jsonl"
    TAG = "cold-start-sft"
    train_ids = []
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




