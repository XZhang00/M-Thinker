import json, random, os
from datasets import Dataset
import json, random, argparse


parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, default="data/M-Thinker-1.5B-RL-Iter1-data")

args = parser.parse_args()




save_dir = args.save_dir
data_path = f"{save_dir}-ori/train.parquet"


data = {}
data0 = json.load(open(f"{save_dir}/cnt-0-done.json"))
data1 = json.load(open(f"{save_dir}/cnt-1-done.json"))
data2 = json.load(open(f"{save_dir}/cnt-2-done.json"))
data3 = json.load(open(f"{save_dir}/cnt-3-done.json"))


data.update(data0)
data.update(data1)
data.update(data2)
data.update(data3)



print(len(data))



have_dataset = Dataset.from_parquet(data_path)

ans = 0
final_data = []
for i_data in have_dataset:
    cur_id = i_data["extra_info"]["id"].split("-")[-1]
    if cur_id in data:
        assert i_data["extra_info"]["en_question"] == data[cur_id]["en_question"]
        i_data["extra_info"]["en_answer"] = data[cur_id]["prediction_0"]
        assert "<think>\n" in i_data["extra_info"]["en_answer"] and "</think>" in i_data["extra_info"]["en_answer"] and len(i_data["extra_info"]["en_answer"][len("<think>\n"):].split("</think>")) == 2
    else:
        i_data["extra_info"]["en_answer"] = None
        ans += 1 

    final_data.append(i_data)

print(len(final_data), ans, ans/len(final_data)*100)
train_dataset = Dataset.from_list(final_data)
test_dataset = Dataset.from_list(final_data[:64])


train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))


