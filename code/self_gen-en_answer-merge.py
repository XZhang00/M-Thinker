import json, random, argparse


parser = argparse.ArgumentParser()

parser.add_argument("--split", type=int, default=64)
parser.add_argument("--path", type=str, default="data/M-Thinker-1.5B-RL-Iter1-data")
parser.add_argument("--cnt", type=int, default=0)
parser.add_argument("--rollout", type=int, default=1)


args = parser.parse_args()

path = args.path
split = args.split
cnt=args.cnt
rollout=args.rollout


save_data = {}
need_gen_ids = []
for i in range(split):
    cur_path = f"{path}/cnt-{cnt}/en-predict-p{i}.json"

    cur_data = json.load(open(cur_path))
    if cur_data == {}: continue

    for id in list(cur_data.keys()):
        if cur_data[id]["cons_acc_num"] >= 1:
            # save_data[id] = cur_data[id]
            select_gens = []
            for j in range(rollout):
                if cur_data[id][f"cons_correct_{j}"]:
                    assert cur_data[id][f"correct_{j}"]
                    select_gens.append(cur_data[id][f"prediction_{j}"])
            save_data[id] = {}
            save_data[id]["en_question"] = cur_data[id]["en_question"]
            save_data[id]["ground_truth_solution"] = cur_data[id]["ground_truth_solution"]
            save_data[id]["final_prompt"] = cur_data[id]["final_prompt"]
            save_data[id]["prediction_0"] = random.sample(select_gens, 1)[0]


        else:
            need_gen_ids.append(id)


print(len(save_data))
print(len(need_gen_ids))

save_path = f"{path}/cnt-{cnt}-done.json"
json.dump(save_data, open(save_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

save_path = f"{path}/need_gen_ids-{cnt}.json"
json.dump(need_gen_ids, open(save_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)