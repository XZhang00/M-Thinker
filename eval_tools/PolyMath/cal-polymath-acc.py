import json, jsonlines, argparse
from numpy import mean

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="")

args = parser.parse_args()


path = f"logs-eval/PolyMath-temp_0.9/{args.model_name}/score-eval.jsonl"


data = {}
with open(f"{path}", "r", encoding="utf-8") as files:
    for line in files:
        item = json.loads(line)
        key = list(item.keys())[0]
        data[key] = item[key]

print(len(data))

# ja	ko	fr	pt	th	en	es	ar	vi	zh
langs = ["ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh"]

acc_langs = []
strict_acc_langs = []
think_cons_langs = []
answer_cons_langs = []
cons_langs = []
for lang in langs:
    acc_avg = 0
    strict_acc_avg = 0
    think_cons_avg = 0
    answer_cons_avg = 0
    cons_avg = 0
    for i in range(4):
        benchmark_weighted_acc = round(sum([(2 ** _i) * data[f"{lang}-{level}-{i}"][f"acc"] for _i, level in enumerate(["low", "medium", "high", "top"])]) / 15, 2)
        benchmark_weighted_strict_acc = round(sum([(2 ** _i) * data[f"{lang}-{level}-{i}"][f"strict_acc"] for _i, level in enumerate(["low", "medium", "high", "top"])]) / 15, 2)
        benchmark_thinking_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"thinking_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)
        benchmark_answer_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"answer_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)
        benchmark_lang_cons = round(sum([ data[f"{lang}-{level}-{i}"][f"all_lang_cons"] for level in ["low", "medium", "high", "top"]]) / 4, 2)


        # print(f"{lang}-{i}", benchmark_weighted_acc, benchmark_weighted_strict_acc, benchmark_thinking_lang_cons, benchmark_answer_lang_cons, benchmark_lang_cons)
        acc_avg += benchmark_weighted_acc
        strict_acc_avg += benchmark_weighted_strict_acc
        think_cons_avg += benchmark_thinking_lang_cons
        answer_cons_avg += benchmark_answer_lang_cons
        cons_avg += benchmark_lang_cons
    
    acc_langs.append(str(round(acc_avg/4, 2)))
    strict_acc_langs.append(str(round(strict_acc_avg/4, 2)))
    think_cons_langs.append(str(round(think_cons_avg/4, 2)))
    answer_cons_langs.append(str(round(answer_cons_avg/4, 2)))
    cons_langs.append(str(round(cons_avg/4, 2)))


float_strict_acc_langs = [float(i) for i in strict_acc_langs]
float_acc_langs = [float(i) for i in acc_langs]
float_cons_langs = [float(i) for i in cons_langs]


print(path)
print("Metrics", "ja",	"ko", "fr",	"pt", "th", "en", "es", "ar", "vi", "zh", "ID-avg", "OOD-avg", "ALL-avg")
print("LC&Acc:\t", "\t".join(strict_acc_langs), "\t", round(mean(float_strict_acc_langs[:5]), 2), "\t", round(mean(float_strict_acc_langs[5:]), 2), "\t", round(mean(float_strict_acc_langs), 2))
print("Acc:\t", "\t".join(acc_langs), "\t", round(mean(float_acc_langs[:5]), 2), "\t", round(mean(float_acc_langs[5:]), 2), "\t", round(mean(float_acc_langs), 2))
print("LC:\t", "\t".join(cons_langs), "\t", round(mean(float_cons_langs[:5]), 2), "\t", round(mean(float_cons_langs[5:]), 2), "\t", round(mean(float_cons_langs), 2))

print("-"*100)

float_think_cons_langs = [float(i) for i in think_cons_langs]
float_answer_cons_langs = [float(i) for i in answer_cons_langs]

# print("\t".join(think_cons_langs), "\t", round(mean(float_think_cons_langs[:5]), 2), "\t", round(mean(float_think_cons_langs[5:]), 2), "\t", round(mean(float_think_cons_langs), 2))
# print("\t".join(answer_cons_langs), "\t", round(mean(float_answer_cons_langs[:5]), 2), "\t", round(mean(float_answer_cons_langs[5:]), 2), "\t", round(mean(float_answer_cons_langs), 2))

print()
print()




