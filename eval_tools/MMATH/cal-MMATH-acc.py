import json
from langdetect import detect_langs, DetectorFactory
import re, argparse
from transformers import AutoTokenizer
from numpy import mean


langs = ["ja", "ko", "fr", "pt", "th", "en", "es", "ar", "vi", "zh"]


def whether_cons(pred, language):
    re_pred = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', pred, flags=re.DOTALL)
    re_pred = re.sub(r'\$[^$]*\$', '', re_pred)
    if len(re_pred) <= 15:
        lang_cons_binary = True
    else:
        lang_cons_binary = False
        try:
            DetectorFactory.seed = 42
            lang_prob = detect_langs(re_pred)
            detect_lang = "zh-cn" if language == "zh" else language
            pred_lang = [lang.lang for lang in lang_prob]
            lang_cons_binary = True if (len(lang_prob) == 1 and detect_lang in pred_lang) else False
            # if len(lang_prob) != 1:
            #     print(lang_prob)
        except:
            pass
    
    return lang_cons_binary



parser = argparse.ArgumentParser()
parser.add_argument("--res_path", type=str, default="")

args = parser.parse_args()
    
path = args.res_path

acc_langs = []
strict_acc_langs = []
think_cons_langs = []
answer_cons_langs = []
cons_langs = []
response_length_langs = []

tokenizer = AutoTokenizer.from_pretrained("Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


for lang in langs:
    data = json.load(open(f"{path}/{lang}.json"))

    sources = ["AIME2024", "AIME2025", "CNMO", "MATH500"]
    dif_cnt_acc = 0
    dif_cnt_strict_acc = 0
    dif_cnt_think_cons = 0
    dif_cnt_answer_cons = 0
    dif_cnt_cons = 0

    response_length_list = []

    for i_cnt in range(4):
        acc_dict = {}
        strict_acc_dict = {}
        think_cons_dict = {}
        answer_cons_dict = {}
        cons_dict = {}

        for i_source in sources:
            acc_dict[i_source] = {"num": 0, "acc": 0}
            strict_acc_dict[i_source] = {"num": 0, "acc&cons": 0}
            think_cons_dict[i_source] = {"num": 0, "think_cons": 0}
            answer_cons_dict[i_source] = {"num": 0, "answer_cons": 0}
            cons_dict[i_source] = {"num": 0, "cons": 0}


        for item in data:
            assert item["data_source"] in sources
            acc_dict[item["data_source"]]["num"] += 1
            strict_acc_dict[item["data_source"]]["num"] += 1
            think_cons_dict[item["data_source"]]["num"] += 1
            answer_cons_dict[item["data_source"]]["num"] += 1
            cons_dict[item["data_source"]]["num"] += 1

            if item[f"correct_{i_cnt}"]: 
                acc_dict[item["data_source"]]["acc"] += 1
            
            think_pred = None
            answer_pred = None
            generated_text = item[f"prediction_{i_cnt}"]
            response_length_list.append(len(tokenizer(generated_text)['input_ids']))
            if "</think>" in generated_text:
                tmp = generated_text.split("</think>")
                if len(tmp) == 2:
                    think_pred = tmp[0]
                    answer_pred = tmp[1]

            if think_pred is not None:
                whether_cons_think = whether_cons(think_pred, lang)
            else:
                whether_cons_think = False

            if answer_pred is not None:
                whether_cons_answer = whether_cons(answer_pred, lang)
            else:
                whether_cons_answer = False

            if whether_cons_think:
                think_cons_dict[item["data_source"]]["think_cons"] += 1

            if whether_cons_answer:
                answer_cons_dict[item["data_source"]]["answer_cons"] += 1
            
            if whether_cons_think and whether_cons_answer:
                cons_dict[item["data_source"]]["cons"] += 1

            if whether_cons_think and whether_cons_answer and item[f"correct_{i_cnt}"]: 
                strict_acc_dict[item["data_source"]]["acc&cons"] += 1

        dif_source_acc = 0
        dif_source_strict_acc = 0
        dif_source_think_cons = 0
        dif_source_answer_cons = 0
        dif_source_cons = 0

        for i_source in sources:
            dif_source_acc += acc_dict[i_source]["acc"] / acc_dict[i_source]["num"] * 100
            dif_source_strict_acc += strict_acc_dict[i_source]["acc&cons"] / strict_acc_dict[i_source]["num"] * 100
            dif_source_think_cons += think_cons_dict[i_source]["think_cons"] / think_cons_dict[i_source]["num"] * 100
            dif_source_answer_cons += answer_cons_dict[i_source]["answer_cons"] / answer_cons_dict[i_source]["num"] * 100
            dif_source_cons += cons_dict[i_source]["cons"] / cons_dict[i_source]["num"] * 100

        # print(lang, acc_dict, round(dif_source_acc/4, 4))
        # print(lang, strict_acc_dict, round(dif_source_strict_acc/4, 4))
        # print(lang, think_cons_dict, round(dif_source_think_cons/4, 4))
        # print(lang, answer_cons_dict, round(dif_source_answer_cons/4, 4))
        # print(lang, cons_dict, round(dif_source_cons/4, 4))

        dif_cnt_acc += dif_source_acc/4
        dif_cnt_strict_acc += dif_source_strict_acc/4
        dif_cnt_think_cons += dif_source_think_cons/4
        dif_cnt_answer_cons += dif_source_answer_cons/4
        dif_cnt_cons += dif_source_cons/4

    acc_langs.append(str(round(dif_cnt_acc/4, 4)))
    strict_acc_langs.append(str(round(dif_cnt_strict_acc/4, 4)))
    think_cons_langs.append(str(round(dif_cnt_think_cons/4, 4)))
    answer_cons_langs.append(str(round(dif_cnt_answer_cons/4, 4)))
    cons_langs.append(str(round(dif_cnt_cons/4, 4)))

    # print(len(response_length_list), mean(response_length_list))
    response_length_langs.append(str(round(mean(response_length_list), 4)))


print(path)

float_strict_acc_langs = [float(i) for i in strict_acc_langs]
float_acc_langs = [float(i) for i in acc_langs]
float_cons_langs = [float(i) for i in cons_langs]
float_response_length_langs = [float(i) for i in response_length_langs]

print("Metrics", "ja",	"ko", "fr",	"pt", "th", "en", "es", "ar", "vi", "zh", "ID-avg", "OOD-avg", "ALL-avg")
print("LC&Acc:\t", "\t".join(strict_acc_langs), "\t", round(mean(float_strict_acc_langs[:5]), 2), "\t", round(mean(float_strict_acc_langs[5:]), 2), "\t", round(mean(float_strict_acc_langs), 2))
print("Acc:\t", "\t".join(acc_langs), "\t", round(mean(float_acc_langs[:5]), 2), "\t", round(mean(float_acc_langs[5:]), 2), "\t", round(mean(float_acc_langs), 2))
print("LC:\t", "\t".join(cons_langs), "\t", round(mean(float_cons_langs[:5]), 2), "\t", round(mean(float_cons_langs[5:]), 2), "\t", round(mean(float_cons_langs), 2))
# print("Response-length\t", "\t".join(response_length_langs), "\t", round(mean(float_response_length_langs[:5]), 2), "\t", round(mean(float_response_length_langs[5:]), 2), "\t", round(mean(float_response_length_langs), 2))

print("-"*100)

float_think_cons_langs = [float(i) for i in think_cons_langs]
float_answer_cons_langs = [float(i) for i in answer_cons_langs]

# print("\t".join(think_cons_langs), "\t", round(mean(float_think_cons_langs[:5]), 2), "\t", round(mean(float_think_cons_langs[5:]), 2), "\t", round(mean(float_think_cons_langs), 2))
# print("\t".join(answer_cons_langs), "\t", round(mean(float_answer_cons_langs[:5]), 2), "\t", round(mean(float_answer_cons_langs[5:]), 2), "\t", round(mean(float_answer_cons_langs), 2))

print()
print()

