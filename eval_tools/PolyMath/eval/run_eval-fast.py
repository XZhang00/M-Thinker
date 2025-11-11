import os
import re
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# from scripts import math_equal
from math_verify import parse, verify
from langdetect import detect_langs, DetectorFactory

language_list = ["en", "zh", "ar", "bn", "de", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", ]
language_list = ["ko", "ja", "pt", "th", "en", "zh", "ar", "es", "fr", "vi"]
level_list = ["low", "medium", "high", "top"]


def extract_boxed_content(text):
    pattern = re.compile(r'boxed{')
    text = text.replace(' ', '')

    matches = pattern.finditer(text)
    results = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[start_pos:i-1])
    return results


def evaluation(args):
    model = args.model
    language = args.language
    level = args.level
    cnt = args.cnt
    print(model, language, level, cnt)

    output_file = f"logs-eval/PolyMath-temp_0.9/{model}/{level}/{language}.json"
    try:
        data = json.load(open(output_file, 'r', encoding='utf-8'))
        
    except FileNotFoundError:
        ori_data = {}
        for seed in [10, 20, 30, 40]:
            ori_data[seed] = []
            output_file = output_file.replace(f"{language}.json", f"{language}-seed_{seed}.jsonl")
            with open(output_file, 'r', encoding='utf-8') as file:
                for line in file:
                    item = json.loads(line)
                    ori_data[seed].append(item)
            assert len(ori_data[seed]) == 125
        data = []
        for i in range(125):
            data.append({
                "id": ori_data[10][i]["id"],
                "question": ori_data[10][i]["question"],
                "answer": ori_data[10][i]["answer"],
                "thinking_pred_0": ori_data[10][i]["thinking_pred"],
                "answer_pred_0": ori_data[10][i]["answer_pred"],
                "thinking_pred_1": ori_data[20][i]["thinking_pred"],
                "answer_pred_1": ori_data[20][i]["answer_pred"],
                "thinking_pred_2": ori_data[30][i]["thinking_pred"],
                "answer_pred_2": ori_data[30][i]["answer_pred"],
                "thinking_pred_3": ori_data[40][i]["thinking_pred"],
                "answer_pred_3": ori_data[40][i]["answer_pred"],
            })

    if len(data) < 125:
        print(f"Warning! Test data is incomplete, current data size: {len(data)}")
    elif len(data) > 125:
        print(f"Warning! Test data is redundant, current data size: {len(data)}")
    else:
        pass


    acc, strict_acc, thinking_lang_cons, answer_lang_cons, all_lang_cons = 0, 0, 0, 0, 0
    for i in range(len(data)):
        # idx = data[i]["idx"]
        # question = data[i]["question"]
        answer = data[i]["answer"]
        thinking_pred = data[i][f"thinking_pred_{cnt}"]
        answer_pred = data[i][f"answer_pred_{cnt}"]

        ### answer extraction & correctness judgement
        try:
            extracted_pred = extract_boxed_content(answer_pred)
            extracted_pred = extracted_pred[0] if len(extracted_pred) > 0 else None
            # acc_binary = math_equal(extracted_pred, answer)
            if extracted_pred is not None:
                gold = parse('$' + answer + '$')
                pred = parse('$' + extracted_pred + '$')
                acc_binary = verify(gold, pred)
            else:
                acc_binary = False
        except:
            acc_binary = False
        acc += 1 if acc_binary else 0
        
        
        ### language consistency judgement
        if thinking_pred is None: 
            thinking_lang_cons_binary = False
        else:
            re_thinking_pred = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', thinking_pred, flags=re.DOTALL)
            re_thinking_pred = re.sub(r'\$[^$]*\$', '', re_thinking_pred)
            if len(re_thinking_pred) <= 15:
                thinking_lang_cons_binary = True
            else:
                thinking_lang_cons_binary = False
                try:
                    DetectorFactory.seed = 42
                    lang_prob = detect_langs(re_thinking_pred)
                    detect_lang = "zh-cn" if language == "zh" else language
                    thinking_lang = [lang.lang for lang in lang_prob]
                    thinking_lang_cons_binary = True if (len(lang_prob) == 1 and detect_lang in thinking_lang) else False
                except:
                    pass
        if answer_pred is None:
            answer_lang_cons_binary = False
        else:
            re_answer_pred = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', answer_pred, flags=re.DOTALL)
            re_answer_pred = re.sub(r'\$[^$]*\$', '', re_answer_pred)
            if len(re_answer_pred) <= 15:
                answer_lang_cons_binary = True
            else:
                answer_lang_cons_binary = False
                try:
                    DetectorFactory.seed = 42
                    lang_prob = detect_langs(re_answer_pred)
                    detect_lang = "zh-cn" if language == "zh" else language
                    answer_lang = [lang.lang for lang in lang_prob]
                    answer_lang_cons_binary = True if (len(lang_prob) == 1 and detect_lang in answer_lang) else False
                except:
                    pass
        
        thinking_lang_cons += 1 if thinking_lang_cons_binary else 0
        answer_lang_cons += 1 if answer_lang_cons_binary else 0
        all_lang_cons += 1 if thinking_lang_cons_binary and answer_lang_cons_binary else 0

        if thinking_lang_cons_binary and answer_lang_cons_binary and acc_binary:
            strict_acc += 1
    
    acc = round(acc / len(data) * 100, 2)
    strict_acc = round(strict_acc / len(data) * 100, 2)
    thinking_lang_cons = round(thinking_lang_cons / len(data) * 100, 2)
    answer_lang_cons = round(answer_lang_cons / len(data) * 100, 2)
    all_lang_cons = round(all_lang_cons / len(data) * 100, 2)

    print(f"Test Data Size: {len(data)}; {language}-{level}-{cnt}\n"
        f"Strict Accuracy (%) = {strict_acc}\n"
        f"Accuracy (%) = {acc}\n"
        f"Language Consistency (thinking) (%) = {thinking_lang_cons}\n"
        f"Language Consistency (answer) (%) = {answer_lang_cons}\n"
        f"ALl language consistency (%) = {all_lang_cons}")
    print("*"*30)

    res_dict = {}
    res_dict[f"{language}-{level}-{cnt}"] = {
        "strict_acc": strict_acc,
        "acc": acc,
        "thinking_lang_cons": thinking_lang_cons,
        "answer_lang_cons": answer_lang_cons,
        "all_lang_cons": all_lang_cons
    }

    ### save results
    score_file = os.path.join(f"logs-eval/PolyMath-temp_0.9/{model}", "score-eval.jsonl")
    save_f_jsonlines = open(score_file, 'a+', encoding="utf-8")
    save_f_jsonlines.write(json.dumps(res_dict, ensure_ascii=False) + '\n')
    save_f_jsonlines.flush()




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--level', type=str, required=True)
    parser.add_argument('--cnt', type=str, required=True)


    args = parser.parse_args()
    evaluation(args)
