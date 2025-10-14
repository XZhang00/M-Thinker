import json, jsonlines, os
import random
from datasets import Dataset


from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    try:
        res = remove_boxed(last_boxed_only_string(solution_str))
    except:
        # print(solution_str)
        res = solution_str.split("the final answer is \(\boxed{")[-1]  # 有一个特殊情况 “Thus, the final answer is \(\boxed{504”
    return res


LANG_TO_INSTRUCTIONS = {
    'en': "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    'es': "{question}\nPor favor, razona paso a paso y pon tu respuesta final dentro de \\boxed{{}}.",
    'fr': "{question}\nVeuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{{}}.",
    'zh': "{question}\n请逐步推理，并将您的最终答案放在 \\boxed{{}} 中。",
    'ja': "{question}\nステップバイステップで推論し、最終的な答えを \\boxed{{}} の中に入れてください。",
    'th': "{question}\nกรุณาเหตุผลขั้นตอนต่อขั้นตอนและใส่คำตอบสุดท้ายของคุณใน \\boxed{{}}.",
    'ko': "{question}\n단계별로 추론하고 최종 답변을 \\boxed{{}} 안에 넣어주세요.",
    'pt': "{question}\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{{}}.",
    'vi': "{question}\nVui lòng lý giải từng bước và đặt câu trả lời cuối cùng của bạn trong \\boxed{{}}.",
    'ar': "{question}\nيرجى المنطق خطوة بخطوة، ووضع إجابتك النهائية داخل \\boxed{{}}."
}



final_data = []

split = 64

ft_data_number = 3000

for lang in ["fr", "pt", "th", "ja", "ko"]:
    predict_path = "data/predict-other_data-cold-start-sft/1.5B-cold-start-SFT"
    cnt = 0
    acc = 0
    acc_all = 0
    cons_acc = 0
    cons_acc_all = 0
    lang_filter_data = []
    for part in range(split):
        try:
            cur_predict = json.load(open(f"{predict_path}/{lang}/{lang}-predict-p{part}.json"))
            for item in cur_predict:
                cnt += 1
                if item["acc_num"] > 0: acc += 1
                if item["acc_num"] == 8: acc_all += 1
                if item["cons_acc_num"] == 8: cons_acc_all += 1
                if item["cons_acc_num"] > 0 and item["cons_acc_num"] < 8: 
                    cons_acc += 1
                    question = item["translated_question"]
                    question = LANG_TO_INSTRUCTIONS[lang].format(question=question)
                    lang_filter_data.append({
                            "data_source": lang,
                            "prompt": [{"role": "user", "content": question}],
                            "ability": "math",
                            "reward_model": {"style": "rule", "ground_truth": item["ground_truth_solution"]},
                            "extra_info": {"en_question": item["en_question"], "id": item["id"], "en_answer": item["en_answer"]},
                        })
                    
            print(lang, part, len(lang_filter_data))
        except:
            print(lang, part, "error!")

                
    print(lang, cnt, acc, f"[{round(acc/cnt*100, 2)}%]", cons_acc, f"[{round(cons_acc/cnt*100, 2)}%]", acc_all, cons_acc_all)
    print(lang, len(lang_filter_data))


    # ja/ko, if the data volume does not meet the requirements, use the following data to make up to 3000 records;
    if len(lang_filter_data) < ft_data_number:
        predict_path = "data/predict-train_data-cold-start-sft/1.5B-cold-start-SFT"
        for part in range(8):
            try:
                cur_predict = json.load(open(f"{predict_path}/{lang}-predict-p{part}.json"))
                for item in cur_predict:
                    cnt += 1
                    if item["acc_num"] > 0: acc += 1
                    if item["cons_acc_num"] > 0 and item["cons_acc_num"] < 8: 
                        cons_acc += 1

                        question = item["translated_question"]
                        question = LANG_TO_INSTRUCTIONS[lang].format(question=question)
                        if len(lang_filter_data) < ft_data_number:
                            lang_filter_data.append({
                                    "data_source": lang,
                                    "prompt": [{"role": "user", "content": question}],
                                    "ability": "math",
                                    "reward_model": {"style": "rule", "ground_truth": item["ground_truth_solution"]},
                                    "extra_info": {"en_question": item["en_question"], "id": item["id"], "en_answer": item["en_answer"]},
                                })
            except:
                print(lang, part, "error!")
            
        print("add-train-part", lang, len(lang_filter_data))

        if len(lang_filter_data) < ft_data_number:
            predict_path = "data/predict-other_data-cold-start-sft/1.5B-cold-start-SFT"
            for part in range(split):
                try:
                    cur_predict = json.load(open(f"{predict_path}/{lang}/{lang}-predict-p{part}.json"))
                    for item in cur_predict:
                        cnt += 1
                        if item["acc_num"] > 0: acc += 1
                        if item["acc_num"] > 0 and item["cons_acc_num"] == 0: 
                            question = item["translated_question"]
                            question = LANG_TO_INSTRUCTIONS[lang].format(question=question)
                            if len(lang_filter_data) < ft_data_number:
                                lang_filter_data.append({
                                        "data_source": lang,
                                        "prompt": [{"role": "user", "content": question}],
                                        "ability": "math",
                                        "reward_model": {"style": "rule", "ground_truth": item["ground_truth_solution"]},
                                        "extra_info": {"en_question": item["en_question"], "id": item["id"], "en_answer": item["en_answer"]},
                                    })
                except:
                    print(lang, part, "error!")
            print("add-other-acc-no_cons-part", lang, len(lang_filter_data))

        if len(lang_filter_data) < ft_data_number:
            predict_path = "data/predict-other_data-cold-start-sft/1.5B-cold-start-SFT"
            for part in range(split):
                try:
                    cur_predict = json.load(open(f"{predict_path}/{lang}/{lang}-predict-p{part}.json"))
                    for item in cur_predict:
                        cnt += 1
                        if item["acc_num"] > 0: acc += 1
                        if item["acc_num"] == 0: 
                            question = item["translated_question"]
                            question = LANG_TO_INSTRUCTIONS[lang].format(question=question)
                            if len(lang_filter_data) < ft_data_number:
                                lang_filter_data.append({
                                        "data_source": lang,
                                        "prompt": [{"role": "user", "content": question}],
                                        "ability": "math",
                                        "reward_model": {"style": "rule", "ground_truth": item["ground_truth_solution"]},
                                        "extra_info": {"en_question": item["en_question"], "id": item["id"], "en_answer": item["en_answer"]},
                                    })
                except:
                    print(lang, part, "error!")
            print("add-other-no-acc-no_cons-part", lang, len(lang_filter_data))
  

    random.seed = 42
    random.shuffle(lang_filter_data)
    sample_lang_filter_data = lang_filter_data[:ft_data_number]
    print(len(sample_lang_filter_data))


    final_data += sample_lang_filter_data

print(len(final_data))


random.seed = 3072
random.shuffle(final_data)

train_dataset = Dataset.from_list(final_data)
test_dataset = Dataset.from_list(final_data[:64])

save_dir = "data/M-Thinker-1.5B-RL-Iter1-ori"
train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))