import json
from vllm import LLM, SamplingParams
import os
# from utils import *
# from math_verify import parse, verify
import argparse
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv
from verl.utils.reward_score.math_verify import compute_score
from transformers import AutoTokenizer
from numpy import mean
from langdetect import detect_langs, DetectorFactory
import re
from datasets import Dataset



def extract_solution(solution_str):
    try:
        res = remove_boxed(last_boxed_only_string(solution_str))
    except:
        # print(solution_str)
        res = solution_str.split("the final answer is \(\boxed{")[-1]  # 有一个特殊情况 “Thus, the final answer is \(\boxed{504”
    if res != solution_str:
        return res
    else:
        return None


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


os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="XueZhang-bjtu/1.5B-cold-start-SFT")

parser.add_argument("--tensor_parallel_size", type=int, default=1, help="The size of the tensor parallelism.")
parser.add_argument("--temp", type=float, default=0.9)
parser.add_argument("--split", type=int, default=6000)
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--data_path", type=str, default="data/M-Thinker-1.5B-RL-Iter1-data-ori/train.parquet")
parser.add_argument("--save_path", type=str, default="data/M-Thinker-1.5B-RL-Iter1-data")
parser.add_argument("--cnt", type=int, default=0)
parser.add_argument("--rollout", type=int, default=1)


args = parser.parse_args()

model_path = args.model_path
data_path = args.data_path
save_path = args.save_path


have_dataset = Dataset.from_parquet(data_path)

try:
    need_gen_ids = json.load(open(f"{save_path}/need_gen_ids-{args.cnt-1}.json"))
except:
    need_gen_ids = []

en_ids_data = {}
for i_data in have_dataset:
    cur_id = i_data["extra_info"]["id"].split("-")[-1]
    if need_gen_ids != [] and cur_id not in need_gen_ids: continue
    if cur_id not in en_ids_data:
        en_ids_data[cur_id] = {}
        en_ids_data[cur_id]["en_question"] = i_data["extra_info"]["en_question"]
        en_ids_data[cur_id]["ground_truth_solution"] = i_data["reward_model"]["ground_truth"]
    else:
        assert en_ids_data[cur_id]["en_question"] == i_data["extra_info"]["en_question"]
        assert en_ids_data[cur_id]["ground_truth_solution"] == i_data["reward_model"]["ground_truth"]

print(len(en_ids_data), list(en_ids_data.keys())[:5])

en_ids = list(en_ids_data.keys())

sampling_params = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=32768, n=args.rollout)
llm = LLM(model=model_path, max_model_len=36000, tensor_parallel_size=args.tensor_parallel_size)
tokenizer = llm.llm_engine.tokenizer.tokenizer


lang = "en"
print("Testing on languages:", lang)

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

save_path = save_path + f"/cnt-{args.cnt}"  
os.makedirs(save_path, exist_ok=True)

if args.split is not None:
    assert args.part is not None
    cnt = int(len(en_ids_data) / args.split) + 1
    st = args.part * cnt
    ed = min((args.part + 1) * cnt, len(en_ids_data))
    print(st, ed, ed-st)
    save_path = f"{save_path}/{lang}-predict-p{args.part}.json"
else:
    st = 0
    ed = len(en_ids_data)
    save_path = f"{save_path}/{lang}-predict.json"

cur_part_ids = en_ids[st:ed]
print(len(cur_part_ids))
cur_all_prompts = [] 
cur_part_data = {}
for i_id in cur_part_ids:
    question = en_ids_data[i_id]["en_question"]
    formatted_prompt = LANG_TO_INSTRUCTIONS[lang].format(question=question)
    chat_template_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_prompt}], 
        tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    cur_part_data[i_id] = en_ids_data[i_id]
    cur_part_data[i_id]['final_prompt'] = chat_template_prompt
    cur_all_prompts.append(chat_template_prompt)

# Step 2: Run vLLM once for all prompts
outputs = llm.generate(cur_all_prompts, sampling_params)

for output, idx in tqdm(zip(outputs, cur_part_ids)):
    for i, output_i in enumerate(output.outputs):
        generated_text = output_i.text
        cur_part_data[idx][f'prediction_{i}'] = "<think>\n" + generated_text
        
json.dump(cur_part_data, open(save_path.replace(".json", "-raw_outputs.json"), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


# Step 3: Map outputs back to their language and question
for output, idx in tqdm(zip(outputs, cur_part_ids)):
    gold_solution = cur_part_data[idx]["ground_truth_solution"]
    acc_num = 0
    cons_acc_num = 0

    for i, output_i in enumerate(output.outputs):
        generated_text = output_i.text
        cur_part_data[idx][f'prediction_{i}'] = "<think>\n" + generated_text
        cur_part_data[idx][f'pred_answer_{i}'] = extract_solution(generated_text)
        
        if cur_part_data[idx][f'pred_answer_{i}'] is None:
            if_correct = False
        elif gold_solution is not None and compute_score(generated_text, gold_solution) == 1.0:
            if_correct = True
            acc_num += 1
        else:
            if_correct = False

        if_cons_correct = False

        think_pred = None
        answer_pred = None
        if if_correct:
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

            if whether_cons_think and whether_cons_answer:
                cons_acc_num += 1
                if_cons_correct = True
        
        cur_part_data[idx][f'correct_{i}'] = if_correct
        cur_part_data[idx][f'cons_correct_{i}'] = if_cons_correct

    cur_part_data[idx][f'acc_num'] = acc_num
    cur_part_data[idx][f'cons_acc_num'] = cons_acc_num

json.dump(cur_part_data, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)




