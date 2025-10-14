import json, jsonlines
import os
import argparse
from tqdm import tqdm
import time, random
from vllm import LLM, SamplingParams
from datasets import load_dataset




parser = argparse.ArgumentParser()

parser.add_argument("--lang", type=str, default='en', help="The language of the dataset.")
parser.add_argument("--bench", type=str, default='eval_tools/PolyMath/data-polymath', help="The benchmark dataset.")
# level_list=(low medium high top)
parser.add_argument("--level", type=str, default='low', help="The level of the dataset.")
parser.add_argument("--temp", type=float, default=0.6)

parser.add_argument("--model_path", type=str, default="Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")

parser.add_argument("--tensor_parallel_size", type=int, default=1, help="The size of the tensor parallelism.")


parser.add_argument("--inference_type", type=str, 
                        default="no_constrain")  


parser.add_argument("--save_path", type=str, 
                    default="logs-eval/PolyMath-temp_0.9")


args = parser.parse_args()

temp = args.temp
save_path = args.save_path
model_name = args.model_name
model_path = args.model_path

sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=32768, n=4)
llm = LLM(model=model_path, max_model_len=36000, tensor_parallel_size=args.tensor_parallel_size)
tokenizer = llm.llm_engine.tokenizer.tokenizer


LANGUAGE = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

LANGUAGE = args.lang
level = args.level
print("Testing on languages:", LANGUAGE)

no_constrain_LANG_TO_INSTRUCTIONS = {
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


src = load_dataset(f"{args.bench}/{LANGUAGE}")["train"]

# sample = {
#   "idx": 114,    ### unique sample id
#   "question": "假设在平面上的一个紧集 $C$ 满足以下条件：对每一个方向，都存在一条该方向上的直线 $l$，使得 $l \\cap C$ 的维数至少为 $\\frac{1}{2}$。那么，$C$ 的最小可能维数是多少？",    ### question in corresponding language version
#   "answer": "$\\frac{5}{4}$",    ### ground truth
#   "thinking_pred": "嗯，这个问题看起来有点挑战性，不过让我慢慢想想。题目是说，在平面上有一个紧集C...",    ### Note: Model's thinking content. Note: If it is a non-reasoning model, leave this field blank.
#   "answer_pred": "题目要求在平面上的一个紧集 \\( C \\)，满足对于每一个方向，...",    ### Note: Model's answer content.
# }

all_prompts = []
inputs = []
for item in src:
    if level in item["id"]:
        question = item['question']
        if args.inference_type == "no_constrain":
            formatted_prompt = no_constrain_LANG_TO_INSTRUCTIONS[LANGUAGE].format(question=question)
        chat_template_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_prompt}], 
            tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        all_prompts.append(chat_template_prompt)
        inputs.append(item)

print(len(all_prompts))

outputs = llm.generate(all_prompts, sampling_params)

save_res = []
for output, item in zip(outputs, inputs):
    for i, output_i in enumerate(output.outputs):
        generated_text = output_i.text
        think_pred = None
        answer_pred = None
        
        if "</think>" in generated_text:
            tmp = generated_text.split("</think>")
            
            if len(tmp) == 2:
                item[f"thinking_pred_{i}"] = tmp[0]
                item[f"answer_pred_{i}"] = tmp[1]
            else:
                item[f"thinking_pred_{i}"] = None
                item[f"answer_pred_{i}"] = generated_text
        else:
            item[f"thinking_pred_{i}"] = None
            item[f"answer_pred_{i}"] = generated_text
    save_res.append(item)

print(len(save_res))

cur_save_path = f"{save_path}/{model_name}/{level}"
os.makedirs(cur_save_path, exist_ok=True)

json.dump(save_res, open(f"{cur_save_path}/{LANGUAGE}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)



