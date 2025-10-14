import json
from vllm import LLM, SamplingParams
import os
from utils import *
from math_verify import parse, verify
import argparse
from tqdm import tqdm

os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")

parser.add_argument("--tensor_parallel_size", type=int, default=1, help="The size of the tensor parallelism.")
parser.add_argument("--lang", type=str, default='all', help="The language of the dataset.", nargs='+')
parser.add_argument("--bench", type=str, default='mmath', help="The benchmark dataset.")
parser.add_argument("--temp", type=float, default=0.6)

args = parser.parse_args()

model_path = args.model_path
model_name = args.model_name

sampling_params = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=32768, n=4)
llm = LLM(model=model_path, max_model_len=36000, tensor_parallel_size=args.tensor_parallel_size)
tokenizer = llm.llm_engine.tokenizer.tokenizer

LANGUAGE = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']
if args.lang != 'all':
    LANGUAGE = args.lang
print("Testing on languages:", LANGUAGE)

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

def save_results(mmath, lang):
    os.makedirs(f'logs-eval/MMATH-temp_{args.temp}/{model_name}', exist_ok=True)
    with open(f'logs-eval/MMATH-temp_{args.temp}/{model_name}/{lang}.json', 'w', encoding='utf-8') as f:
        json.dump(mmath[lang], f, ensure_ascii=False, indent=4)

# Step 1: Load all prompts and track (lang, idx) for mapping later
mmath = {}
all_prompts = []
prompt_lang_idx = []  # Keep track of which (lang, index) each prompt belongs to

for lang in LANGUAGE:
    with open(f'eval_tools/MMATH/{args.bench}/{lang}.json', 'r', encoding='utf-8') as f:
        mmath[lang] = json.load(f)
    
    for i, item in enumerate(mmath[lang]):
        question = item['question']
        formatted_prompt = LANG_TO_INSTRUCTIONS[lang].format(question=question)
        chat_template_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_prompt}], 
            tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        
        mmath[lang][i]['final_prompt'] = chat_template_prompt
        all_prompts.append(chat_template_prompt)
        prompt_lang_idx.append((lang, i))

# Step 2: Run vLLM once for all prompts
outputs = llm.generate(all_prompts, sampling_params)

# Step 3: Map outputs back to their language and question
for output, (lang, idx) in tqdm(zip(outputs, prompt_lang_idx)):
    for i, output_i in enumerate(output.outputs):
        generated_text = output_i.text
        mmath[lang][idx][f'prediction_{i}'] = generated_text
        mmath[lang][idx][f'pred_answer_{i}'] = math_postprocess_v2(generated_text)
        
        if mmath[lang][idx][f'pred_answer_{i}'] is None:
            if_correct = False
        else:
            gold = parse(mmath[lang][idx]['answer'])
            pred = parse('$' + mmath[lang][idx][f'pred_answer_{i}'] + '$')
            if_correct = verify(gold, pred)
        
        mmath[lang][idx][f'correct_{i}'] = if_correct

# Step 4: Save results language by language
for lang in LANGUAGE:
    save_results(mmath, lang)



