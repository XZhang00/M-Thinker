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
parser.add_argument("--control_type", type=str, default="PROMPT")

args = parser.parse_args()

model_path = args.model_path
model_name = args.model_name
control_type = args.control_type

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

Prompt_Control_INSTRUCTIONS = {
    'en': "Use English to think and answer.",
    'es': "Usa español para pensar y responder.",
    'fr': "Utilisez le français pour penser et répondre.",
    'zh': "使用中文进行思考和回答。",
    'ja': "日本語を使って考え、回答してください。",
    'th': "ใช้ภาษาไทยในการคิดและตอบคำถาม.",
    'ko': "한국어로 생각하고 답변하세요.",
    'pt': "Use português para pensar e responder.",
    'vi': "Sử dụng tiếng Việt để suy nghĩ và trả lời.",
    'ar': "استخدم العربية للتفكير والإجابة.",
}

DIT_INSTRUCTIONS = {
    'en': "Alright, Okay",
    'es': "Buneo",
    'fr': "Bon",
    'zh': "嗯, 好",
    'ja': "まず",
    'th': "โอเค",
    'ko': "좋아",
    'pt': "Ok, Bem",
    'vi': "Được rồi, Đầu tiên",
    'ar': " حسنًا "
}


QRT_INSTRUCTIONS = {
    'en': "OK, so the problem is {question} Let me think in English. First",
    'es': "Bien, el problema es {question} Déjame pensar en español. Primero",
    'fr': "D\'accord, donc le problème est {question} Laissez-moi réfléchir en français. D\'abord",
    'zh': "好的，问题是{question} 让我用中文思考一下。首先",
    'ja': "わかりました。問題は{question}です。日本語で考えさせてください。まず",
    'th': "ตกลง ดงันนัÊ ปัญหาคือ{question} ใหฉ้นั คิดเป็นภาษาไทยก่อนอÉืน",
    'ko': "좋습니다. 문제는 {question}입니다 한국어로 생각해 보겠습니다. 먼저",
    'pt': "Ok, então o problema é {question} Deixe-me pensar em português. Primeiro",
    'vi': "Được rồi, vấn đề là {question} Hãy để tôi nghĩ bằng tiếng Việt. Đầu tiên",
    'ar': " دعني أفكر باللغة العربیة. أولاً‘,  {question}حسنًا، المشكلة ھي"
}

def save_results(mmath, lang):
    os.makedirs(f'logs-eval/MMATH-control/{model_name}', exist_ok=True)
    with open(f'logs-eval/MMATH-control/{model_name}/{lang}.json', 'w+', encoding='utf-8') as f:
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

        if control_type == "PROMPT":
            formatted_prompt = formatted_prompt + "\n" + Prompt_Control_INSTRUCTIONS[lang]

        chat_template_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_prompt}], 
            tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        if control_type == "DIT":
            chat_template_prompt = chat_template_prompt + DIT_INSTRUCTIONS[lang]
        elif control_type == "QRT":
            chat_template_prompt = chat_template_prompt + QRT_INSTRUCTIONS[lang].format(question=question)
        
        mmath[lang][i]['final_prompt'] = chat_template_prompt
        all_prompts.append(chat_template_prompt)
        prompt_lang_idx.append((lang, i))

print(all_prompts[:5])
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



