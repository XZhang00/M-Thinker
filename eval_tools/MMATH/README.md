# MMATH

This is the official repository for the paper **"MMATH: A Multilingual Benchmark for Mathematical Reasoning"**.

## 📖 Introduction

**MMATH** is a new benchmark specifically designed for multilingual complex reasoning. It comprises 374 carefully selected math problems from high-quality sources, including AIME, CNMO, and MATH-500, and covers ten typologically and geographically diverse languages. Each problem is translated and validated through a rigorous pipeline that combines frontier LLMs with human verification, ensuring semantic consistency.

## 🛠️ Code and Data Usage

MMATH has the following resources. The `mmath` folder contains the benchmark data. The `train` folder contains the training data at Section 3. `mmath_eval.py` is the main program to evaluate the accuracy results, while `calculate_lcr.py` calculates the value of LCR as defined in our paper.

```
│  calculate_lcr.py
│  mmath_eval.py
│  utils.py
│
├─mmath
│      ar.json
│      en.json
│      es.json
│      fr.json
│      ja.json
│      ko.json
│      pt.json
│      th.json
│      vi.json
│      zh.json
│
└─train
        enSFT-3k.json
        enThink-3k.json
        nativeThink-3k.json
```

The example in mmath/xx.json has the following format:

```json
{
        "question": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.", # The question 
        "answer": "204",			# The answer
        "data_source": "AIME2024",  # The data source, which might be AIME2024/AIME2025/CNMO/MATH500
        "data_source_id": 0,        # The index in original data
        "lang": "en",				# Language type
        "gid": 0					# The global id in our benchmark MMATH
},
```

The example in train/yy.json has the following format:

```json
{
        "index":0,
        "answer":"364",
        "conversations":[
            {
                "from":"user",
                "value":"For a positive integer \\( n \\), consider the function\n\n\\[ \nf(n)=\\frac{4n+\\sqrt{4n^2-1}}{\\sqrt{2n+1}+\\sqrt{2n-1}} \n\\]\n\nCalculate the value of\n\n\\[ \nf(1)+f(2)+f(3)+\\cdots+f(40) \n\\]"
            },
            {
                "from":"assistant",
                "value":"<think>\nOkay, let's see. I need to find the sum of f(1) + f(2) + ...Thus, the sum is:\n\n\\[\n\\frac{1}{2} (729 - 1) = \\frac{728}{2} = 364.\n\\]\n\nThe final answer is:\n\n\\[\n\\boxed{364}\n\\]"
            }
        ]
},
```

## 🧪 Experiment Setup

### Environment Setups

To accelerate the process of environment setup, we use `uv` to manage the packages. And our training code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), you can install it based on your requirements (e.g, with -e option).

```bash
conda create -n mmath python=3.10
conda activate mmath
pip install uv
uv pip install -r requirements.txt
```

### Evaluation Commands

#### Accuracy Results

To calculate accuracy results on our benchmark, you can run with:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python mmath_eval.py --model_name_or_path DeepSeek-R1-Distill-Qwen-32B --tensor_parallel_size 2
```

This will generate a `results` directory with a subdirectory with the model name (e.g, DeepSeek-R1-Distill-Qwen-32B). Inside the directory are the results belonging to different languages, such as `en.json`.

#### LCR Results

As for calculating LCR, please download the `lid.176.bin` with:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

After that, please set the `model_list_full` variable in `calculate_lcr.py`. Then, you can run the command using `python calculate_lcr.py`.

```python
model_list_full = [
    "DeepSeek-R1-Distill-Qwen-32B",
    # ...
]
```

This will rewrite some keys in `results/model_name/xx.json` and output a LaTeX table about the whole results.

## Training Setups

As mentioned before, our training code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Here we provide the hyperparameters used in our paper. 

```yaml
### model
model_name_or_path: Qwen2.5-32B-Instruct
trust_remote_code: true

### method
stage: sft
template: qwen
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
packing: false

### dataset
dataset: en-Think
cutoff_len: 32768
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: Qwen2.5-32B-Instruct-en-Think
logging_steps: 10
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true
save_only_model: true
save_total_limit: 10

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1 
save_total_limit: 10 
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
enable_liger_kernel: true
```

## 📄 Citation

```
@article{luo2025mmath,
  title={MMATH: A Multilingual Benchmark for Mathematical Reasoning},
  author={Luo, Wenyang and Zhao, Wayne Xin and Sha, Jing and Wang, Shijin and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2505.19126},
  year={2025}
}
```
