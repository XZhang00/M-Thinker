export CODE_DIR=your_path/M-Thinker/LLaMA-Factory


ROOT_DIR=your_path/M-Thinker
PRETRAIN_MODEL=your_path/Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

DATA=fr,ja,ko,pt,th
LR=5e-7

LENGTH=16384
EPOCH=1.0
BSZ=256

let Gradient_accumulation_steps=$BSZ/8

EXP_TAG=bsz_${BSZ}-lr_${LR}-epoch${EPOCH}-max_seqlen$LENGTH
OUTPUT_DIR=$ROOT_DIR/checkpoints/DeepSeek-R1-Distill-Qwen-7B-cold_SFT/${EXP_TAG}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
chmod 777 -R $OUTPUT_DIR



export DISABLE_VERSION_CHECK=1
export WANDB_DISABLED=true
nohup your_path/anaconda3/envs/lf_train/bin/deepspeed \
    --num_gpus 8 --master_port=9902 $CODE_DIR/src/train.py \
    --deepspeed $CODE_DIR/examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --dataset_dir your_path/M-Thinker/data/M-Thinker-SFT-data \
    --model_name_or_path $PRETRAIN_MODEL \
    --dataset $DATA \
    --template deepseekr1 \
    --finetuning_type full \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --preprocessing_num_workers 32 \
    --output_dir $OUTPUT_DIR \
    --cutoff_len $LENGTH \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $Gradient_accumulation_steps \
    --packing false \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --bf16 \
    --plot_loss >> $OUTPUT_DIR/train.log 2>&1 &



wait 
sleep 30s
