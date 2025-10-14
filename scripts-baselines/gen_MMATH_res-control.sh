
ROOT_PATH=eval_tools/MMATH

# control_type=PROMPT
# control_type=DIT
# control_type=QRT
control_type=$1

temp=0.6
MODEL_PATH=Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B-control_$control_type-temp_$temp

# MODEL_PATH=Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B-control_$control_type-temp_$temp



CUR_LOG=logs-eval/MMATH-control/$MODEL_NAME/logs


if [ ! -d "$CUR_LOG" ]; then
    mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG


CUDA_VISIBLE_DEVICES=0 nohup python $ROOT_PATH/mmath_eval-control.py --lang en es --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en-es.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python $ROOT_PATH/mmath_eval-control.py --lang zh fr --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh-fr.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python $ROOT_PATH/mmath_eval-control.py --lang ar --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ar.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python $ROOT_PATH/mmath_eval-control.py --lang ja --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ja.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python $ROOT_PATH/mmath_eval-control.py --lang ko --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ko.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python $ROOT_PATH/mmath_eval-control.py --lang pt --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/pt.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python $ROOT_PATH/mmath_eval-control.py --lang th --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/th.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python $ROOT_PATH/mmath_eval-control.py --lang vi --temp $temp --control_type $control_type --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/vi.log 2>&1 &


wait

sleep 30s

