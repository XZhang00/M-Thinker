ROOT_PATH=eval_tools/MMATH

temp=$1
MODEL_PATH=$2
MODEL_NAME=$3


# python verl/scripts/model_merger.py merge \
#     --backend fsdp \
#     --local_dir $MODEL_PATH/actor \
#     --target_dir $MODEL_PATH


CUR_LOG=logs-eval/MMATH-temp_$temp/$MODEL_NAME/logs


if [ ! -d "$CUR_LOG" ]; then
    mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG


CUDA_VISIBLE_DEVICES=0 nohup python $ROOT_PATH/mmath_eval.py --lang en ko --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en-ko.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python $ROOT_PATH/mmath_eval.py --lang zh ja --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh-ja.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python $ROOT_PATH/mmath_eval.py --lang ar --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ar.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python $ROOT_PATH/mmath_eval.py --lang fr --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/fr.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python $ROOT_PATH/mmath_eval.py --lang es --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/es.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python $ROOT_PATH/mmath_eval.py --lang pt --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/pt.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python $ROOT_PATH/mmath_eval.py --lang th --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/th.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python $ROOT_PATH/mmath_eval.py --lang vi --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/vi.log 2>&1 &


wait

sleep 30s

