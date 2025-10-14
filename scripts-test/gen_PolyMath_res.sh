ROOT_PATH=eval_tools/PolyMath


temp=$1
MODEL_PATH=$2
MODEL_NAME=$3
CUR_LOG=logs-eval/PolyMath-temp_$temp/$MODEL_NAME/logs

if [ ! -d "$CUR_LOG" ]; then
    mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG



level_array=(low medium high top)
for level in ${level_array[@]};
do
    echo ${level}
    CUDA_VISIBLE_DEVICES=0 nohup python $ROOT_PATH/polymath_res_gen.py --lang es --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/es_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python $ROOT_PATH/polymath_res_gen.py --lang fr --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/fr_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 nohup python $ROOT_PATH/polymath_res_gen.py --lang ar --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ar_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 nohup python $ROOT_PATH/polymath_res_gen.py --lang ja --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ja_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 nohup python $ROOT_PATH/polymath_res_gen.py --lang ko --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/ko_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=5 nohup python $ROOT_PATH/polymath_res_gen.py --lang pt --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/pt_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6 nohup python $ROOT_PATH/polymath_res_gen.py --lang th --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/th_$level.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 nohup python $ROOT_PATH/polymath_res_gen.py --lang vi --level ${level} --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/vi_$level.log 2>&1 &
    
    wait
done

wait
echo "en-zh"

CUDA_VISIBLE_DEVICES=0 nohup python $ROOT_PATH/polymath_res_gen.py --lang en --level low --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en_low.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python $ROOT_PATH/polymath_res_gen.py --lang en --level medium --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en_medium.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python $ROOT_PATH/polymath_res_gen.py --lang en --level high --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en_high.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python $ROOT_PATH/polymath_res_gen.py --lang en --level top --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/en_top.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python $ROOT_PATH/polymath_res_gen.py --lang zh --level low --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh_low.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python $ROOT_PATH/polymath_res_gen.py --lang zh --level medium --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh_medium.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python $ROOT_PATH/polymath_res_gen.py --lang zh --level high --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh_high.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python $ROOT_PATH/polymath_res_gen.py --lang zh --level top --temp $temp --model_path $MODEL_PATH --model_name $MODEL_NAME >> $CUR_LOG/zh_top.log 2>&1 &


wait

sleep 30s

