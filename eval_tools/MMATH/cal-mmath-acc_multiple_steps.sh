model_name=$1

CUR_LOG=logs-eval/mmath-eval-paper-temp_0.9.log

RES_PATH=logs-eval/MMATH-temp_0.9/$model_name


steps=(320 325 330 335 340 345 350 355 360 365 370 375 380 385 390 395 400 405 410 415 420 425 430 435)


for step in ${steps[@]};
do  
    cur_res_path=$RES_PATH/step${step}-temp_0.9
    nohup python -u eval_tools/MMATH/cal-MMATH-acc.py \
        --res_path $cur_res_path >> $CUR_LOG 2>&1 &
    sleep 3s
done