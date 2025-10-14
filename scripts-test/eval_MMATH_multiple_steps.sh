temp=$1

MODEL_PATH=$2
MODEL_NAME=$3
# from 320 ~ 435 (24个模型)

# 320 325 330 335 340 345 350 355 
# 360 365 370 375 380 385 390 395 
# 400 405 410 415 420 425 430 435


if [ $RANK -eq 0 ]; then
    steps=(320 325 330)
    for step in ${steps[@]};
    do  
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 1 ]; then
    steps=(335 340 345)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 2 ]; then
    steps=(350 355 360)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 3 ]; then
    steps=(365 370 375)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 4 ]; then
    steps=(380 385 390)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 5 ]; then
    steps=(395 400 405)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 6 ]; then
    steps=(410 415 420)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

elif [ $RANK -eq 7 ]; then
    steps=(425 430 435)
    for step in ${steps[@]};
    do
        bash scripts-test/gen_MMATH_res-run.sh $temp ${MODEL_PATH}/global_step_${step} ${MODEL_NAME}/step${step}-temp_$temp
        wait
    done
    

fi

