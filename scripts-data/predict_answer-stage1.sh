### for 1.5B
MODEL_PATH=XueZhang-bjtu/1.5B-cold-start-SFT
MODEL_NAME=1.5B-cold-start-SFT

DATA_PATH=data/predict-other_data-cold-start-sft
# for 1.5B, we also conduct rejection sampling on "predict-train_data-cold-start-sft" for selecting more data for ja/ko
# DATA_PATH=data/predict-train_data-cold-start-sft



# ### for 7B
# MODEL_PATH=XueZhang-bjtu/7B-cold-start-SFT
# MODEL_NAME=7B-cold-start-SFT

# DATA_PATH=data/predict-other_data-cold-start-sft



temp=0.9
split=64
lang_array=(ja ko pt th fr)


run_on_node() {
    local node_id=$1 

    if [ -z "$node_id" ]; then
        echo "error：you must provide node_id"
        return 1
    fi

    local start_id=$((node_id * 8))
    local end_id=$((start_id + 7))

    for lang in "${lang_array[@]}"; do 
        echo "processing language: $lang"
        local CUR_LOG=$DATA_PATH/logs/${lang}-temp_${temp}
        
        if [ ! -d "$CUR_LOG" ]; then
            mkdir -p $CUR_LOG
        fi
        chmod 777 -R $CUR_LOG

        for ((name=start_id; name<=end_id; name++)); do
            local GPU_id=$((name % 8))
            echo "processing: ${lang}-GPU-${GPU_id}-${name}"
            
            CUDA_VISIBLE_DEVICES=$GPU_id nohup python -u code/predict_answer.py \
                --model_path $MODEL_PATH \
                --model_name $MODEL_NAME \
                --data_path $DATA_PATH \
                --lang $lang \
                --temp $temp \
                --split $split \
                --part $name \
                >> $CUR_LOG/gen-${name}.log 2>&1 &
        done

        wait
        sleep 30s 
    done

}



if [[ $RANK == "0" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "1" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "2" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "3" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "4" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "5" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "6" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
elif [[ $RANK == "7" ]]; then
    echo "log rank: $RANK"
    CUR_ID=$RANK
    run_on_node $CUR_ID
fi

