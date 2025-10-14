# MODEL_PATH=XueZhang-bjtu/1.5B-cold-start-SFT

# DATA_PATH=data/M-Thinker-1.5B-RL-Iter1-data-ori
# SAVE_PATH=data/M-Thinker-1.5B-RL-Iter1-data


# temp=0.9


# lang=en
# cnt=0
# rollout=1
# cnt=1
# rollout=8
# cnt=2
# rollout=16
# cnt=3
# rollout=32

MODEL_PATH=$1

DATA_PATH=$2
SAVE_PATH=$3

temp=0.9


lang=en
cnt=$4
rollout=$5



run_on_node() {
    local node_id=$1 

    if [ -z "$node_id" ]; then
        echo "error：you must provide node_id"
        return 1
    fi

    local start_id=$((node_id * 8))
    local end_id=$((start_id + 7))

    echo "processing language: $lang"
    local CUR_LOG=$SAVE_PATH/logs-${lang}-temp_${temp}
    
    if [ ! -d "$CUR_LOG" ]; then
        mkdir -p $CUR_LOG
    fi
    chmod 777 -R $CUR_LOG


    for ((name=start_id; name<=end_id; name++)); do
        local GPU_id=$((name % 8))
        echo "processing: ${RANK}-${lang}-GPU-${GPU_id}-${name}"
        
        CUDA_VISIBLE_DEVICES=$GPU_id nohup python -u code/self_gen-en_answer.py \
            --model_path $MODEL_PATH \
            --data_path $DATA_PATH/train.parquet \
            --temp $temp \
            --split $split \
            --part $name \
            --save_path $SAVE_PATH \
            --cnt $cnt \
            --rollout $rollout \
            >> $CUR_LOG/gen-${name}.log 2>&1 &
    done

    wait
    sleep 30s
}

split=64
if [[ $RANK == "0" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "1" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "2" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "3" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "4" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "5" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "6" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
elif [[ $RANK == "7" ]]; then
    echo "log rank: $RANK"
    run_on_node $RANK
fi




