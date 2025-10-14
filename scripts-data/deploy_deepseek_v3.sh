#!/bin/bash
set -x

export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_IB_TIMEOUT=22


check_port() {
    (echo > /dev/tcp/$MASTER_ADDR/$PORT) >/dev/null 2>&1
    return $?
}

PORT=6379

if [ $RANK -eq 0 ]; then
    ray start --head --node-ip-address=$MASTER_ADDR --port $PORT
else
    while ! check_port; do
        echo "Port $PORT on $MASTER_ADDR is not open yet. Retrying in 5 seconds..."
        sleep 30s # wait for head node to start
    done
    ray start --address=$MASTER_ADDR:$PORT
fi

echo "Ray started on rank $RANK"

sleep 30s


export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

MODEL_PATH=your_path/models/DeepSeek-V3-0324

if [ $RANK -eq 0 ]; then
    while true; do
        RAY_STATUS=$(ray status 2>/dev/null)
        
        if [ $? -ne 0 ]; then
            echo "Ray cluster is unavailable, exiting monitoring..."
            break
        fi
        
        if echo "$RAY_STATUS" | grep -qE '0\.0/16\.0[[:space:]]+GPU'; then
            echo "16 GPUs are available."
            break
        fi
        
        sleep 30s
    done

    vllm serve $MODEL_PATH \
    --tensor-parallel-size 16 \
    --pipeline-parallel-size 1 \
    --trust-remote-code \
    --served-model-name deepseek-v3-0324 \
    --max-model-len 65536  \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
    
    sleep 30d
else
    sleep 365d

fi

