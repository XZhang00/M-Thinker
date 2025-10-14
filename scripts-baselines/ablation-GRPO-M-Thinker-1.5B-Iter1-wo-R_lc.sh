set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

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


export PATH="your_path/anaconda3/envs/verl/bin":$PATH
which python

PORT=6379

check_port() {
    (echo > /dev/tcp/$MASTER_ADDR/$PORT) >/dev/null 2>&1
    return $?
}

if [ $RANK -eq 0 ]; then
    your_path/anaconda3/envs/verl/bin/ray start --head --node-ip-address=$MASTER_ADDR --port $PORT
else
    while ! check_port; do
        echo "Port $PORT on $MASTER_ADDR is not open yet. Retrying in 5 seconds..."
        sleep 30s # wait for head node to start
    done
    your_path/anaconda3/envs/verl/bin/ray start --address=$MASTER_ADDR:$PORT
fi

echo "Ray started on rank $RANK"


sleep 30s


if [ $RANK -eq 0 ]; then
    while true; do
        RAY_STATUS=$(ray status 2>/dev/null)
        
        if [ $? -ne 0 ]; then
            echo "Ray cluster is unavailable, exiting monitoring..."
            break
        fi
        
        if echo "$RAY_STATUS" | grep -qE '0\.0/64\.0[[:space:]]+GPU'; then
            echo "64 GPUs are available. Conduct training..."
            break
        fi
        
        sleep 30s
    done
    
    wandb offline
    export WANDB_MODE=offline 

    HOME=your_path/M-Thinker

    MODEL_PATH=XueZhang-bjtu/1.5B-cold-start-SFT

    lr=5e-6
    kl_loss_coef=0.0

    exp_name=M-Thinker-1.5B-RL-Iter1-wo-R_lc-lr${lr}_kl_loss_coef${kl_loss_coef}
    proj_name=DeepSeek-R1-Distill-Qwen-1.5B-GRPO_cold_SFT

    v1_train_path=data/M-Thinker-1.5B-RL-Iter1-data/train.parquet
    v1_test_path=data/M-Thinker-1.5B-RL-Iter1-data/test.parquet


    SAVE_PATH=$HOME/checkpoints/$proj_name/$exp_name
    mkdir -p ${SAVE_PATH}
    export WANDB_DIR=$SAVE_PATH


    python3 -um verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$v1_train_path \
        data.val_files=$v1_test_path \
        data.train_batch_size=512 \
        data.val_batch_size=512 \
        data.max_prompt_length=2048 \
        data.max_response_length=14336 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=${lr} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=512 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.temperature=0.9 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${proj_name} \
        trainer.experiment_name=${exp_name} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=5 \
        trainer.test_freq=100 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=${SAVE_PATH} \
        trainer.resume_mode=auto \
        trainer.total_epochs=15 \
        custom_reward_function.path=$HOME/verl/verl/utils/reward_score/Reward_acc_cta.py \
        custom_reward_function.name=compute_score_R_acc_cta_batch_async_chunk \
        reward_model.reward_manager=batch \
        trainer.val_before_train=False \
        >> ${SAVE_PATH}/log.$(date +%Y-%m-%d-%H) 2>&1


    echo "Training is done on rank 0"

    echo "Training is done on rank 0, stopping Ray..."
    ray stop --force
else
    echo "Worker rank $RANK is waiting for Ray to stop..."

    while true; do
        ray status 1>/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Ray cluster no longer available. Exiting worker..."
            break
        fi
        sleep 10m
    done

fi

echo "Rank $RANK script ended."

