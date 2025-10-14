model=$1
path=$2
path_ori=$path-ori

# We increase the rollout from 1, 8, 16 to 32 on different subsets for a fast generation.
# You can run the following commands sequentially in order to prevent some errors.

bash scripts-data/self_gen_en_answer.sh $model $path_ori $path 0 1
sleep 30s
python code/self_gen-en_answer-merge.py --path $path --cnt 0 --rollout 1

bash scripts-data/self_gen_en_answer.sh $model $path_ori $path 1 8
sleep 30s
python code/self_gen-en_answer-merge.py --path $path --cnt 1 --rollout 8

bash scripts-data/self_gen_en_answer.sh $model $path_ori $path 2 16
sleep 30s
python code/self_gen-en_answer-merge.py --path $path --cnt 2 --rollout 16

bash scripts-data/self_gen_en_answer.sh $model $path_ori $path 3 32
sleep 30s
python code/self_gen-en_answer-merge.py --path $path --cnt 3 --rollout 32

python code/self_gen-en_answer-for_RL_data.py --save_dir $path