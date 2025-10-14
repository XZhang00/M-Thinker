model_name=$1
model_list=($model_name)

# language_list=(en zh ar bn de es fr id it ja ko ms pt ru sw te th vi)

language_list=(ko ja pt th en zh ar es fr vi)
level_list=(low medium high top)
cnt_array=(0 1 2 3)

for cnt in ${cnt_array[@]};
do  
    export PYTHONPATH=eval_tools/PolyMath/eval
    for i in ${model_list[*]}; do
        for j in ${language_list[*]}; do
            for k in ${level_list[*]}; do
                nohup python eval_tools/PolyMath/eval/run_eval-fast.py --model $i --language $j --level $k --cnt $cnt >> logs-eval/PolyMath-temp_0.9/$i/eval.log 2>&1 &
            done
        done
    done
done

echo "waiting..."
wait 
echo "Done!"