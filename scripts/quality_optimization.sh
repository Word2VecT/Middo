[ -z "$DATASET_PATH" ] && DATASET_PATH="LLaMA-Factory/data/wizardlm_mistral/turn2.json"
[ -z "$CURRENT_MODEL_PATH" ] && CURRENT_MODEL_PATH="saves/Llama3.1-8B-Middo-Alpaca"

python quality_instruction_dataset.py \
    --dataset_path $DATASET_PATH

python quality_response_dataset.py \
    --dataset_path $DATASET_PATH

echo "将生成的 self_judge_*.json 文件放在 LLaMA-Factory data 文件夹中并在 dataset_info.json 文件注册，注册名与文件名一致"
read -p "按回车继续..."

source activate lf

for dimention in factuality clarity completeness factuality_response clarity_response completeness_response; do
    python quality_infer.py \
        --dataset "self_judge_$dimention" \
        --model_name_or_path $CURRENT_MODEL_PATH
done

while true; do
    read -p "输入超参数 m 的值 (直接按回车退出循环): " HYPER_M

    if [ -z "$HYPER_M" ]; then
        echo "退出"
        break
    fi

    python quality_filter.py \
        --dataset_path "$DATASET_PATH" \
        --m $HYPER_M
done

source activate dd

python quality_synthetic.py \
    --base_url $BASE_URL \
    --skey $SKEY