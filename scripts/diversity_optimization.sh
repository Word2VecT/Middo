[ -z "$DATASET_PATH" ] && DATASET_PATH="LLaMA-Factory/data/MiddOptimized-llama_alpaca.json"
[ -z "$CURRENT_MODEL_PATH" ] && CURRENT_MODEL_PATH="saves/Llama3.1-8B-Middo-Alpaca"

source activate lf

while true; do
    read -p "输入超参数 m 的值 (直接按回车退出循环): " HYPER_M

    if [ -z "$HYPER_M" ]; then
        echo "退出"
        break
    fi

    python diversity_filter.py \
        --dataset_path "$DATASET_PATH" \
        --model_path "$CURRENT_MODEL_PATH" \
        --m $HYPER_M
done

source activate dd

python diversity_synthetic.py \
    --base_url $BASE_URL \
    --skey $SKEY