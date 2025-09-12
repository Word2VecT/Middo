# caculating pre and current loss

[ -z "$PRE_MODEL_PATH" ] && PRE_MODEL_PATH="meta-llama/Llama-3.1-8B"
[ -z "$CURRENT_MODEL_PATH" ] && CURRENT_MODEL_PATH="saves/Llama3.1-8B-Middo-Alpaca"
[ -z "$DATASET" ] && DATASET="MiddOptimized-llama_alpaca"
[ -z "$DATASET_PATH" ] && DATASET_PATH="LLaMA-Factory/data/MiddOptimized-llama_alpaca.json"

source activate lf

python complexity_cal_loss.py \
    --model_name_or_path $PRE_MODEL_PATH \
    --save_name pre_loss.json \
    --dataset $DATASET

python complexity_cal_loss.py \
    --model_name_or_path $CURRENT_MODEL_PATH \
    --save_name current_loss.json \
    --dataset $DATASET

while true; do
    read -p "输入超参数 m 的值 (直接按回车退出循环): " HYPER_M

    if [ -z "$HYPER_M" ]; then
        echo "退出"
        break
    fi

    python complexity_filter.py \
        --train_file_path "$DATASET_PATH" \
        --m $HYPER_M
done

source activate dd

python complexity_synthetic.py \
    --base_url $BASE_URL \
    --skey $SKEY