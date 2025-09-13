[ -z "$MODEL_PATH" ] && MODEL_PATH="saves/Llama3.1-8B-Middo-Alpaca"

source activate oc

cd opencompass

EXP_NAME=$(basename "$MODEL_PATH")
OUTPUT_FILE="opencompass/opencompass/configs/models/middo/${EXP_NAME}.py"

cat << EOF > ${OUTPUT_FILE}
from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='llama-3_1-8b-instruct-vllm',
        path='${MODEL_PATH}',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=64,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
EOF
echo "The model configuration file has been created successfully: ${OUTPUT_FILE}"

cd opencompass

opencompass -w logs/test/${EXP_NAME} \
    -r \
    --retry 1 \
    --dump-eval-details False \
    --max-out-len 4096 \
    --datasets \
    mmlu_zero_shot_gen_47e2c0 IFEval_gen_3321a3 gsm8k_gen_1d7fe4 math_gen_265cce humaneval_gen_8e312c mbpp_gen_830460 hellaswag_gen_6faab5 gpqa_openai_simple_evals_gen_5aeece \
    --hf-type chat --models ${EXP_NAME} --max-num-worker ${MAX_NUM_WORKER}