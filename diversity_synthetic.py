import json
import os
import argparse

import pyarrow as pa
from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import JSONDataSource, ProcessWithPrompt


def main(base_url, skey, file_path):
    os.environ["OPENAI_API_KEY"] = skey
    os.environ["OPENAI_BASE_URL"] = base_url

    with DataDreamer(f"{file_path}/generate_prompt"):
        data = JSONDataSource(
            "load sparse prompt",
            f"{file_path}/",
            f"{file_path}/sparse_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate diverse prompt",
                inputs={
                    "inputs": data.output["input"],
                },
                args={
                    "llm": llm,
                    "instruction": (
                        "You are a powerful LLM with the task to create brand new prompts for weaker open source LLMs (e.g. LLaMA, Qwen). You need to create a brand new complete prompt for them to learn in order to improve their knowledge and skills. Follow the steps below carefully.\n\
Use #Hint Prompt 1# and #Hint Prompt 2# as guiding examples. Then read the #Core Prompt# in detail. Be inspired to suggest additional new prompts, and ultimately create only one completely original and diverse #New Prompt#. \n\
Please respond strictly in the following format:\n\
#New Prompt#:"
                    ),
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                outputs={"inputs": "sparse_prompt", "generations": "steps"},
            )
            .select_columns(["sparse_prompt", "steps"])
            .save()
        )  # 挑选并保存

    with open(
        f"{file_path}/generate_prompt/generate-diverse-prompt-select_columns-save/_dataset/data-00000-of-00001.arrow",
        "rb",
    ) as source:
        table = pa.ipc.RecordBatchStreamReader(source)
        df = table.read_pandas()

    json_output = df.to_json(
        f"{file_path}/sparse_diverse_prompt.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/sparse_diverse_prompt.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        original_instruction = item.get("steps", "")
        start_index = original_instruction.find("#New Prompt#:")
        if start_index != -1:
            new_instruction = original_instruction[start_index + len("#New Prompt#:") :].strip()
            if new_instruction.startswith('"') and new_instruction.endswith('"'):
                new_instruction = new_instruction[1:-1]
            item["diverse_prompt"] = new_instruction
        else:
            print("未找到 #New Prompt#:")

    with open(f"{file_path}/sparse_diverse_prompt.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("新的 instruction 字段已成功导出到 sparse_diverse_prompt.json 文件中。")

    with DataDreamer(f"{file_path}/generate_response"):
        data = JSONDataSource(
            "load diverse prompt",
            file_path,
            f"{file_path}/sparse_diverse_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate response",
                inputs={
                    "inputs": data.output["diverse_prompt"],
                },
                args={
                    "llm": llm,
                    "instruction": ("Given the following prompt, please provide a comprehensive and accurate response."),
                    "top_p": 1.0,
                },
                outputs={"inputs": "instruction", "generations": "output"},
            )
            .select_columns(["instruction", "output"])
            .save()
        )

    with open(
        f"{file_path}/generate_response/generate-response-select_columns-save/_dataset/data-00000-of-00001.arrow",
        "rb",
    ) as source:
        table = pa.ipc.RecordBatchStreamReader(source)
        df = table.read_pandas()

    json_output = df.to_json(
        f"{file_path}/diversity.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/diversity.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["instruction"] = item["instruction"]
        item["input"] = ""
        item["output"] = item["output"]

    with open(f"{file_path}/diversity.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("数据已成功处理！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--skey", type=str, required=True)
    args = parser.parse_args()
    main(args.base_url, args.skey, args.file_path)
