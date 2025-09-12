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
            "load low quality prompt",
            f"{file_path}/",
            f"{file_path}/low_quality_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate quality prompt",
                inputs={
                    "inputs": data.output["input"],
                },
                args={
                    "llm": llm,
                    "instruction": (
                        "You are a powerful LLM with the task to rewrite the given #Prompt# for weaker open source LLMs (e.g., LLaMA, Qwen). Since the original #Prompt# is of poor quality for them to learn, you need to rewrite it into a higher quality version that these weaker models can better learn from. Follow the steps below carefully.\n\
    Step 1: Read the #Prompt# in detail. Identify reasons for the poor quality of the #Prompt#. Suggest possible methods to improve the quality of the #Prompt#, then list these methods as your #Methods List#.\n\
    Step 2: Create a comprehensive plan to rewrite the #Prompt# using several methods from your #Methods List#. Present your plan in a clear, step-by-step format.\n\
    Step 3: Execute your plan to rewrite the #Prompt# into a higher quality version.\n\
    Step 4: Finally, review your rewritten version for any problems. Present only the #Final Rewritten Prompt#, without any additional explanation.\n\
    Please respond strictly in the following format:\n\
    Step 1 #Methods List#:\n\
    Step 2 #Plan#:\n\
    Step 3 #Rewritten Prompt#:\n\
    Step 4 #Final Rewritten Prompt#:"
                    ),
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                outputs={"inputs": "low_quality_prompt", "generations": "steps"},
            )
            .select_columns(["low_quality_prompt", "steps"])
            .save()
        ) 

    with open(
        f"{file_path}/generate_prompt/generate-quality-prompt-select_columns-save/_dataset/data-00000-of-00001.arrow",
        "rb",
    ) as source:
        table = pa.ipc.RecordBatchStreamReader(source)
        df = table.read_pandas()

    json_output = df.to_json(
        f"{file_path}/low_high_quality_prompt.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/low_high_quality_prompt.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        original_instruction = item.get("steps", "")

        start_index = original_instruction.find("#Final Rewritten Prompt#:")
        if start_index != -1:
            new_instruction = original_instruction[start_index + len("#Final Rewritten Prompt#:") :].strip()
            if new_instruction.startswith('"') and new_instruction.endswith('"'):
                new_instruction = new_instruction[1:-1]

            item["high_quality_prompt"] = new_instruction
        else:
            print(f"{original_instruction}未找到 #Final Rewritten Prompt#:")

    with open(f"{file_path}/low_high_quality_prompt.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("新的 instruction 字段已成功导出到 low_high_quality_prompt.json 文件中。")

    with DataDreamer(f"{file_path}/generate_response"):
        data = JSONDataSource(
            "load low quality prompt",
            file_path,
            f"{file_path}/low_high_quality_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate response",
                inputs={
                    "inputs": data.output["high_quality_prompt"],
                },
                args={
                    "llm": llm,
                    "instruction": (
                        "Given the following prompt, please provide a comprehensive, accurate and high quality response."
                    ),
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
        f"{file_path}/quality.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/quality.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["instruction"] = item["instruction"]
        item["input"] = ""
        item["output"] = item["output"]

    with open(f"{file_path}/quality.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("数据已成功处理！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--skey", type=str, required=True)
    args = parser.parse_args()
    main(args.file_path, args.base_url, args.skey)