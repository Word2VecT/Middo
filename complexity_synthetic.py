import json
import os
import argparse

import pyarrow as pa
from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import JSONDataSource, ProcessWithPrompt


def main(file_path, base_url, skey):
    os.environ["OPENAI_API_KEY"] = skey
    os.environ["OPENAI_BASE_URL"] = base_url

    with DataDreamer(f"{file_path}/generate_prompt"):
        data = JSONDataSource(
            "load hard prompt",
            f"{file_path}/",
            f"{file_path}/hard_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate simple prompt",
                inputs={
                    "inputs": data.output["input"],
                },
                args={
                    "llm": llm,
                    "instruction": (
                        "You are a powerful LLM with the task to simplify the given #Prompt# for weaker open source LLMs (e.g. LLaMA, Qwen). Since the original #Prompt# is hard for them to handle, you need to rewrite it into a simpler version that these weaker LLMs can handle or learn from more easily. Follow the steps below carefully.\n\
Step 1: Read the #Prompt# in detail. Suggest possible methods to make this prompt easier for weaker LLMs to handle or learn from, then list these methods as your #Methods List#.\n\
Step 2: Create a comprehensive plan to simplify the #Prompt# using several methods from your #Methods List#. Present your plan in a clear, step-by-step format.\n\
Step 3: Execute your plan to rewrite the #Prompt# into a simpler, more learnable version. You can change scenarios, contexts, or settings as needed. Your goal is to ensure that weaker LLMs learn from this prompt, rather than just memorizing an answer.\n\
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
                outputs={"inputs": "hard_prompt", "generations": "steps"},
            )
            .select_columns(["hard_prompt", "steps"])
            .save()
        )

    with open(
        f"{file_path}/generate_prompt/generate-simple-prompt-select_columns-save/_dataset/data-00000-of-00001.arrow",
        "rb",
    ) as source:
        table = pa.ipc.RecordBatchStreamReader(source)
        df = table.read_pandas()

    json_output = df.to_json(
        f"{file_path}/hard_simple_prompt.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/hard_simple_prompt.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        original_prompt = item.get("steps", "")

        start_index = original_prompt.find("#Final Rewritten Prompt#:")
        if start_index != -1:
            new_prompt = original_prompt[start_index + len("#Final Rewritten Prompt#:") :].strip()
            if new_prompt.startswith('"') and new_prompt.endswith('"'):
                new_prompt = new_prompt[1:-1]
            if new_prompt.startswith("'") and new_prompt.endswith("'"):
                new_prompt = new_prompt[1:-1]

            item["simple_prompt"] = new_prompt
        else:
            print("未找到 #Final Rewritten Prompt#:")

    with open(f"{file_path}/hard_simple_prompt.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("新的 prompt 字段已成功导出到 hard_simple_prompt.json 文件中。")

    with DataDreamer(f"{file_path}/generate_response"):
        data = JSONDataSource(
            "load simple prompt",
            file_path,
            f"{file_path}/hard_simple_prompt.json",
        )

        llm = OpenAI(
            model_name="gpt-4o-mini",
        )

        explain = (
            ProcessWithPrompt(
                "generate response",
                inputs={
                    "inputs": data.output["simple_prompt"],
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
        f"{file_path}/complexity.json",
        orient="records",
        lines=False,
        force_ascii=False,
    )

    print("Arrow 文件已成功转换为 JSON！")

    with open(f"{file_path}/complexity.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["instruction"] = item["instruction"]
        item["input"] = ""
        item["output"] = item["output"]

    with open(f"{file_path}/complexity.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("数据已成功处理！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--skey", type=str, required=True)
    args = parser.parse_args()
    main(args.file_path, args.base_url, args.skey)
