import json
import os
import argparse

system_prompt = "We would like to request your feedback on the {dimention} of the prompt displayed below."

user_prompt = "Please rate according to the {dimention} of the prompt to evaluate {explain}. Each prompt is scored on a scale of 0 to 10, with higher scores indicating higher {dimention}. Try to avoid scoring a full 10. Give your rating number first, then give a explanation of your rating."

dimention = {
    "factuality": "whether the information provided in the prompt is accurate and based on reliable facts and data",
    "clarity": "whether the prompt is clear and understandable, and whether it uses concise language and structure",
    "completeness": "whether the prompt provides sufficient information and details",
}


def main(dataset_path, file_path):
    for sub_dimention, explain in dimention.items():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item["input"]:
                prompt = f"{system_prompt.format(dimention=sub_dimention)}\n\nPrompt:\n{item['instruction']}\n{item['input']}\n\n{user_prompt.format(dimention=sub_dimention, explain=explain)}"
            else:
                prompt = f"{system_prompt.format(dimention=sub_dimention)}\n\nPrompt:\n{item['instruction']}\n\n{user_prompt.format(dimention=sub_dimention, explain=explain)}"

            item["instruction"] = prompt
            item["input"] = ""
            item["output"] = ""

        with open(f"{file_path}/self_judge_{sub_dimention}.json", "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"新的 instruction 字段已成功导出到 {sub_dimention}.json 文件中。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, default=".")
    args = parser.parse_args()
    main(args.dataset_path, args.file_path)
