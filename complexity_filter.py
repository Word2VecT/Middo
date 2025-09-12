import json
import os
import argparse

import numpy as np


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def calculate_threshold(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean + 1 * std


def find_common_significant_indices(data1, data2, threshold1, threshold2):
    result = []
    for i, (value1, value2) in enumerate(zip(data1, data2)):
        if value1 > threshold1 and value2 > threshold2:
            result.append({"index": i + 1, "pre_loss": value1, "current_loss": value2})
    return result


def identify_big_loss(pre_loss_path, current_loss_path, big_loss_out_path):
    data1 = read_json(pre_loss_path)
    data2 = read_json(current_loss_path)

    if len(data1) != len(data2):
        raise ValueError("pre_loss 和 current_loss 文件长度不一致，无法处理")

    threshold1 = calculate_threshold(data1)
    threshold2 = calculate_threshold(data2)

    significant_items = find_common_significant_indices(data1, data2, threshold1, threshold2)

    save_json(big_loss_out_path, significant_items)

    print(f"Big loss 数据已保存到: {big_loss_out_path}")
    return significant_items


def combine_instruction_input(instruction, input_text):
    if instruction and input_text:
        return f"#Prompt#:\n{instruction}\n{input_text}"
    elif instruction:
        return f"#Prompt#:\n{instruction}"
    elif input_text:
        return f"#Prompt#:\n{input_text}"
    else:
        return ""


def generate_hard_origin_and_remaining_data(
    big_loss, train_data, hard_data_out_path, remaining_data_out_path
):
    hard_data = []
    origin_hard_data = []
    selected_indices = set()

    for item in big_loss:
        index = item.get("index")
        if index is None:
            print("big_loss 中的一个条目缺少 'index' 字段，跳过该条目。")
            continue
        if not isinstance(index, int) or index < 1:
            print(f"无效的 'index' 值: {index}，应为从 1 开始的整数，跳过该条目。")
            continue
        if index > len(train_data):
            print(f"'index' 值 {index} 超出了 train.json 的范围（总共有 {len(train_data)} 条数据），跳过该条目。")
            continue
        train_entry = train_data[index - 1]
        instruction = train_entry.get("instruction", "").strip()
        input_text = train_entry.get("input", "").strip()
        combined_input = combine_instruction_input(instruction, input_text)
        hard_data.append({"input": combined_input})
        origin_hard_data.append(train_entry)
        selected_indices.add(index - 1) 

    remaining_data = []
    for i, train_entry in enumerate(train_data):
        if i not in selected_indices:
            remaining_data.append(train_entry)

    save_json(hard_data_out_path, hard_data)
    print(f"Hard data 已成功导出到: {hard_data_out_path}")

    save_json(remaining_data_out_path, remaining_data)
    print(f"Remaining data 已成功导出到: {remaining_data_out_path}")
    
    print(f"hard_data 占比: {len(hard_data) / len(train_data) * 100}%")


def main(
    train_file_path,
    pre_loss_path,
    current_loss_path,
    big_loss_out_path,
    hard_data_out_path,
    remaining_data_out_path,
):
    try:
        big_loss = identify_big_loss(pre_loss_path, current_loss_path, big_loss_out_path)
    except Exception as e:
        print(f"识别 big_loss 时出错: {e}")
        return

    if not os.path.isfile(train_file_path):
        print(f"train.json 文件不存在: {train_file_path}")
        return

    try:
        with open(train_file_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"无法解析 train.json 文件: {e}")
        return
    except Exception as e:
        print(f"加载 train.json 文件时出错: {e}")
        return

    if not isinstance(train_data, list):
        print("train.json 文件的格式不正确，应该是一个 JSON 数组。")
        return

    try:
        generate_hard_origin_and_remaining_data(big_loss, train_data, hard_data_out_path, remaining_data_out_path)
    except Exception as e:
        print(f"生成 hard_data、origin_hard_data 和 remaining_data 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--pre_loss_path", type=str, default="pre_loss.json")
    parser.add_argument("--current_loss_path", type=str, default="current_loss.json")
    parser.add_argument("--big_loss_out_path", type=str, default="big_loss.json")
    parser.add_argument("--hard_data_out_path", type=str, default="hard_prompt.json")
    parser.add_argument("--remaining_data_out_path", type=str, default="complexity_remain.json")
    parser.add_argument("--m", type=float, required=True)
    args = parser.parse_args()
    main(
        args.train_file_path,
        args.pre_loss_path,
        args.current_loss_path,
        args.big_loss_out_path,
        args.hard_data_out_path,
        args.remaining_data_out_path,
    )
