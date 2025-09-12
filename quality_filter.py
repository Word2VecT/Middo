import json
import re
import statistics
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm




base_path = "."

clarity_file1 = "clarity_response.jsonl"
clarity_file2 = "clarity.jsonl"

completeness_file1 = "completeness_response.jsonl"
completeness_file2 = "completeness.jsonl"

factuality_file1 = "factuality_response.jsonl"
factuality_file2 = "factuality.jsonl"

output_filtered_json = "low_quality_prompt.json"
output_unfiltered_original_json = f"quality_remain.json"


# =============== 2. 辅助函数定义 ===============
def extract_data(file_path):
    lines_data = []
    scores = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {file_path}"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON line in {file_path}. Skipping.")
                continue

            predict_text = data.get("predict", "")
            all_digits_str = re.findall(r"\d+\.\d+|\d+", predict_text)
            if not all_digits_str:
                lines_data.append(data)
                scores.append(0)
                continue

            stat = float(all_digits_str[0])
            if stat < 0:
                stat = 0
            if stat > 10:
                stat = 10
            lines_data.append(data)
            scores.append(stat)

    return lines_data, scores


def plot_histogram(values, title, output_fig, range_min=0, range_max=10, bins=50):
    filtered_values = [v for v in values if range_min <= v <= range_max]
    if not filtered_values:
        print(f"Warning: no values in [{range_min}, {range_max}] for {title}.")
        return

    plt.figure(figsize=(8, 5))
    counts, edges, bars = plt.hist(
        filtered_values, bins=bins, range=(range_min, range_max), color="steelblue", edgecolor="black", alpha=0.7
    )
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xlim(range_min, range_max)

    for count, edge in zip(counts, edges[:-1]):
        if count > 0:
            plt.text(edge + (edges[1] - edges[0]) / 2, count, str(int(count)), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()
    print(f"{title} histogram saved to: {output_fig}")


def merge_single_field(item):
    instruction = item.get("instruction", "")
    inp = item.get("input", "")

    if inp:
        merged = f"#Prompt#:\n{instruction}\n{inp}"
    else:
        merged = f"#Prompt#:\n{instruction}"

    return merged


def main(
    dataset_path,
    clarity_file1,
    clarity_file2,
    completeness_file1,
    completeness_file2,
    factuality_file1,
    factuality_file2,
    output_filtered_json,
    output_unfiltered_original_json,
    m,
):
    clarity_data1, clarity_scores1 = extract_data(clarity_file1)
    clarity_data2, clarity_scores2 = extract_data(clarity_file2)
    if len(clarity_scores1) != len(clarity_scores2):
        print("Warning: the number of lines in the two Clarity files are not the same!")

    clarity_scores = [(s1 + s2) / 2 for s1, s2 in zip(clarity_scores1, clarity_scores2)]
    clarity_data = clarity_data1

    completeness_data1, completeness_scores1 = extract_data(completeness_file1)
    completeness_data2, completeness_scores2 = extract_data(completeness_file2)
    if len(completeness_scores1) != len(completeness_scores2):
        print("Warning: the number of lines in the two Completeness files are not the same!")
    completeness_scores = [(s1 + s2) / 2 for s1, s2 in zip(completeness_scores1, completeness_scores2)]
    completeness_data = completeness_data1

    factuality_data1, factuality_scores1 = extract_data(factuality_file1)
    factuality_data2, factuality_scores2 = extract_data(factuality_file2)
    if len(factuality_scores1) != len(factuality_scores2):
        print("Warning: the number of lines in the two Factuality files are not the same!")
    factuality_scores = [(s1 + s2) / 2 for s1, s2 in zip(factuality_scores1, factuality_scores2)]
    factuality_data = factuality_data1

    plot_histogram(clarity_scores, "Clarity Distribution (0-10)", "hist_clarity.png")
    plot_histogram(completeness_scores, "Completeness Distribution (0-10)", "hist_completeness.png")
    plot_histogram(factuality_scores, "Factuality Distribution (0-10)", "hist_factuality.png")

    if not (len(clarity_scores) == len(completeness_scores) == len(factuality_scores)):
        print("Warning: the number of lines in the three scores lists are not the same!")

    averages = []
    combined_data = []

    for c_score, comp_score, f_score in zip(clarity_scores, completeness_scores, factuality_scores):
        avg_score = (c_score + comp_score + f_score) / 3
        averages.append(avg_score)
        combined_data.append(
            {
                "clarity_score": c_score,
                "completeness_score": comp_score,
                "factuality_score": f_score,
                "average_score": avg_score,
            }
        )

    plot_histogram(averages, "Average Score Distribution (0-10)", "hist_average.png")

    mean_val = statistics.mean(averages) if len(averages) > 0 else 0
    std_val = statistics.stdev(averages) if len(averages) > 1 else 0.0
    threshold = mean_val - m * std_val

    print(f"Mean of average: {mean_val:.3f}")
    print(f"Std of average : {std_val:.3f}")
    print(f"Threshold      : {threshold:.3f}")

    with open(dataset_path, "r", encoding="utf-8") as f_dataset:
        dataset = json.load(f_dataset)

    if len(dataset) != len(combined_data):
        print(f"Warning: dataset size ({len(dataset)}) != scoring data size ({len(combined_data)})")

    filtered_dataset_single_field = []
    filtered_dataset_original = []
    unfiltered_dataset_original = []

    for i, item in enumerate(combined_data):
        if i >= len(dataset):
            break

        original_item = dataset[i]
        avg_score = item["average_score"]

        if avg_score < threshold:
            filtered_dataset_single_field.append({"input": merge_single_field(original_item)})
            filtered_dataset_original.append(original_item)
        else:
            unfiltered_dataset_original.append(original_item)

    with open(output_filtered_json, "w", encoding="utf-8") as fw:
        json.dump(filtered_dataset_single_field, fw, ensure_ascii=False, indent=2)
    print(f"Low-quality (single-field) data saved to: {output_filtered_json}")

    with open(output_unfiltered_original_json, "w", encoding="utf-8") as fw:
        json.dump(unfiltered_dataset_original, fw, ensure_ascii=False, indent=2)
    print(f"High-quality (original) data saved to: {output_unfiltered_original_json}")

    print("Done.")
    
    print(f"quality_data 占比: {len(filtered_dataset_single_field) / len(dataset) * 100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--clarity_file1", type=str, default="clarity_response.jsonl")
    parser.add_argument("--clarity_file2", type=str, default="clarity.jsonl")
    parser.add_argument("--completeness_file1", type=str, default="completeness_response.jsonl")
    parser.add_argument("--completeness_file2", type=str, default="completeness.jsonl")
    parser.add_argument("--factuality_file1", type=str, default="factuality_response.jsonl")
    parser.add_argument("--factuality_file2", type=str, default="factuality.jsonl")
    parser.add_argument("--output_filtered_json", type=str, default="low_quality_prompt.json")
    parser.add_argument("--output_unfiltered_original_json", type=str, default="quality_remain.json")
    parser.add_argument("--m", type=float, required=True)
    args = parser.parse_args()
    main(
        args.dataset_path,
        args.clarity_file1,
        args.clarity_file2,
        args.completeness_file1,
        args.completeness_file2,
        args.factuality_file1,
        args.factuality_file2,
        args.output_filtered_json,
        args.output_unfiltered_original_json,
        args.m,
    )
