import json
import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model


def get_sentence_embedding(tokenizer, model, sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)


    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    sentence_embedding = last_hidden_state.mean(dim=1)
    return sentence_embedding.cpu().numpy()


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_cosine_similarity_torch(embeddings_list, device="cuda"):
    if isinstance(embeddings_list, np.ndarray):
        embeddings_tensor = torch.from_numpy(embeddings_list).float().to(device)
    else:
        embeddings_tensor = torch.from_numpy(np.vstack(embeddings_list)).float().to(device)
    embeddings_normed = F.normalize(embeddings_tensor, p=2, dim=1)
    cosine_sim_matrix = embeddings_normed @ embeddings_normed.t()
    return cosine_sim_matrix


def main(
    top_k,
    model_path,
    m,
    dataset_path,
    output_embedding_path,
    embedding_json_path,
    output_similarity_path,
    output_diverse_path,
    output_processed_diverse_path,
    output_diverse_embedding_path,
):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Total instructions in dataset: {len(dataset)}")

    embeddings = []
    instructions = []

    if os.path.exists(output_embedding_path) and os.path.getsize(output_embedding_path) > 0:
        print(f"Loading existing embeddings from {output_embedding_path}...")
        data = np.load(output_embedding_path, allow_pickle=True)
        instructions = data["instructions"].tolist()
        embeddings = data["embeddings"]
    elif os.path.exists(embedding_json_path) and os.path.getsize(embedding_json_path) > 0:
        print(f"Loading existing embeddings from {embedding_json_path}...")
        with open(embedding_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        instructions = [item["instruction"] for item in json_data]
        embeddings = [np.array(item["embedding"]) for item in json_data]
        embeddings = np.array(embeddings)
    else:
        print("Loading model and tokenizer...")
        tokenizer, model = load_model(model_path)
        model.eval()

        print("No existing embeddings found. Computing new embeddings...")
        for entry in tqdm(dataset, desc="Computing embeddings"):
            instruction = entry.get("instruction", "")
            input_text = entry.get("input", "")
            if input_text:
                instruction = f"{instruction}\n{input_text}"
            if instruction:
                embedding = get_sentence_embedding(tokenizer, model, instruction)
                embeddings.append(embedding.squeeze())  # shape: (hidden_size,)
                instructions.append(instruction)

        print(f"Saving embeddings to {output_embedding_path}...")
        np.savez_compressed(output_embedding_path, instructions=np.array(instructions), embeddings=np.array(embeddings))

    num_instructions = len(instructions)
    print(f"Total embeddings computed: {num_instructions}")

    if os.path.exists(output_similarity_path) and os.path.getsize(output_similarity_path) > 0:
        print(f"Loading existing similarity data from {output_similarity_path}...")
        with open(output_similarity_path, "rb") as f:
            similarity_data = pickle.load(f)
        print("Similarity data loaded.")
        top_similarities = [item[f"top_{top_k}_cosine_similarities"] for item in similarity_data]
        top_instructions_list = [item[f"top_{top_k}_similar_instructions"] for item in similarity_data]
        average_similarities = [item[f"average_top_{top_k}_cosine_similarity"] for item in similarity_data]
    else:
        print("Computing cosine similarity matrix with PyTorch (CUDA)...")
        cosine_sim_matrix = compute_cosine_similarity_torch(embeddings, device="cuda")
        print("Cosine similarity matrix computed.")

        print(f"Finding top {top_k} similar instructions for each instruction...")
        top_similarities = []
        top_instructions_list = []
        average_similarities = []

        for idx in tqdm(range(num_instructions), desc="Processing similarities"):
            row_sim = cosine_sim_matrix[idx]
            row_sim[idx] = -float("inf")
            top_vals, top_inds = torch.topk(row_sim, k=top_k, largest=True)
            top_vals = top_vals.cpu().numpy().tolist()
            top_inds = top_inds.cpu().numpy().tolist()
            top_instr = [instructions[i] for i in top_inds]
            avg_sim = float(np.mean(top_vals))

            top_similarities.append(top_vals)
            top_instructions_list.append(top_instr)
            average_similarities.append(avg_sim)

        print("Compiling embedding data and similarity information...")
        similarity_data = []
        for i in range(num_instructions):
            similarity_data.append(
                {
                    "instruction": instructions[i],
                    f"top_{top_k}_similar_instructions": top_instructions_list[i],
                    f"top_{top_k}_cosine_similarities": top_similarities[i],
                    f"average_top_{top_k}_cosine_similarity": average_similarities[i],
                }
            )
        print(f"Saving similarity information to {output_similarity_path}...")
        with open(output_similarity_path, "wb") as f:
            pickle.dump(similarity_data, f)

    overall_mean = np.mean(average_similarities)
    overall_std = np.std(average_similarities)
    threshold = overall_mean - m * overall_std
    print(f"Overall Mean: {overall_mean:.4f}, Overall Std Dev: {overall_std:.4f}, Threshold: {threshold:.4f}")

    print("Selecting instructions with low average similarity...")
    selected_instructions = []
    selected_embeddings = []
    for i in range(num_instructions):
        if average_similarities[i] < threshold:
            selected_instructions.append(
                {
                    "instruction": instructions[i],
                    f"average_top_{top_k}_cosine_similarity": average_similarities[i],
                    f"top_{top_k}_similar_instructions": top_instructions_list[i],
                    f"top_{top_k}_cosine_similarities": top_similarities[i],
                }
            )
            selected_embeddings.append(embeddings[i])
    print(f"Total diverse instructions selected: {len(selected_instructions)}")

    print(f"Saving diverse data to {output_diverse_path}...")
    with open(output_diverse_path, "w", encoding="utf-8") as f:
        json.dump(selected_instructions, f, ensure_ascii=False, indent=4)

    print("Creating processed diverse data with formatted 'input' fields...")
    processed_diverse_data = []
    for entry in tqdm(selected_instructions, desc="Processing diverse instructions"):
        core_instruction = entry["instruction"]
        hint_instructions = entry[f"top_{top_k}_similar_instructions"]
        input_field = ""
        for idx, hint in enumerate(hint_instructions, 1):
            input_field += f"#Hint Prompt {idx}#:\n{hint}\n\n"
        input_field += f"#Core Prompt#:\n{core_instruction}"
        processed_diverse_data.append({"input": input_field})
    print(f"Saving processed diverse data to {output_processed_diverse_path}...")
    with open(output_processed_diverse_path, "w", encoding="utf-8") as f:
        json.dump(processed_diverse_data, f, ensure_ascii=False, indent=4)

    print(f"Saving diverse embeddings to {output_diverse_embedding_path}...")
    diverse_instructions = [entry["instruction"] for entry in selected_instructions]
    np.savez_compressed(
        output_diverse_embedding_path,
        instructions=np.array(diverse_instructions),
        embeddings=np.array(selected_embeddings),
    )

    print("Processing complete.")
    
    print(f"diversity_data 占比: {len(selected_instructions) / len(dataset) * 100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--m", type=float, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_embedding_path", type=str, default="embeddings.npz")
    parser.add_argument("--embedding_json_path", type=str, default="embeddings.json")
    parser.add_argument("--output_similarity_path", type=str, default="similarities.pkl")
    parser.add_argument("--output_diverse_path", type=str, default="sparse_data.json")
    parser.add_argument("--output_processed_diverse_path", type=str, default="sparse_prompt.json")
    parser.add_argument("--output_diverse_embedding_path", type=str, default="diverse_embeddings.npz")
    args = parser.parse_args()
    main(
        args.top_k,
        args.model_path,
        args.m,
        args.dataset_path,
        args.output_embedding_path,
        args.embedding_json_path,
        args.output_similarity_path,
        args.output_diverse_path,
        args.output_processed_diverse_path,
        args.output_diverse_embedding_path,
    )
