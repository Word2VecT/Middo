import json
import argparse
import sys

def find_intersection(files):
    if not files:
        return []

    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            first_file_data = json.load(f)
        if not isinstance(first_file_data, list):
             print(f"Error: Data in {files[0]} is not a list of objects as expected.", file=sys.stderr)
             return []
        outputs_set = {(item['instruction'], item['output']) for item in first_file_data if 'instruction' in item and 'output' in item}
        output_to_object = {(item['instruction'], item['output']): item for item in first_file_data if 'instruction' in item and 'output' in item}
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error processing file {files[0]}: {e}", file=sys.stderr)
        return []

    for file_path in files[1:]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            if not isinstance(current_data, list):
                print(f"Error: Data in {file_path} is not a list of objects as expected.", file=sys.stderr)
                continue
            current_outputs = {(item['instruction'], item['output']) for item in current_data if 'instruction' in item and 'output' in item}
            outputs_set.intersection_update(current_outputs)
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue

    intersection_data = [output_to_object[output] for output in outputs_set if output in output_to_object]

    return intersection_data

def main():
    parser = argparse.ArgumentParser(description="Find the intersection of multiple JSON files based on the 'instruction' and 'output' keys.")
    parser.add_argument('files', nargs='+', help='Paths to the JSON files to process.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file. If not provided, prints to standard output.')
    
    args = parser.parse_args()

    intersected_data = find_intersection(args.files)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(intersected_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully found intersection and saved to {args.output}")
    else:
        json.dump(intersected_data, sys.stdout, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main() 