import json
import argparse
import sys

def merge_json_files(files):
    """
    Merges multiple JSON files containing lists into a single list.

    Args:
        files: A list of file paths to JSON files.

    Returns:
        A single list containing all items from the input files.
    """
    merged_data = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                # If the data is not a list, we can choose to append it as a single item
                # or print a warning. For this case, we'll print a warning and skip.
                print(f"Warning: Data in {file_path} is not a list. Skipping this file.", file=sys.stderr)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue
    
    return merged_data

def main():
    parser = argparse.ArgumentParser(description="Merge multiple JSON files into one.")
    parser.add_argument('files', nargs='+', help='Paths to the JSON files to merge.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file. If not provided, prints to standard output.')
    
    args = parser.parse_args()

    final_data = merge_json_files(args.files)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully merged {len(args.files)} files into {args.output}")
    else:
        json.dump(final_data, sys.stdout, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main() 