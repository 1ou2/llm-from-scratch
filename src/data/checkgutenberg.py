from pathlib import Path
from collections import defaultdict
import pprint
import os
import os

def extract_character_set(file_paths):
    """
    Extract all unique characters from known valid files.
    
    Args:
        file_paths: List of paths to known valid files
    
    Returns:
        set: Set of all unique characters found in valid files
    """
    valid_chars = set()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                valid_chars.update(set(content))
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return valid_chars

def analyze_new_characters(file_path, valid_chars):
    """
    Analyze a file for characters not in the valid set.
    
    Args:
        file_path: Path to the file to analyze
        valid_chars: Set of known valid characters
    
    Returns:
        set: Set of new characters found
        dict: Dictionary with statistics about new characters
    """
    new_chars = set()
    char_stats = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find new characters and count their occurrences
            for char in content:
                if char not in valid_chars:
                    new_chars.add(char)
                    char_stats[char] += 1
                    
    except UnicodeDecodeError as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, str(e)
    
    return new_chars, char_stats, None

def analyze_files(valid_file_paths, files_to_check):
    """
    Main function to analyze files and report new characters.
    
    Args:
        valid_file_paths: List of paths to known valid files
        files_to_check: List of paths to files to analyze
    """
    # First, extract valid character set
    print("Extracting valid character set...")
    valid_chars = extract_character_set(valid_file_paths)
    print(f"Found {len(valid_chars)} valid characters")
    
    # Print valid characters (excluding common ASCII)
    print("\nValid non-ASCII characters:")
    non_ascii_chars = {c for c in valid_chars if ord(c) > 127}
    pprint.pprint(sorted(non_ascii_chars))
    
    # Analyze each file
    print("\nAnalyzing files for new characters...")
    results = []
    
    for file_path in files_to_check:
        new_chars, char_stats, error = analyze_new_characters(file_path, valid_chars)
        
        if error:
            results.append({
                'file': file_path,
                'error': error
            })
            continue
            
        if new_chars:
            results.append({
                'file': file_path,
                'new_chars': new_chars,
                'stats': char_stats
            })
    
    # Print results
    print("\nResults:")
    for result in results:
        print(f"\nFile: {result['file']}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue
            
        if result['new_chars']:
            print("  New characters found:")
            for char in sorted(result['new_chars']):
                count = result['stats'][char]
                hex_val = hex(ord(char))
                print(f"    '{char}' (hex: {hex_val}, count: {count})")
        else:
            print("  No new characters found")

# Example usage
def main():
    # Define your known valid files
    valid_files = [
        Path("data/raw/pg26504.txt"),
        Path("data/raw/pg17990.txt"),
        Path("data/raw/pg17991.txt"),
        Path("data/raw/pg17992.txt"),
        Path("data/raw/pg17989.txt"),
        Path("data/raw/pg2419.txt")
    ]
    
    # Define files to check
    files_to_check = list(Path("data/raw").glob("*.txt"))
    
    # Remove valid files from files to check
    files_to_check = [f for f in files_to_check if f not in valid_files]
    
    # Run analysis
    analyze_files(valid_files, files_to_check)

    # Optionally, save the valid character set for future reference
    def save_valid_chars(valid_chars, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            chars_list = sorted(valid_chars)
            for char in chars_list:
                hex_val = hex(ord(char))
                f.write(f"{char}\t{hex_val}\n")

def preprocess():
    # Define files to check
    files_to_check = list(Path("data/raw").glob("*.txt"))
    # create preprocessed dir if it doesn't exist
    if not os.path.exists("data/preprocessed"):
        os.makedirs("data/preprocessed")

    for file_path in files_to_check:
        print(f"Preprocessing {file_path}...")
        startline = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("***"):
                    startline = i +1
                    break
            # write all lines after startline to file
            # get basename of file and write to "preprocessed" dir
            basename = file_path.name
            preprocessed_path = Path("data/preprocessed") / basename
            with open(preprocessed_path, 'w', encoding='utf-8') as f:
                f.writelines(lines[startline:])
                

                    

if __name__ == "__main__":
    #main()
    preprocess()