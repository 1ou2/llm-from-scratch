import os
from pathlib import Path

# get book files from project gutenberg
monte_cristo = [17989,17990,17991,17992]
non_fiction = [70312,51516,32948,16234,69621,53536,39884,19854,16237,37053]
import chardet



def detect_encoding(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def test_encoding():
    text = """Les deux frŠres

Le 20 ao–t 1672, la ville de la Haye, si vivante, si"""
    # French special characters (accented 
    # letters, etc.) were entered as DOS upper-  ASCII characters.
    test_string = "frŠres"
    for encoding in ["utf-8",'cp850', 'cp437', 'iso-8859-1',"UTF-8-SIG",'utf-8-sig']:
        try:
            bytes_data = text.encode('latin1')  # Preserve the byte values
            decoded = bytes_data.decode(encoding)
            print(f"{encoding}: {decoded}")
        except:
            print(f"{encoding}: failed")
    
def convert_to_utf8(input_file, output_file):
    # First try CP850 (DOS Latin-1) as it's most likely
    encodings_to_try = ['cp850', 'cp437', 'iso-8859-1','utf-8-sig',"cp1252"]
    
    with open(input_file, 'rb') as file:
        content = file.read()
        
    # Try different encodings
    for encoding in encodings_to_try:
        try:
            decoded_text = content.decode(encoding)
            # If successful, write with UTF-8 encoding
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(decoded_text)
            print(f"Successfully converted using {encoding} to UTF-8")
            return True
        except UnicodeDecodeError:
            continue
    
    print("Could not find correct encoding")
    return False


def download_gutenberg_book(book_id,data_dir):
    gutenberg_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    filename = f"pg{book_id}.txt"
    # check if file already exists
    if os.path.exists(f"{data_dir}/{filename}"):
        print(f"File {filename} already exists")
        return
    os.system(f"wget {gutenberg_url} -P {data_dir}")
   
def get_dumas_id(filename):
    # read lines of dumas.txt files
    with open(filename, "r") as f:
        lines = f.readlines()
    # get the book ids
    book_ids = []
    for line in lines:
        # check if line is an integer
        if line.strip().isdigit():
            book_ids.append(int(line.strip()))
    return sorted(book_ids)

def save_dumas_id(book_ids):
    data_dir = "./data/resources"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open("./data/resources/dumas.txt", "w") as f:
        for book_id in book_ids:
            f.write(f"{book_id}\n")

def preprocess():
    # Define files to check
    files_to_check = list(Path("data/raw/gutenberg").glob("*.txt"))
    preprocessed_dir = Path("data/preprocessed/gutenberg")
    # create preprocessed dir if it doesn't exist
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    for file_path in files_to_check:
        print(f"Preprocessing {file_path}...")
        startline = 0
        endline = -1
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # only keep lines that are not empty
            lines = [line for line in lines if line.strip() != ""]

            # looking for line : *** START OF THE PROJECT GUTENBERG EBOOK
            for i, line in enumerate(lines):
                if line.startswith("***"):
                    startline = i +1
                    break

            if startline != 0:
                # after what should be the start line we still have other comments in the subsequent lines
                headers = ["produced", "distributed", "proofreading","etext","file","by","http","is","mobipocket"
                "online","available","e-text","the", "Bibliothèque",
                "from","(http","of","at","you","before","whatsoever", "Text", "and the", "we",
                "this", "is", "made","encoded", "note:"]
                for i, line in enumerate(lines[startline:]):
                    if line.strip() == "":
                        startline += 1
                    else:
                        start_with_header = False
                        # check if line starts with any of the headers
                        for header in headers:
                            if line.lower().startswith(header):
                                startline += 1
                                start_with_header = True
                        # did not find a line starting with a header, nor an empty line
                        # we should be at the start of the book
                        if not start_with_header:
                            break

                # looking for line : *** END OF THE PROJECT GUTENBERG EBOOK
                for i, line in enumerate(lines[startline:]):
                    if line.startswith("***"):
                        endline = i
                        break


            # write all lines after startline to file
            # get basename of file and write to "preprocessed" dir
            basename = file_path.name
            preprocessed_path = Path(preprocessed_dir) / basename
            with open(preprocessed_path, 'w', encoding='utf-8') as f:
                f.writelines(lines[startline:endline])               

if __name__ == "__main__":
    #

    data_dir = "./data/raw/gutenberg"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    book_ids = get_dumas_id("./data/resources/dumas.txt")
    
    for book_id in book_ids:
        download_gutenberg_book(book_id, data_dir)

    preprocess()