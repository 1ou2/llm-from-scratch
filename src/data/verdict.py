"""The verdict is a novel in the public domain"""

import urllib.request


def download():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"

    file_path = "./data/raw/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def explore():
    with open("./data/raw/the-verdict.txt", "r",encoding="utf-8") as f:
        text = f.read()
        print(f"Len :{len(text)}")
        print(text[:1000])

if __name__ == "__main__":
    explore()