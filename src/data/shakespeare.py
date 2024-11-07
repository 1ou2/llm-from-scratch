import os, requests

def download_shakespeare():
    """
    Download the Shakespeare dataset from the given URL and save it to the specified path.
    """
    data_dir = os.path.join('data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)


if __name__ == "__main__":
    #download_shakespeare()