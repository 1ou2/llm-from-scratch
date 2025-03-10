Model GPT context 256
B = 64 -> crash

# Run basique
B = 2
Steps = 50
Device = cpu
Time = 100s

B = 2
Steps = 50
Device = cpu
Time = 11.45s

Time: 26.638831853866577
Tokens: 104448
Tokens/s: 3920.892649233775

B = 16 -> out of memory

# Torch.compile
RuntimeError: Found NVIDIA GeForce GTX 1070 Ti which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.1

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

# Bfloat16
Changement du format pour
 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss = model(xgen)

# Tokenizer
# tiktoken gpt-2
Premier essai, utilisation du tokenizer de GPT-2
tokenizer = tiktoken.get_encoding("gpt2")
Le vocabulaire est 50256 mots.
Lors du pre-processing, on transforme notre dataset wikipedia français en shards de taille de taille 1 million.
À l'issue de cette phase on obtient 2663 shards.

# custom tokenizer
On entraine un tokenizer sur notre dataset avec un vocabulaire taille de 32768 
```python
def train():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>", ]

    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Load the dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")

    # Get 10% of the training split
    texts = dataset["train"].select(range(len(dataset["train"]) // 10))["text"]

    # Train on your dataset while adding special tokens
    tokenizer.train_from_iterator(
        texts, 
        vocab_size=32768, 
        min_frequency=2, 
        special_tokens=special_tokens
    )

    # Save the tokenizer
    # ls french_tokenizer/
    # gabgpt-merges.txt  gabgpt-vocab.json
    tokenizer.save_model("french_tokenizer",prefix="gabgpt")
```

Alors qu'on a un plus petit nombre de tokens à notre disposition, le pre-processing va donner 1892 shards.
On a donc un gain de `28.9%` !
L'explication est que le tokenizer de gpt-2 n'a pas été optimisé pour le français contrairement à celui qu'on a mis en place

# Performance
Amazon g4dn.xlarge
GPU T4 : 
Tesla T4 does not support bfloat16 compilation natively,
step   100 | loss: 9.018543 | lr: 0.0000026 | norm: 6.0303 | dt: 195185.89ms | tok/sec: 4197.02
step   200 | loss: 8.389067 | lr: 0.0000052 | norm: 3.5643 | dt: 165684.37ms | tok/sec: 4944.34
(.venv) ubuntu@ip-172-31-2-164:~/llm-from-scratch$ nvidia-smi 
Tue Mar  4 19:45:04 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.10              Driver Version: 570.86.10      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:1E.0 Off |                    0 |
| N/A   29C    P0             27W /   70W |       1MiB /  15360MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# Hyperbolic
## Préparation des tokens pour transfert
La vitesse d'upload étant limitée à 1Mb/s, une façon d'optimiser et de lancer des upload de tokens en parallèle. Pour cela, on prépare des archives de 200M qu'on va uploader en parallèle.
On installe le package parallel:
`sudo apt install parallel`

Aller dans le répertoire contenant les tokens de training
```bash
tar -cvf training.tar train/
split -b 200M training.tar tokens_part
```

## Configuration initiale
Connection en SSH, exemple:
`ssh ubuntu@horrible-lilyofthevalley-clam.1.cricket.hyperbolic.xyz -p 31553`


```bash
sudo apt update
supo apt install python3-pip
git clone https://github.com/1ou2/llm-from-scratch.git
cd llm-from-scratch
python3 -m venv .venv
pip install -r requirements
```
## Copy des tokens
Copie des données de validation
scp -P 31553 data/shards/valid/shard_000000.npy /home/ubuntu/llm-from-scratch/data/shards/valid

Copie des données d'entrainements, 20 fichiers dans le répertoire transfer
```bash
parallel -j 20 scp -P 31553 transfer/{} ubuntu@horrible-lilyofthevalley-clam.1.cricket.hyperbolic.xyz:~/llm-from-scratch/data/shards/train ::: token_part_*
```



