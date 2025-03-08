import os
import sys
import random

token_dir = "data/tokenized"
sub_dirs = ["reddit/train", "wikipedia/train"]
output_dir = "data/shards/train"

os.makedirs(output_dir,exist_ok=True)

shard_files = []
for sub_dir in sub_dirs:
    full_path = os.path.join(token_dir, sub_dir)
    if not os.path.exists(full_path):
        print(f"Directory {full_path} does not exist")
        sys.exit(1)
    # list shard files that end with .npy
    shard_files.extend([os.path.join(token_dir, sub_dir,f) for f in os.listdir(full_path) if f.endswith(".npy")])
    print(f"Found {len(shard_files)} shards in {full_path}")

print(len(shard_files))
print(shard_files[0:10])

# shuffle the shards
random.shuffle(shard_files)
print("Shuffled shards")
print(shard_files[0:20])

# the shard files are named shard_{i}.npy, where i is 6 digits in length
# but we have shards with the same name in the sub_dirs
# so we want to copy the shards in output_dir and rename them
for i, shard_file in enumerate(shard_files):
    # get the shard name
    shard_name = os.path.basename(shard_file)
    # copy the shard to the output directory
    new_shard_name = f"shard_{i:06d}.npy"
    new_shard_path = os.path.join(output_dir, new_shard_name)
    # copy the shard
    os.system(f"cp {shard_file} {new_shard_path}")
    print(f"Copied {shard_file} to {new_shard_path}")

print("-------------")
print(f"{len(shard_files)} shards copied in {output_dir}")
