import os
import numpy as np
import tiktoken  # Assuming tiktoken is correctly installed and configured

# Define file paths
base_path = os.path.dirname(os.path.abspath(__file__))

wt2_valid_path = os.path.join(base_path, 'ptb.valid.txt')

# Function to read data from a file
def read_data(file_path):
    with open(file_path, 'r') as f:
        return f.read()


wt2_val_data = read_data(wt2_valid_path)

# Get GPT-2 BPE encoding
enc = tiktoken.get_encoding("gpt2")

wt2_val_ids = enc.encode_ordinary(wt2_val_data)

print(f"WT2 val has {len(wt2_val_ids):,} tokens")

np.array(wt2_val_ids, dtype=np.uint16).tofile(os.path.join(base_path, 'wt2_val.bin'))
