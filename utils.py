import torch
import tiktoken

def get_ppl(model, start, device):

    # ok let's assume gpt-2 encodings by default
    # print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

    start_ids = encode(start)
    x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...])
    y = (torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
            logits, loss = model(x, y)
            return loss.exp().item()
