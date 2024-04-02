import os

out_dir = 'collapse-generated'
eval_interval = 1
eval_iters = 100
wandb_log = True # feel free to turn on
wandb_project = 'collapse-lm'
wandb_run_name = 'ptb-gpt2-lr4e-5'

dataset = 'collapse-lm'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# ofensive_hate has 508,764 tokens, so 1 epoch ~= 15 iters
#val has 44,428 tokens
batch_size = 1
gradient_accumulation_steps = 1
max_iters = 1

# finetune at constant LR
learning_rate = 4e-5
decay_lr = False
block_size=100

# set to True if you want to use a contrastive learning
# contrastive_learning = False

# ckpt_path = os.path.join(out_dir, f'{wandb_run_name}.pt')