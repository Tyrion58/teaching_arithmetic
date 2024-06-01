# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/out-addition-bilabel'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'addition-mixed'

data_format = 'eval_format'
dataset = 'mixed'
batch_size = 256
block_size = 256 # context of up to 256 previous characters

val_data_path = 'val.bin'
ckpt_path_name = 'ckpt.pt'
eval_addition = True
eval_judge=True
judge_mode='judge_op'

data_type = 'text' # 'binary' by default, can be 'text'
judge_start = "FILE:data/addition_judge/test_3digit_judge_W1_10000.txt"
# "FILE:data/addition_judge/test_3digit_judge_W1_10000.txt" or "FILE:data/addition_judge/test_3digit_judge_W2_10000.txt"
start = "FILE:data/mixed/test_3digit_mixed_W1_10000.txt" # "FILE:data/mixed/test_3digit_mixed_W1_10000.txt" or "FILE:data/mixed/test_3digit_mixed_W2_10000.txt"
train_data_path="train_3digit_mixed_W1_10000.txt" # "train_3digit_mixed_W1_10000.txt" or "train_3digit_mixed_W2_10000.txt"
judge = True
label_exp = False
# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device='cuda:0'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
