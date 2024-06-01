# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/out-addition-from-mistakes-instruction'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
# 从一个add-only的model开始做fine-tune
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
resume_from = 'out/out-check-add-only-eval-form/ckpt_10000_acc.pt'

wandb_log = True # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'addition-from-mistakes'

dataset = 'bal'
batch_size = 256
block_size = 256 # context of up to 256 previous characters
train_data_path = "answer_1st.txt"
val_data_path = "answer_1st.txt"
ckpt_path_name = 'ckpt.pt'
eval_addition = True

data_type = 'text' # 'binary' by default, can be 'text'
instruction = True
# 判断测试集使用test_3digit_judge_10000.txt, 这是合理的，因为这是我们自己构造的错误数据集，不完全等同于model自己犯的错误
judge_start = "FILE:data/addition_judge/test_3digit_judge_10000.txt"
start = "FILE:data/addition_judge/test_3digit_add_from_judge_10000.txt"

eval_addition = True
eval_judge = True
judge_mode = 'judge_op' # 'bilabel' or 'judge_op'
judge = True
reverse_c = False
data_format = 'eval_format' # 'plain' or 'reverse' or 'algo_reasoning'
# label_exp = True 
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
