# 采样no-balanced addition
python sample.py --out_dir=out/out-addition-no-bal

# 训练balanced addition
python train.py config/babygpt/train_addition_bal.py
nohup python train.py config/babygpt/train_addition_bal.py &

# 训练balanced label addition
python train.py config/babygpt/train_addition_label.py
nohup python train.py config/babygpt/train_addition_label.py &