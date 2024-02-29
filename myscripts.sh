# 采样no-balanced addition
python sample.py --out_dir=out/out-addition-no-bal

# 训练balanced addition
python train.py config/babygpt/train_addition_bal.py
nohup python train.py config/babygpt/train_addition_bal.py &

# 训练balanced label addition
python train.py config/babygpt/train_addition_label.py
nohup python train.py config/babygpt/train_addition_label.py &

nohup python train.py config/babygpt/train_addition_bilabel.py &

nohup python train.py config/babygpt/train_addition_bilabel_5neg.py &

# addition reverse
nohup python train.py config/babygpt/train_addition_rev.py &

python train.py config/babygpt/train_addition_bal.py \
    --wandb_run_name="add-bal-rev_10000" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-new-rev' \
    --data_type='text' --data_format='reverse' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_10000.txt' \
    --eval_addition_train=True