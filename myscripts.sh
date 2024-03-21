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

    python train.py config/babygpt/train_addition_bal.py \
    --wandb_run_name="add-bal-rev_10000" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-new-rev' \
    --data_type='binary' --data_format='reverse' --reverse_c=True \
    --dataset='bal' \
    --eval_addition=True --start='FILE:data/bal/test_3digit_10000.txt' \

python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_10000.txt' \


python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-rev-10000-label$" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel-label$' \
    --data_type='text' --data_format='reverse' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_bilabel_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_10000.txt' \

python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000-label-ga40" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel-label-ga40' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_10000.txt' \
    --gradient_accumulation_steps=40

python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000-last-dig-V1" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel-last-dig-V1' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_last_digV1_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_last_digV1_10000.txt' \
    --gradient_accumulation_steps=1 \

python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000-last-dig-V2" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel-last-dig-V2' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_last_dig10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_last_dig10000.txt' \
    --gradient_accumulation_steps=1 \

python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000-V2" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-bilabel-V2' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_V2_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_V2_10000.txt' \
    --gradient_accumulation_steps=1 \

# 仅训练judge
python train.py config/babygpt/train_addition_judge.py \
    --gradient_accumulation_steps=1 \

# 仅训练addition
python train.py config/babygpt/train_addition_bal.py \
    --wandb_run_name="add-only-10000" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out-check-add-only' \
    --data_type='text' --data_format='plain' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_10000.txt' \
    --gradient_accumulation_steps=1 

# 从judge-only训练addition
python train.py config/babygpt/train_addition_from_judge.py \
    --gradient_accumulation_steps=1 

# 增加GA尝试
nohup python train.py config/babygpt/train_addition_from_judge.py --gradient_accumulation_steps=40 \
    --wandb_run_name="add-only-10000GA40" --out_dir='out/out-check-add-onlyGA40' &