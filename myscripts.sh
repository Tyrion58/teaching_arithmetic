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

# 从addition训练judge-only
python train.py config/babygpt/train_judge_from_addition.py \
    --gradient_accumulation_steps=1 

# 仅训练addition,使用eval format
python train.py config/babygpt/train_addition_bal.py \
    --wandb_run_name="add-only-10000-eval-form" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out/out-check-add-only-eval-form' \
    --data_type='text' --data_format='eval_format' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_10000.txt' \
    --gradient_accumulation_steps=1 
# mix 使用eval format
python train.py config/babygpt/train_addition_bilabel.py \
    --wandb_run_name="add-bilabel-10000-eval-form" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='out/out-check-bilabel' \
    --data_type='text' --data_format='eval_format' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_bilabel_V2_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_V2_10000.txt' \
    --judge_mode='judge_op' --eval_judge='True'

# 从judge-only训练addition eval_format
python train.py config/babygpt/train_addition_from_judge.py \
    --gradient_accumulation_steps=1 --data_format='eval_format'

# 从addition训练judge-only
python train.py config/babygpt/train_judge_from_addition.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \

# 从addition训练judge-only, 但在训练judge时使用1%的计算样例
python train.py config/babygpt/train_judge_from_addition.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \
    --train_data_path="train_3digit_judge_10000_1eval.txt" \
    --out_dir='out/out-judge-from-addition-1eval-only-true' \

# 从judge-only训练addition eval_format，但在训练add时使用1%的judge样例
python train.py config/babygpt/train_addition_from_judge.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \
    --train_data_path="train_3digit_add_from_judge_10000_1eval.txt" \
    --out_dir='out/out-addition-from-judge-1eval' \
    --judge=True

# 从addition训练judge，但在训练judge时使用mix数据
python train.py config/babygpt/train_judge_from_addition.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \
    --train_data_path="train_3digit_judge_10000_1eval.txt" \
    --dataset='bal' --train_data_path="train_3digit_bilabel_V2_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_V2_10000.txt' \
    --out_dir='out/out-judge-from-addition-mix-data' \
    --wandb_run_name="judge-from-addition-mix-finetune" --label_exp=True 

# 从judge-only训练addition eval_format，但在训练add时使用mix数据
python train.py config/babygpt/train_addition_from_judge.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \
    --dataset='bal' --train_data_path="train_3digit_bilabel_V2_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_bilabel_V2_10000.txt' \
    --out_dir='out/out-add-from-judge-mix-data' \
    --wandb_run_name="addition-from-judge-mix-finetune" --label_exp=True --judge=True