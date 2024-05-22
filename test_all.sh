# 先训练一个addition only model
# 仅训练addition,使用eval format
python train.py config/babygpt/train_addition_bal.py \
    --wandb_project='Test-all-addition-judge' \
    --wandb_run_name="[test]add-only-10000-eval-form" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='test_out/out-check-add-only-eval-form' \
    --data_type='text' --data_format='eval_format' --reverse_c=False \
    --dataset='bal' --train_data_path="train_3digit_10000.txt" \
    --eval_addition=True --start='FILE:data/bal/test_3digit_10000.txt' 

# 再训练一个judge only model
python train.py config/babygpt/train_addition_judge.py\
    --wandb_project='Test-all-addition-judge' \
    --wandb_run_name="[test]judge-only-10000" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='test_out/out-check-judge-only' \
    --wandb_project='addition'

# 训练一个mixed model
python train.py config/babygpt/train_addition_mixed.py \
    --wandb_project='Test-all-addition-judge' \
    --wandb_run_name="[test]add-mixed-10000-eval-form(L+W1)" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='test_out/out-check-mixed-W1'

# 使用之前的错误方式训练mixed model
python train.py config/babygpt/train_addition_mixed.py \
    --wandb_run_name="[test]add-mixed-10000-eval-form-OLD" \
    --ckpt_path_name="ckpt_10000.pt" \
    --out_dir='test_out/out-check-mixed-OLD' \
    --train_data_path="train_3digit_bilabel_V2.txt" --start='FILE:data/mixed/test_3digit_bilabel_V2.txt' \
    --judge_mode='judge_op' --judge_start='FILE:data/mixed/test_3digit_bilabel_V2.txt'

# 从addition训练judge
python train.py config/babygpt/train_judge_from_addition.py \
    --gradient_accumulation_steps=1 --data_format='eval_format' \
    --resume_from='test_out/out-check-add-only-eval-form/ckpt_10000_acc.pt' \