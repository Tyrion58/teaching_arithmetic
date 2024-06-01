#!/bin/bash

declare -a evals=("2000" "4000" "6000" "8000" "10000")
declare -a weights=("W1" "W2")

for weight in "${weights[@]}"
do
    for eval in "${evals[@]}"
    do
        python train.py config/babygpt/train_judge_from_addition.py \
            --wandb_project='Teach arithmetic' \
            --wandb_run_name="judge-from-add-${eval}-eval-L+${weight}" \
            --ckpt_path_name="ckpt_judge_from_add_L+${weight}_${eval}.pt" \
            --out_dir="test_out/out-check-judge-from-add-L+${weight}" \
            --init_from='resume' --resume_from='test_out/out-check-add-only-eval/ckpt_10000_acc.pt' \
            --data_type='text' --data_format='eval_format' --label_exp=False \
            --dataset='addition_judge' --eval_addition=True --train_data_path="train_3digit_judge_${weight}_${eval}.txt" \
            --eval_addition=True --start="FILE:data/bal/test_3digit_10000.txt" \
            --judge_start="FILE:data/addition_judge/test_3digit_judge_${weight}_10000.txt"
    done
done