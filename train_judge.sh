#!/bin/bash

declare -a evals=("2000" "4000" "6000" "8000" "10000")
declare -a weights=("W1" "W2")

for weight in "${weights[@]}"
do
    for eval in "${evals[@]}"
    do
        python train.py config/babygpt/train_addition_judge.py \
            --wandb_project='Teach arithmetic' \
            --wandb_run_name="judge-only-${eval}-L+${weight}" \
            --ckpt_path_name="ckpt_judge_only_${eval}_${weight}.pt" \
            --out_dir="test_out/out-check-judge-only-${weight}" \
            --data_type='text' --judge_mode='judge_op' \
            --dataset='addition_judge' --train_data_path="train_3digit_judge_${weight}_${eval}.txt" \
            --eval_judge=True --judge_start="FILE:data/addition_judge/test_3digit_judge_${weight}_10000.txt"
    done
done

python train.py config/babygpt/train_addition_judge.py \
            --wandb_project='Teach arithmetic' \
            --wandb_run_name="judge-only-2000-L+W2" \
            --ckpt_path_name="ckpt_judge_only_2000_W2.pt" \
            --out_dir="test_out/out-check-judge-only-W2" \
            --data_type='text' --judge_mode='judge_op' \
            --dataset='addition_judge' --train_data_path="train_3digit_judge_W2_2000.txt" \
            --eval_judge=True --judge_start="FILE:data/addition_judge/test_3digit_judge_W2_10000.txt"