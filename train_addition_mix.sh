for num in 2000 4000 6000 8000 10000
do

python train.py config/babygpt/train_addition_mixed.py \
        --wandb_project='Teach arithmetic' \
        --wandb_run_name="mixed-${num}-eval-L+W1" \
        --ckpt_path_name="ckpt_mixed_L+W1_${num}.pt" \
        --out_dir='test_out/out-check-mixed-eval-L+W1' \
        --data_type='text' --data_format='eval_format' --label_exp=False \
        --dataset='mixed' --train_data_path="train_3digit_mixed_W1_${num}.txt" \
        --eval_addition=True --start="FILE:data/mixed/test_3digit_mixed_W1_10000.txt" \
        --judge_start="FILE:data/mixed/test_3digit_mixed_W1_10000.txt"
done

for num in 2000 4000 6000 8000 10000
do

python train.py config/babygpt/train_addition_mixed.py \
        --wandb_project='Teach arithmetic' \
        --wandb_run_name="mixed-${num}-eval-L+W2" \
        --ckpt_path_name="ckpt_mixed_L+W2_${num}.pt" \
        --out_dir='test_out/out-check-mixed-eval-L+W2' \
        --data_type='text' --data_format='eval_format' --label_exp=False \
        --dataset='mixed' --train_data_path="train_3digit_mixed_W2_${num}.txt" \
        --eval_addition=True --start="FILE:data/mixed/test_3digit_mixed_W2_10000.txt" 
done