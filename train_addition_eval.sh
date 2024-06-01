for num in 4000 6000 8000 10000
do

python train.py config/babygpt/train_addition_bal.py \
        --wandb_project='Teach arithmetic' \
        --wandb_run_name="add-only-${num}-eval" \
        --ckpt_path_name="ckpt_${num}.pt" \
        --out_dir='test_out/out-check-add-only-eval' \
        --data_type='text' --data_format='eval_format' --reverse_c=False \
        --dataset='bal' --train_data_path="train_3digit_${num}.txt" \
        --eval_addition=True --start="FILE:data/bal/test_3digit_10000.txt" 
done
