for num in 2000 4000 6000 8000 10000
do
for error in 1 2
do
python train.py config/babygpt/train_addition_from_mistakes.py \
        --wandb_project='Teach arithmetic' \
        --init_from='resume' --resume_from='test_out/out-check-add-only-eval/ckpt_10000_acc.pt' \
        --wandb_run_name="judge-from-mistakes-eval-L+W-${num}" \
        --ckpt_path_name="ckpt_judge_from_mistakes_L+W${error}_${num}.pt" \
        --out_dir="test_out/out-check-mixed-eval-L+W${error}"\
        --data_type='text' --data_format='eval_format' --label_exp=False \
        --dataset='answer' --train_data_path="answer_${num}_W_${error}.txt" \
        --eval_addition=True --start="FILE:data/bal/z_3digit_10000.txt" \
        --judge_start="FILE:data/addition_judge/train_3digit_judge_W${error}_10000.txt"
done
done