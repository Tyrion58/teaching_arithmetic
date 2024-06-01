# Teaching LLMs to do arithmetic tasks

Data description:
In `data` file, we have:

1. `bal` is balanced addition dataset. It can be used to train addition and test addition. 

2. `mixed` is mixed dataset. The expressions are different from `bal`. It can be used to train mixed tasks. 

3. `bal2` is another balanced addition dataset, which is not overlap with `bal` and `addition_judge`. 
4. `addition_judge` is balanced judgment dataset. They can be used to trian judgment models. And test the ability of judgment. 
    - `train_3digit_judge_10000.txt` and `test_3_digit_jugde_10000.txt` are transfered from `bal`. 
    - `train_3digit_judge_2_10000.txt` and `test_3_digit_jugde_2_10000.txt` are transfered from `bal2`. It can be used to fine-tune a model that is pre-trained on `bal`.