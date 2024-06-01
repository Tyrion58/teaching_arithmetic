from main_utils import *
from model import GPTConfig, GPT, JudgeGPT
import torch
from contextlib import nullcontext
import re              


def eval_judge_batch(config, model, ctx, encode, decode, data_format='plain', reverse_c=False, num_digit=3, 
                     max_new_tokens=1, mode='bilabel', verbose=True):
    model.eval()
    start = config['judge_start']
    device = config['device']
    # test_data_file = start[5:]
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    # 设置max_new_tokens为1，因为只需要输出判断结果
    max_new_tokens = max_new_tokens
    
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    
    print(f'evaluating addition from: {start}')
    
    lines = []
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            # 除去每一行后面的空白字符，保存为列表，列表的每一个元素是一个算式，如“2+2=”
            old_lines = [line.rstrip() for line in f] 
            if data_format=='reverse' and reverse_c:
                for line in old_lines:
                    a, b = line.split('=')
                    b = b[:-1]
                    lines.append(a + f'={str(b)[::-1]}?')
            else:
                lines = old_lines
    
    else:
        raise NotImplementedError("This method is not implemented yet!")
    
    pred_correct = 0
    no_judge = 0
    
    TP = 0 # True Positive
    FP = 0 # False Positive
    TN = 0 # True Negative
    FN = 0 # False Positive
    
    #总行数，也是总算式个数
    total = len(lines)
    
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    #注意区别，corrtec和total
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    prompt_dict = {}
    # 创建字典统计错误类型，包括：
    # 1.字符串长度不符 2.第0位错误 3.第1位错误 4.第2位错误 5.第3位错误
    wrong_type_dict = None
    # wrong_type_dict = {f'wrong_{i}':0 for i in range(num_digit+1)}
    # wrong_type_dict['len_not_match'] = 0 
    
    
    for line_idx in tqdm(range(total)):
        #line_idx是所取出算式的index，取出对应行line
        line = lines[line_idx]
        line = line.strip('\n')
   
        if line[0] == 'j':
            label = line.split('~')[-1]
            line = line.split('~')[0]
            pattern = r"\d+"
            numbers = re.findall(pattern, line)
            numbers = [int(number) for number in numbers]
            x1, x2, y2 = numbers
            line = f'{x1}+{x2}={y2}?'
        else:
            label = line[0]
            line = line[1:]
    
        if data_format=='reverse':
            line = '$'+line+'$'
        # 对line这个string做编码
        a,b,c,op = get_abc(line)
        if mode in ['judge_op']:
            line = line.split('?')[0]
            line = f'j{line}~'
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b)
        start_ids = encode(line)
        # 将编码转换为张量，并额外加一个维度，从len(start_ids)变为(1,len(start_ids))
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        # 在character level tokenization时，这个len_x其实就是len(start_ids)。。。
        prompt_length = len(start_ids)
        # NOTE: prompt_length != len(line) if we're not using character level tokenization
        input_tuple = (x, len(line), label, a, b, c, a_d, b_d, num_carry)
        if prompt_length in prompt_dict.keys():
            prompt_dict[prompt_length].append(input_tuple)
        else:
            prompt_dict[prompt_length] = [input_tuple]
        # prompt是一个字典，键值是所有可能出现的prompt_length
        # 这样划分是为了保证每一个batch中的len_x相等
        
    # construct batches of prompts now
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)):
            #每个sequence（或算式）对应一个tuple，每test_batch_size个tuple划分为同一个batch，对应这一个list，
            # 也就是每个list就是一个batch，所有batch组成一个更大的batch_list
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])
            
    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        # 单取出所有x
        x_list = [input_tuple[0] for input_tuple in batch]
        # x.size=(batch_size, )
        x = torch.cat(x_list, dim=0)
        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome_list = [decode(y_i.tolist()) for y_i in y]
                # 下面逐个分析这个batch中的model的预测结果
                for i, outcome in enumerate(outcome_list):
                     # 取出对应的tuple
                    _, len_x, label, a, b, c, a_d, b_d, num_carry = batch[i]
                    Pred = outcome[-1]
                    if label == 'T':
                        if Pred == 'T':
                            pred_correct += 1
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            TP += 1
                        elif Pred == 'F':
                            FN += 1
                            if verbose:
                                print('wrong outputs(x): ', outcome)
                        else: 
                            no_judge += 1
                            if verbose:
                                print('no judging outputs(x): ', outcome)
                            
                    elif label == 'F':
                        if Pred == 'F':
                            pred_correct += 1
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            TN += 1
                        elif Pred == 'T':
                            FP += 1
                            if verbose:
                                print('wrong outputs(x): ', outcome)
                            # update_wrong_type_dict(wrong_type_dict, outcome)
                        else:
                            no_judge += 1
                            if verbose:
                                print('no judging outputs(x): ', outcome)
                    else:
                        no_judge += 1
                        
                    carry_dictionary[f'carry{num_carry}_total']+=1
    
    pred_accuracy = pred_correct/total*100
    no_judging_probability = no_judge/total*100
    
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 \
        if carry_dictionary[f'carry{i}_total']!=0 else np.nan for i in range(num_digit+1)}
    if verbose:
        print(f"Judgement accuracy of {total} examples: {pred_correct}/{total} ({pred_accuracy}%)")
        print(f"No judging probability of {total} examples: {no_judge}/{total} ({no_judging_probability}%)")
        print(f'True Positive Examples: {TP}/{total}')
        print(f'False Positive Examples: {FP}/{total}')
        print(f'True Negative Examples: {TN}/{total}')
        print(f'False Negative Examples: {FN}/{total}')
        print(accuracy_dictionary)
    
    model.train()
    
    return pred_accuracy, no_judging_probability, accuracy_dictionary

class model_tester:
    def __init__(self, model_path, 
                 meta_path='meta_all_ascii_chars.pkl',
                 extra_test_file='./test_data/extra_num_judge_prompt.txt',
                 add_noise_test_file='./test_data/add_noise_judge_prompt.txt',
                 together_test_file='./addition_jugde/test_3digit_jugde_W1_10000.txt',
                 max_new_tokens=1,
                 data_format='plain',
                 reverse_c=False,
                 mode='bilabel',
                 mydevice='mps',
                 model_state=None,
                 vebrose=False,
                 judge_gpt=False) -> None:
        # init from a model saved in a specific directory
        ckpt_path = model_path
        self.max_new_tokens = max_new_tokens
        
        checkpoint = torch.load(ckpt_path, map_location=mydevice)
        gptconf = GPTConfig(**checkpoint['model_args'])
        if judge_gpt:
            self.model = JudgeGPT(gptconf)
        else:
            self.model = GPT(gptconf)
        if model_state is not None:
            state_dict = model_state
        else:
            state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.to(mydevice)
        self.encode, self.decode = get_encode_decode(meta_path)
        self.extra_path = 'FILE:' + extra_test_file
        self.add_noise_path = 'FILE:' + add_noise_test_file
        self.test_path = 'FILE:' + together_test_file
        self.device = mydevice
        self.dataformat = data_format
        self.rev_c = reverse_c
        self.verbose = vebrose
        self.mode = mode
        
    def test_extra(self):
        ctx = nullcontext()
        config={
            'judge_start': self.extra_path,
            'device': self.device,
        }
        pred_accuracy, no_judging_probability, accuracy_dictionary = eval_judge_batch(config, self.model, ctx, self.encode, self.decode, max_new_tokens=self.max_new_tokens, 
                         data_format=self.dataformat, reverse_c=self.rev_c, mode=self.mode, verbose=self.verbose)
        
        return pred_accuracy, no_judging_probability, accuracy_dictionary
        
    def test_add_noise(self):
        ctx = nullcontext()
        config={
            'judge_start': self.add_noise_path,
            'device': self.device,
        }
        pred_accuracy, no_judging_probability, accuracy_dictionary = eval_judge_batch(config, self.model, ctx, self.encode, self.decode, max_new_tokens=self.max_new_tokens, 
                         data_format=self.dataformat, reverse_c=self.rev_c, mode=self.mode, verbose=self.verbose)

        return pred_accuracy, no_judging_probability, accuracy_dictionary
        
    def test_jugdment_together(self):
        ctx = nullcontext()
        config={
            'judge_start': self.test_path,
            'device': self.device,
        }
        pred_accuracy, no_judging_probability, accuracy_dictionary = eval_judge_batch(config, self.model, ctx, self.encode, self.decode, max_new_tokens=self.max_new_tokens,
                            data_format=self.dataformat, reverse_c=self.rev_c, mode=self.mode, verbose=self.verbose)
        
        return pred_accuracy, no_judging_probability, accuracy_dictionary
        
class model_addition_tester:
    def __init__(self, model_path, 
                 meta_path='meta_all_ascii_chars.pkl',
                 test_file='./bal/train_3digit_10000.txt',
                 num_digit=3,
                 data_format='plain',
                 reverse_c=False,
                 operator='+',
                 mydevice='mps',
                 model_state=None,
                 judge=False, 
                 label_exp=False, 
                 vebrose=True) -> None:
        
        self.meta_path = meta_path
        self.test_file = 'FILE:' + test_file
        self.num_digit = num_digit
        self.data_format = data_format
        self.reverse_c = reverse_c
        self.operator = operator
        self.judge = judge
        self.label_exp = label_exp
        self.verbose = vebrose
        self.device = mydevice
        
         # init from a model saved in a specific directory
        ckpt_path = model_path
        checkpoint = torch.load(ckpt_path, map_location=mydevice)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)
        if model_state is not None:
            state_dict = model_state
        else:
            state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.to(mydevice)
        self.encode, self.decode = get_encode_decode(meta_path)
        
    def test_addition(self):
        ctx = nullcontext()
        config={
            'start': self.test_file,
            'device': self.device,
        }
        return eval_addition_batch(config=config, model=self.model, ctx=ctx, encode=self.encode, 
                            decode=self.decode, judge=self.judge, reverse_c=self.reverse_c, num_digit=self.num_digit,
                            data_format=self.data_format, operator=self.operator, label_exp=self.label_exp, verbose=self.verbose)