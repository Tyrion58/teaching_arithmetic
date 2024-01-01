from tqdm import tqdm
import torch
import math
import numpy as np

def get_encode_decode(meta_path=None, tokenizer='char'):
    import pickle, tiktoken
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if meta_path and tokenizer == 'char':
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif tokenizer:
        print(f"Trying to load tiktoken's openAI {tokenizer} tokenizer")
        enc = tiktoken.get_encoding(f"{tokenizer}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode

def get_abc(expression: str):
    """
    return: a(str), b(str), c(int), operation(str)
    """
    try:
        # 尝试将表达式中的 'a' 和 'b' 转换为整数
        if expression[0] == 'T' or expression[0] == 'F':
            if expression[4] == 'T' or expression[4] == 'F':
                expression = expression[6:]
            else:
                expression = expression[2:]
        if '+' in expression:
            operation = '+'
        [a, b] = expression.split(operation)
        b = b[:-1]
        if operation == '+':
            # 计算和
            c = int(a) + int(b)

        # 返回结果
        return a, b, c, '+'
    except ValueError:
        # 如果转换失败，抛出异常
        raise ValueError("Invalid input. 'a' and 'b' must be integers.")


def get_num_digits(a: str):
    if a == '':
        return 0
    else:
        if '.' in a: # if a contains a decimal point
            return len(a) - 1
        else:
            return len(str(int(a)))
        
        
def numCarryOps(a, b, binary=False):
    def digitSum(n):
        return sum(map(int,str(n)))
    if b == '':
        return 0
    
    if not binary:
        a,b=int(a),int(b)        
        # assert(a >= 0); assert(b >= 0);
        return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)
    else:
        raise NotImplementedError
        #c = int(a,2) + int(b,2)
        #return int((digitSum(a) + digitSum(b) - digitSum(convert_to_binary(c))) )
        
def is_number(s):
    # handle "xey" case (e.g. 1.2e-3) - we do not use this notation in our dataset
    if 'e' in s:
        return False
    elif 'E' in s:
        return False
    elif 'inf' in s or "INF" in s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False
        
# making a function to batch evaluate addition
# NOTE: Teaching arithmetic task is different from others. The condition/prompt is necessary, such like "2+2="

def get_real_c_hat(fake_c_hat, line_start):
    """
    This function is used to extract real c_hat from generated string in different inference model
    """
    Pred = None
    if '$' == line_start: # handle $ prompt $
        c_hat = fake_c_hat.split('$')[0]
        
    else:
        c_hat = fake_c_hat.split('\n')[0]
    # 来处理\n开头的情况
    
    if 'T' == line_start or 'F' == line_start:
        Pred = line_start
        
    if c_hat != '':
        if 'T' == c_hat[-1] or 'F' == c_hat[-1]:
            Pred = c_hat[-1]
            c_hat = c_hat[:-1]
    else:
        return c_hat, Pred
                            
    c_hat2 = c_hat.strip()
    c_hat2 = c_hat2.split('\n')[0]
    

    return c_hat2, Pred


def eval_addition_batch(config, model, ctx, encode, decode, judge = False, num_digit=3):

    model.eval()
    start = config['start']
    device = config['device']
    
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    
    print(f'evaluating addition from: {start}')
    
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            # 除去每一行后面的空白字符，保存为列表，列表的每一个元素是一个算式，如“2+2=”
            lines = [line.rstrip() for line in f]
    
    else:
        raise NotImplementedError("This method is not implemented yet!")
    
    correct = 0
    pred_correct = 0
    #总行数，也是总算式个数
    total = len(lines)
    
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    #注意区别，corrtec和total
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    prompt_dict = {}
    
    for line_idx in tqdm(range(total)):
        #line_idx是所取出算式的index，取出对应行line
        line = lines[line_idx]
        line.strip('\n')
        # 对line这个string做编码
        start_ids = encode(line)
        # 将编码转换为张量，并额外加一个维度，从len(start_ids)变为(1,len(start_ids))
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        # 在character level tokenization时，这个len_x其实就是len(start_ids)。。。
        len_x = len(x[0])
        a,b,c,op = get_abc(line)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b)
        prompt_length = len(start_ids)
        # NOTE: prompt_length != len(line) if we're not using character level tokenization
        input_tuple = (x, len(line), line[0], a, b, c, a_d, b_d, num_carry)
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
                    Pred = None
                    # 取出对应的tuple
                    _, len_x, line_start, a, b, c, a_d, b_d, num_carry = batch[i]
                    c_hat = outcome[len_x:]
                    c_hat2, Pred = get_real_c_hat(c_hat, line_start)
                    
                    if is_number(c_hat2):
                        if '.' in c_hat2:
                            c_hat2 = float(c_hat2)
                        else:
                            c_hat2 = int(c_hat2)
                    else: # c_hat2 is not a number
                        c = str(c)
                        
                    if op in ['+','-','*']:
                        if c == c_hat2:
                            correct+=1
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            if Pred == 'T':
                                pred_correct+=1
                                
                        elif Pred == 'F':
                            pred_correct+=1
                    else:
                        raise NotImplementedError
                    
                    
                    carry_dictionary[f'carry{num_carry}_total']+=1
                    # metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
    if judge:
        pred_accuracy = pred_correct/total*100
        print(f"Judgement accuracy of {total} examples: {pred_correct}/{total} ({pred_accuracy}%)")
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 \
        if carry_dictionary[f'carry{i}_total']!=0 else np.nan for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    model.train()
    if judge:
        return pred_accuracy, accuracy, accuracy_dictionary
    
    return accuracy, accuracy_dictionary