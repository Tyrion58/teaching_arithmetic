from tqdm import tqdm
import torch
import math
import numpy as np
import random
import string
import tiktoken
import pickle
import os


def reverse_string(a: str) -> str:
    return a[::-1]


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
        # 先去除首尾的空格与'\n'
        expression = expression.strip()
        
        if '+' in expression:
            operation = '+'
            
        [a, b] = expression.split(operation)
         # 对于有label数据
        if a[0] == 'T' or a[0] == 'F':
            # 如果开头有label，去除label
            a = a[1:].strip()
            
        if a[0] == '$':
            a = a[1:].strip()
       
        b = b.split('=')[0].strip()
        a = a.strip()
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
    
    if c_hat != '':
        if 'T' == c_hat[-1] or 'F' == c_hat[-1]:
            Pred = c_hat[-1]
            c_hat = c_hat[:-1]
    else:
        return c_hat, Pred
                            
    c_hat2 = c_hat.split('?')[0].strip()
    c_hat2 = c_hat2.split('\n')[0]
    

    return c_hat2, Pred


def eval_addition_batch(config, model, ctx, encode, decode, judge=False, reverse_c=False, num_digit=3, operator='+', data_format='plain', verbose=False):
    model.eval()
    start = config['start']
    device = config['device']
    
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    
    print(f'evaluating addition from: {start}')
    
    if start.startswith('FILE:'):
        test_data_file = start[5:]
        print(f"Evaluating Addition using test data file: {test_data_file}")
        # we know test examples are test.txt
        test_data_list = get_data_list(test_data_file, operator=operator, judge=judge, test=True)
        test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=True, judge=judge)
      
        lines = test_data_str.split('\n')[:-1]
        for i, line in enumerate(lines):
            # 去除所有judge line
            if line.startswith('j'):
                lines.pop(i)
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
                    
                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)
                    
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
                            # if judge and (Pred == 'T'):
                            #    pred_correct+=1
                        else:
                            print('outputs(x): ', outcome)
                            print(f'wrong  : {a}{op}{b}={c_hat2}')
                            print(f'correct: {a}{op}{b}={c}')
                                
                            # if judge and (Pred == 'F'):
                            #    pred_correct+=1
                    else:
                        raise NotImplementedError
                    
                    
                    carry_dictionary[f'carry{num_carry}_total']+=1
                    # metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
    # if judge:
    #    pred_accuracy = pred_correct/total*100
    #    print(f"Judgement accuracy of {total} examples: {pred_correct}/{total} ({pred_accuracy}%)")
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 \
        if carry_dictionary[f'carry{i}_total']!=0 else np.nan for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    model.train()
    # if judge:
        # return pred_accuracy, accuracy, accuracy_dictionary
    
    return accuracy, accuracy_dictionary


# adding functions to streamline data loading/generation
# get data from .txt file -> outputs list of tuples (x1, x2, y, operator) or (x, y, operator)
def get_data_list(filename=None, operator='+', delim=None, judge=False, test=False):
    import re
    data_list = []
    label = None
    if filename: # read data from file
        if operator in ['text']:
            with open(filename, 'r') as f:
                data = f.read()
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if judge:
                    if line[0] in ['T', 'F']:
                        label = line[0]
                        line = line.split(label)[1].strip('?')
                    elif line.startswith('j'):
                        label = line[-2]
                        pattern = r"\d+"
                        numbers = re.findall(pattern, line)
                        numbers = [int(number) for number in numbers]
                        x1, x2, y2 = numbers
                        data_list.append((int(x1), int(x2), int(y2), label, 'judge'))
                        continue
                    else:
                        raise ValueError('Can not recognize this label!')
                    
                    
                # if first char is $, assume it's a delimiter
                if line[0] == '$':
                    delim = '$'
                if delim:
                    # remove delim from line
                    line = line.replace(delim, '')
                # x1, x2 = line.strip().split(operator)
                if operator in ['+', '-', '*']:
                    x1, x2 = re.split(r'[+\-\*]', line.strip())
                    x2, y2 = x2.split("=")
                    y2 = y2.strip()
                    if test:
                        if operator == '+':
                            y2 = int(x1) + int(x2)
                        elif operator == '-':
                            y2 = int(x1) - int(x2)
                        elif operator == '*':
                            y2 = int(x1) * int(x2)
                    
                    data_list.append((int(x1), int(x2), int(y2), label, operator))


    else: # generate random data
        if operator in ['text']:
            # TODO: For now for creating validation dataset, we just use the last 10% of the shakespeare dataset
            with open('data/shakespeare/input.txt', 'r') as f:
                data = f.read()
                n_text = len(data)
                data = data[int(n_text*0.9):]
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            for _ in range(1000):
                if operator in ['+', '-', '*']:
                    x1, x2 = random.randint(0, 999), random.randint(0, 999)
                    if operator == '+':
                        y = x1 + x2
                    elif operator == '-':
                        y = x1 - x2
                    elif operator == '*':
                        y = x1 * x2
                    data_list.append((int(x1), int(x2), int(y), label, operator))

    return data_list

# creating a script to take in a list of tuples [(x1, x2, y)] and output a string of the form "x1 x2 y\n"
# this will be used to generate the data for our TF model
def generate_data_str(data_list, operator='+', format='plain', train=True, shuffle=True, judge=False):
    if shuffle:
        random.shuffle(data_list)
        
    # for idx, (x1, x2, y) in enumerate(data_list):
    for idx, data_tuple in enumerate(data_list):
        operator = data_tuple[-1]
        if operator in ['+', '-', '*']:   
            x1, x2, y, label = data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3] 
            if train:
            # create training data (x1+x2=y)
                if format == 'plain':
                    if judge:
                        output_str = f"{label}{x1}{operator}{x2}={y}?{label}\n"
                    else:
                        output_str = f"{x1}{operator}{x2}={y}\n"
                elif format == 'reverse':
                    if judge:
                        output_str = f"{label}${x1}{operator}{x2}={str(y)[::-1]}?${label}\n"
                    else:
                        output_str = f"${x1}{operator}{x2}={str(y)[::-1]}$\n"
            else:
                # create test data (x1+x2=)
                if format == 'plain':
                    if judge:
                        output_str = f"T{x1}{operator}{x2}=\n"
                    else:
                        output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'reverse':
                    if judge:
                        output_str = f"T${x1}{operator}{x2}=\n"
                    else:
                        output_str = f"${x1}{operator}{x2}=\n"
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
        
        elif operator in ['judge'] and judge:
            x1, x2, y, label = data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3]
            if train:
                if format == 'plain':
                    output_str = f"j({x1}{operator}{x2}={y})~{label}\n"
                elif format == 'reverse':
                    output_str = f"j({x1}{operator}{x2}={str(y)[::-1]})~{label}\n"
                else:
                    raise ValueError('Format must be plain or reverse!')
            else:
                # test case
                if format == 'plain':
                    output_str = f"j({x1}{operator}{x2}={y})~\n"
                elif format == 'reverse':
                    output_str = f"j({x1}{operator}{x2}={str(y)[::-1]})~\n"
                else:
                    raise ValueError('Format must be plain or reverse!')
                    
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
        else:
            raise ValueError('Illegal operator!')

    return data_str


# create and save meta file for a given vocabulary
def create_meta_file(vocabulary, input_data_str=None, tokenizer='char'):
    operators_str = string.punctuation
    if vocabulary == 'custom_input_data' and input_data_str:
        print(f"Input file {input_data_str[:100]} specified. Reading data from file...")
        data = input_data_str
        print(f"length of dataset in characters: {len(data):,}")
        vocabulary = 'custom_input_data'
    elif vocabulary == 'numbers_only':
        print(f"Creating meta file for numbers only...")
        data = string.digits + operators_str + ' \n'
    elif vocabulary == 'all_ascii_chars':
        print(f"Creating meta file for all reasonable characters...")
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'
    else:
        raise ValueError(f"Vocabulary {vocabulary} not supported!")
    
    if tokenizer == 'char':
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            print(f"data has {len(data_ids):,} tokens")
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
        def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # data_ids = data_encoder(data)
        # print(f"data has {len(data_ids):,} tokens")
        # # convert to np array for efficiency
        # data_ids = np.array(data_ids, dtype=np.uint16)

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        meta_path = f'meta_{vocabulary}.pkl'

    elif tokenizer == 'gpt2':
        print("Ignore all above messages about the meta file!!!")
        print(f"Tokenizer specified as {tokenizer}. Loading it from tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        # karpathy uses enc.encode_ordinary(), but since there is no decode_ordinary(), I'm switching to .encode()
        def data_encoder(s):
            data_ids = enc.encode(s, allowed_special={"<|endoftext|>"}) # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids

        def data_decoder(l):
            return enc.decode(l) # decoder: take a list of integers, output a string


        meta = {
            'vocab_size': enc.n_vocab,
        }
        meta_path = f'meta_pretrained_gpt2_tokenizer.pkl'

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return meta, meta_path, data_encoder, data_decoder


def get_results_dir(config):
    results_dir = config['out_dir']+'/'
    # results_dir += config['dataset']+'_'
    if config['exp_name'] == 'default_exp_name':
        config['exp_name'] = config['wandb_run_name']
        
    results_dir += config['exp_name']

    if os.path.exists(results_dir):
        print(f"WARNING: results directory {results_dir} already exists, overwriting...")
        id = 1
        while os.path.exists(results_dir+'_'+str(id)):
            id += 1
        results_dir += '_'+str(id)
    
    os.makedirs(results_dir, exist_ok=True)

    return results_dir