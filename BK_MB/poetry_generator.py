import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead

import numpy as np
import torch

import re

from tqdm.auto import tqdm
import torch.nn.functional as F

import pyphen
import jellyfish

dic = pyphen.Pyphen(lang='pl')

filter_value = -float("Inf")

wersy_n = 1

def get_next_token(model_g, generated, n_logits=None, temperature=1, top_p=0.85):
    outputs = model_g(generated, labels=generated)
    loss, logits = outputs[:2]
    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
       
    if n_logits:
        f_softmax = F.softmax(logits, dim=-1)
        indeces_sort = torch.argsort(f_softmax)[0, :n_logits]
    
    logits[:, indices_to_remove] = filter_value
    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    if n_logits:
        return next_token, indeces_sort #, logits_gr0_idx
    else:
        return next_token

def get_last_word(prev_word, possible_endings):
    prev_syllab = ''.join(dic.inserted(prev_word).split('-')[-2:])

    simmilarities = []
    possible_endings = np.array(possible_endings)

    for p in possible_endings:
        last_word = re.sub("[^0-9a-zA-Z\s]+", '', p).strip().split(' ')[-1]
        last_syllab = ''.join(dic.inserted(last_word).split('-')[-2:])
        
        min_l = np.min( [len(last_syllab), len(prev_syllab)] )
        
        if min_l > 1:
            prev_syllab_tmp = prev_word[-min_l:].lower()
            last_syllab_tmp = last_word[-min_l:].lower()
            s = jellyfish.levenshtein_distance(prev_syllab_tmp, last_syllab_tmp)
            simmilarities.append(s/min_l)
        else:
            simmilarities.append(10)

    choose_endings = possible_endings[np.min(simmilarities) == np.array(simmilarities)]
    ending = choose_endings[np.random.randint(0,choose_endings.shape[0])]
    
    return ending

def generate_poetry(prompt, tokenizer, model_g, max_tokens=70, words_to_check=100, temperature=1, top_p=0.85):

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    rhymes = []
    wersy_count = 0
    if len(re.findall('\n', prompt)) > 0:
        wersy_count += 1

    for i in tqdm(range(max_tokens), desc = 'tokens', \
                  leave=True, 
                  position=0):
        next_token = get_next_token(model_g, generated, temperature=temperature, top_p=top_p)    
        curr_word = tokenizer.decode(next_token[0])
        
        if '<|startoftext|>' in curr_word or '<|endoftext|>' in curr_word :
            break
        
        #when we reached the end of line
        elif curr_word == '\n' and wersy_count < wersy_n:
            wersy_count += 1
        elif curr_word == '\n' and wersy_count == wersy_n:
            # generate most relevant words to finish sentence (up to 1k)
            output_list = list(generated.squeeze().numpy())
            # cut last 2 lines
            last_2_lines = tokenizer.decode(output_list).split('\n')[-2:]
            
            if len(last_2_lines[1]) < 1:
                wersy_count = 0
                next
                
            # cut last word
            last_2_lines[1] = ' '.join(last_2_lines[1].split(' ')[:-1])
            # get rhyme from first line
            last_word_1st_line = re.sub("[^0-9a-zA-Z\s]+", '', last_2_lines[0]).strip().split(' ')[-1]
            # join lines
            last_2_lines_str = '\n'.join(last_2_lines)+' '
            tmp_generated = torch.tensor(tokenizer.encode(last_2_lines_str)).unsqueeze(0)
            
            next_token, possible_words = get_next_token(model_g, tmp_generated, words_to_check, temperature=temperature, top_p=top_p)    
                    
            possible_endings = []
            # generate given number of last words
            for word_i in tqdm(range(words_to_check), desc = 'looking for rhymes', 
                               leave=True, 
                               position=1):
                tmp_next_token  = torch.cat((tmp_generated, possible_words[word_i].reshape(1,1)), dim=1)
                tmp_curr_word = tokenizer.decode(tmp_next_token[0][-1])
                
                counter = 0
                while ~np.isin(tmp_curr_word, ['\n']) and counter < 9:
                    tmp_next_token = torch.cat((tmp_next_token, get_next_token(model_g, tmp_next_token, temperature=temperature, top_p=top_p) ), dim=1)
                    tmp_curr_word = tokenizer.decode(tmp_next_token[0][-1])
                    counter += 1
                
                if counter < 9:
                    possible_endings.append(tokenizer.decode(tmp_next_token[0]))
            
            ending = get_last_word(last_word_1st_line, possible_endings).strip() # + '\n'
            
            all_except_last_2_lines = '\n'.join(tokenizer.decode(output_list).split('\n')[:-2])
            if len(all_except_last_2_lines) > 0:
                all_except_last_2_lines = all_except_last_2_lines + '\n'
            
            generated = torch.tensor(tokenizer.encode(all_except_last_2_lines+ending)).unsqueeze(0)
            wersy_count = 0
            next
        
        generated = torch.cat((generated, next_token), dim=1)
        
        if i > max_tokens*0.75 and '.' in curr_word:
            break

    output_list = list(generated.squeeze().numpy())
    output_text = tokenizer.decode(output_list)
    output_text = re.sub('<\|[a-z]+\|>','',output_text)

    return generated, output_text