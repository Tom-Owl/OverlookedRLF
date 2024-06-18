import sys 
import string
import pandas as pd
import enchant
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import torch.nn as nn
pos_tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")
pos_model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")
device_index = 1
device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
pipeline = TokenClassificationPipeline(model=pos_model, tokenizer=pos_tokenizer, device=device)

d = enchant.Dict("en_US")

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text):
    """
    From https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    
    return sentences

def merge_short_items(input_list):
    # Initialize an empty list to hold the output
    output_list = []
    # Iterate through the input list
    for item in input_list:
        # Check if the current item's length is smaller than 5 or if it's mergeable punctuation,
        # and also ensure the output list is not empty before merging
        if (len(item) < 5 or item in {'.', '!', '?', ','}) and output_list:
            if item in {'.', '!', '?', ','}:
                output_list[-1] += item
            else:
                output_list[-1] += ' ' + item
        else:
            output_list.append(item)
    return output_list

def divide_text_to_sentence(text):
    sentences = split_into_sentences(text)
    #sentences = merge_short_strings(sentences)
    sentences = merge_short_items(sentences)
    return sentences

## Locate Lengthning word
def contain_lengthening_word(sentence):
    pattern = r'([a-zA-Z])\1{2,}|[!]{3,}|[?]{3,}|[,]{3,}|[.]{4,}'
    if re.search(pattern, sentence):
        return True
    else:
        return False
    
def check_except_string(s):
    patterns = [
        r'\b\d+\b',  # pure numbers
        r'\b\d{1,3}(?:,\d{3})*\b',  # pure numbers or numbers with commas
        r'https?://[\w./-]+',  # URLs
        r'(https?://[\w./-]+)|(www\.[\w./-]+)',  # URLs
        r'@\w+',  # strings starting with @
        r'\$\d{1,3}(?:,\d{3})*',  # monetary amounts
        r'^[\d\s\n\$]*$', # monetary amounts
        r'^\$?\d+(\.\d{1,2})?$' # monetary amounts
    ]

    for pattern in patterns:
        if re.search(pattern, s):
            return True

    return False
    
def locate_lengthening_word(sentence):
    len_pattern = r'([a-zA-Z])\1{2,}|[!]{3,}|[?]{3,}|[,]{3,}|[.]{4,}'
    pun_pattern = r'[!]{3,}|[?]{3,}|[,]{3,}|[.]{4,}'
    words = sentence.split(' ')
    matches = []
    for word in words:
        if re.search(len_pattern, word) and not check_except_string(word):
            if re.search(pun_pattern, word): # if punctuation repetitive
                matches.append(word)
            else: # remove punctuation
                matches.append(re.sub(r'[!?.,]', '', word))
    return matches   

## Find Root Word
def generate_combinations(word):
    if not word:
        return ['']
    # Find all consecutive letters
    i = 0
    while i < len(word) and word[i] == word[0]:
        i += 1
    # Generate combinations for the rest of the word
    rest_combinations = generate_combinations(word[i:])
    combinations = []
    # Add combinations with one or two letters
    if i >= 3:
        for rest in rest_combinations:
            combinations.append(word[0] + rest)  # Add combination with the letter once
            combinations.append(word[0] * 2 + rest)  # Add combination with the letter twice
    else:
        # If there are less than 3 consecutive letters, keep the original
        for rest in rest_combinations:
            combinations.append(word[:i] + rest)
    return combinations

def find_root_word(informal_word, dictionary=d):
    clean_word = re.sub(r'[!?.,]', '', informal_word)
    if len(clean_word) == 0:
        return -1
    # Check if the cleaned word is in dictionary
    if d.check(clean_word):
        return clean_word
        
    root_word_list = []
    for combo in generate_combinations(clean_word):
        #print('try: ', combo)
        if dictionary.check(combo):   
            root_word_list.append(combo)  
    return '|&|'.join(root_word_list)

## Find Normalized Form 
def get_norma_form(rlf_w, df_w):
    result = ''
    i, j = 0, 0
    while i < len(rlf_w):
        # If we still have characters left in df_w and characters match
        if j < len(df_w) and rlf_w[i] == df_w[j] and rlf_w[i] not in ['!', '?', '.', ',']:
            result += rlf_w[i]
            j += 1
        # If character in rlf_w is repetitive and not in df_w
        elif rlf_w[i] in ['!', '?', ',']:
            result += rlf_w[i] 
            flag = 0
            # Skip all same consecutive punctuation
            while i+1 < len(rlf_w) and rlf_w[i] == rlf_w[i+1]:
                flag = 1
                i += 1      
            if flag == 1:
                result += '+'
        elif rlf_w[i] in ['.']:
            result += '...'
            flag = 0
            # Skip all same consecutive punctuation
            while i+1 < len(rlf_w) and rlf_w[i] == rlf_w[i+1]:
                flag = 1
                i += 1      
            if flag == 1:
                result += '+'
        elif i > 0 and rlf_w[i] == rlf_w[i-1] and rlf_w[i] not in ['!', '?', '.', ',']:
            result += '+'
            # Skip all repeating characters
            while i+1 < len(rlf_w) and rlf_w[i] == rlf_w[i+1]:
                i += 1
        i += 1      
    return result

## Get Pos Tag
def get_rlf_pos_tag(df_sent, df_w):
    output = pipeline(df_sent)
    df_w = ''.join(char for char in df_w if char.isalpha())
    for item in output:
        if df_w in item['word']:
            return item['entity']
        elif item['word'][-2:] == '@@' and item['word'][:3] in df_w:
            return item['entity']    
    return -1

def process_document(document_text, label):
    if contain_lengthening_word(document_text):
        # cut document into sentences
        data = []
        sent_list = divide_text_to_sentence(document_text)
        for sent in sent_list:
            length_w_list = locate_lengthening_word(sent)
            if len(length_w_list) > 0:
                root_sent = sent
                root_w_list = []
                nf_w_list = []
                for w in length_w_list:
                    root_w = find_root_word(w)
                    root_w_list.append(root_w)
                    root_sent = root_sent.replace(w, root_w)
                    nf_w = get_norma_form(w, root_w)
                    nf_w_list.append(nf_w)
                for i in range(len(length_w_list)):
                    w = length_w_list[i]
                    root_w = root_w_list[i]
                    nf_w = nf_w_list[i]
                    pos_tag = get_rlf_pos_tag(root_sent, root_w)
                    # sentence, lengthening_word, root_word, normalized_form, pos_tag, label
                    data.append([sent, w, root_w, nf_w, pos_tag, label])
    return data   
if __name__ == '__main__':
    document_text = 'I loooooove pizza soooo much! It tast good!!!!'
    label = 1
    data = process_document(document_text, label)
    print('sentence, RLF_word, root_word, normalized_form, pos_tag, label')
    print(data)