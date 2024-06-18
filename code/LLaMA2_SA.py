from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

class LLaMA2_SA:
    def __init__(self,
                 load_best=False,
                 model_type = '13b',
                 lora_model_path = ''
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_best = load_best
        self.model_type = model_type
        if self.model_type == '13b':
            model_name="meta-llama/Llama-2-13b-chat-hf"
       
        self.model_name = model_name
        self.lora_model_path = lora_model_path
        ## Load fine-tuned model
        if self.load_best:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                low_cpu_mem_usage=True,
                                return_dict=True,
                                torch_dtype=torch.float16,
                                device_map=self.device,
                            )# Load the tokenizer
            
            self.model = PeftModel.from_pretrained(self.base_model, self.lora_model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                low_cpu_mem_usage=True,
                                return_dict=True,
                                torch_dtype=torch.float16,
                                device_map=self.device,
                            )# Load the original model
            
        # Reload tokenizer to save it
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
       
    def get_pipe_prompt_result(self, prompt):
        pipe = pipeline(task="text-generation", 
                    model=self.model, 
                    tokenizer=self.tokenizer, 
                    device_map=self.device,
                    max_length=len(prompt)+10,
                    do_sample=True,
                    top_k=3,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.4,
                    temperature = 0.2)
        result = pipe(prompt)
        return result[0]['generated_text']
    
    def get_pipe_prompt_sentiment(self, prompt):
        pipe = pipeline(task="text-generation", 
                    model=self.model, 
                    tokenizer=self.tokenizer, 
                    device_map=self.device,
                    max_length=len(prompt)+10,
                    max_new_tokens=10,
                    do_sample=True,
                    top_k=3,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.4,
                    temperature = 0.2)
        result = pipe(prompt)
        return result[0]['generated_text']
    
    def generate_sentiment_analyse_prompt(self, rlf_sent):
            prompt_sa = '''<s>[INST]
Given the input sentence a sentiment label(1: positive, 0:negative):

Note: Directly and only return 1 or 0
Now Input:''' + \
'''{0}\n[/INST]'''.format(rlf_sent)
            return prompt_sa
        
    def get_sentiment_label(self, rlf_sent):
        def find_first_number(s):
            match = re.search(r'[-+]?\d*\.?\d+', s)
            if match:
                return float(match.group())
            else:
                return None
        sa_prompt = self.generate_sentiment_analyse_prompt(rlf_sent)
        sa_res = self.get_pipe_prompt_sentiment(sa_prompt)
        #print('res: ', sa_res)
        pred_label = find_first_number(sa_res.split('[/INST]')[1])
        try:
            return int(pred_label)
        except:
            return 1
    
    def generate_word_imp_score_prompt(self, rlf_sent):
        if self.load_best:
            prompt_sa = '''<s>[INST]
1. Decompose the sentence into words, preserving  punctuation (for example, 'love!!!!', 'good.').
2. Assign sentiment important scores (1-5) based on conveyed sentiment.
3. Organize as list of tuple (word, score) keeping word order.
4. The expected output format is like [(w0, s0), (w1, s1)....]

Note: Directly and only return expected output as [(w0, s0), (w1, s1)....].

Now Input:''' + \
'''{0}\n[/INST]'''.format(rlf_sent)

        else:
            prompt_sa = '''<s>[INST]
1. Decompose the sentence into words, preserving  punctuation (for example, 'love!!!!', 'good.').
2. Assign sentiment important scores (1-5) based on conveyed sentiment.
3. Organize as list of tuple (word, score) keeping word order.
4. The expected output format is like [(w0, s0), (w1, s1)....]

There is an examples of outputs:
Input: LLMs is sooo cooolll!!!
Return: [('LLMs', 2), ('is', 1), ('sooo', 5), ('cooolll!!!', 5)]
Input: I haaate Pizza way too much.
Return: [('I', 2), ('haaate', 5), ('Pizza', 1), ('way', 3), ('too', 3), ('much.', 4)]

Note: Directly and only return expected output as [(w0, s0), (w1, s1)....].

Now Input:''' + \
'''{0}\n[/INST]'''.format(rlf_sent)
            
        return prompt_sa
        
    def norm_imp_score(self, imp_score):
        if len(imp_score) == 0:
            return []
        min_v = min(imp_score)
        max_v = max(imp_score)
        if max_v != min_v:
            imp_score = [(item-min_v)/(max_v-min_v) for item in imp_score]
        else:
            imp_score = [1/len(imp_score)] * len(imp_score)
        imp_score = [item/sum(imp_score) for item in imp_score]
        return imp_score
    
    def get_word_imp_score(self, rlf_sent):
        def extract_word_score_pairs(s):
            # Regex pattern to match (word, score) pairs inside the first square brackets
            content = ''
            s_list = s.split('\n')
            for item in s_list:
                if '[' in item and len(item) > 3:
                    content = item 
                    break 
            pattern = r"\(\s*'([A-Za-z0-9.,!?]+)'\s*,\s*(\d)\s*\)"
            pairs = re.findall(pattern, content)
            return pairs
        sa_prompt = self.generate_word_imp_score_prompt(rlf_sent)
        word_imp_score_res = self.get_pipe_prompt_result(sa_prompt)
        extract_res = word_imp_score_res.split('[/INST]')[1].split(':')[-1].replace("‘", "'").replace("’", "'")
        # print(extract_res)
        res_pair = extract_word_score_pairs(extract_res)
        word_list = [item[0] for item in res_pair]
        w_imp_list = [int(item[1]) for item in res_pair]
        w_imp_list = self.norm_imp_score(w_imp_list)
        return word_list, w_imp_list
    
    def get_sentiment(self, text_list):
        results = []
        for sent in text_list:
            results.append(self.get_sentiment_label(sent))
        return results
    
    def get_text_list_w_imp(self, text_list):
        all_w_imp_list = []
        all_word_list = []
        for sent in text_list:
           w_res, w_imp_res = self.get_word_imp_score(sent)
           all_word_list.append(w_res)
           all_w_imp_list.append(w_imp_res)
        return all_word_list, all_w_imp_list 
    
    