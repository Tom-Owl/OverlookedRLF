
import argparse
import pandas as pd
import sys 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from datasets import load_dataset, Dataset
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
from sklearn.model_selection import KFold

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1
################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
################################################################################
# TrainingArguments parameters
################################################################################
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 1
# Batch size per GPU for evaluation
per_device_eval_batch_size = 1
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 1500
# Log every X updates steps
logging_steps = 1500
################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
################################################################################
model_name = "meta-llama/Llama-2-13b-chat-hf"
num_train_epochs = 5
   
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
device_map = {"": 0}

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

def Instruct_LLaMA2(instruct_train_list, output_dir):
    instruct_train_df = pd.DataFrame()
    instruct_train_df['text'] = instruct_train_list
    dataset = Dataset.from_pandas(instruct_train_df)
 
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    trainer.train()
    # Save trained model
    #trainer.model.save_pretrained(new_model)
    
if __name__ == '__main__':
    def gen_w_imp_prompt(rlf_sent, word_list, imp_score_list):
        w_imp_res = []
        if len(word_list) == len(imp_score_list):
            for i in range(len(word_list)):
                w_imp_res.append((word_list[i], int(imp_score_list[i])))
        w_imp_res_str = str(w_imp_res)
                
        prompt = '''<s>[INST]
        1. Decompose the sentence into words, preserving  punctuation (for example, 'love!!!!', 'good.').
        2. Assign sentiment important scores (1-5) based on conveyed sentiment.
        3. Organize as list of tuple (word, score) keeping word order.
        4. The expected output format is like [(w0, s0), (w1, s1)....]

        Note: Directly and only return expected output as [(w0, s0), (w1, s1)....].
        Now Input:''' + \
        ''' {0} \n[/INST] {1}'''.format(rlf_sent, w_imp_res_str)
        return prompt

    def gen_sa_prompt(rlf_sent, rlf_sent_label):
        prompt = '''<s>[INST]
        Given the input sentence a sentiment label(1: positive, 0:negative):

        Note: Directly and only return 1 or 0
        Now Input:''' + \
        '''{0}\n[/INST]{1}'''.format(rlf_sent, rlf_sent_label)
        return prompt

    eval_df = pd.read_csv('../dataset/sample_data.csv')
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Set a random state for 
    for fold, (train_index, test_index) in enumerate(kf.split(eval_df)):
        train_val_df = eval_df.iloc[train_index]
        test_df = eval_df.iloc[test_index]
        cut_index = int(train_val_df.shape[0] * 0.8) # we use 1600 in the paper when sample size is 3000
        train_df = train_val_df[:cut_index]
        val_df = train_val_df[cut_index:]
        
        instruct_dataset_list = []
        for i in range(train_df.shape[0]):
            row = train_df.iloc[i]
            rlf_sent = row['rlf_sent']
            word_list = row['word_list'].strip('[]').replace("'", "").split()
            imp_score_list = row['imp_score'].strip('[]').replace("'", "").split()
            rlf_sent_label = int(row['label'])
            
            w_imp_prompt = gen_w_imp_prompt(rlf_sent, word_list, imp_score_list)
            sa_prompt = gen_sa_prompt(rlf_sent, rlf_sent_label)
            
            instruct_dataset_list.append(w_imp_prompt)
            instruct_dataset_list.append(sa_prompt)
        output_dir = '../ft_model/folder_{}/'.format(fold)
        Instruct_LLaMA2(instruct_dataset_list, output_dir)
    
    