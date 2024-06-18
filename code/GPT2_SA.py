import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, utils

'''
print(torch.cuda.get_device_name(0))
try:
    print(torch.cuda.get_device_name(1))
    print(torch.cuda.get_device_name(2))
except:
    print('great!')
'''
    
class GPT2_SA:
    # "marc-er/gpt2-sentiment-classifier-dpo"
    def __init__(self, 
                 model_name="michelecafagna26/gpt2-medium-finetuned-sst2-sentiment", 
                 output_dir = "",
                 batch_size=64,
                 load_best = False,
                 output_attentions = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.load_best = load_best
        self.output_attentions = output_attentions
        if not self.load_best:
            print('Loading pre-trained model: ', self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                            output_attentions=self.output_attentions)
        else:
            if len(self.output_dir) == 0:
                print('error: output_dir!')
                return
            print('Loading best model from: ', self.output_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir,
                                                                            output_attentions=self.output_attentions) 

        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_sentiment(self, text_list):
        # Tokenize the input text
        inputs = self.tokenizer(text_list, 
                                truncation=True,
                                padding=True, 
                                return_tensors="pt",
                                max_length=256)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Perform sentiment analysis in batches
        results = []
        num_batches = input_ids.size(0) // self.batch_size
        for i in range(num_batches + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, input_ids.size(0))
            if start < end:
                input_ids_batch = input_ids[start:end]
                attention_mask_batch = attention_mask[start:end]
                with torch.no_grad():
                    outputs = self.model(input_ids_batch, attention_mask=attention_mask_batch)
                batch_results = torch.argmax(outputs.logits, dim=1).tolist()
                results.extend(batch_results)
        return results
    
    def get_sentiment_loss(self, text_list, label_list):
        # Tokenize the input text
        inputs = self.tokenizer(text_list, 
                                truncation=True, 
                                padding=True, 
                                return_tensors="pt", 
                                max_length=256)
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels = torch.tensor(label_list).to(self.device, dtype = torch.long)
        # Define the loss function
        loss_function = nn.CrossEntropyLoss(reduction='none')  # 'none' to get loss for each sample

        # Perform sentiment analysis in batches
        results = []
        num_batches = input_ids.size(0) // self.batch_size
        for i in range(num_batches + 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            labels_batch = labels[start:end] 
            with torch.no_grad():
                outputs = self.model(input_ids_batch, 
                                     attention_mask=attention_mask_batch, 
                                     labels=labels_batch)
                
                batch_losses = loss_function(outputs.logits, labels_batch)
            results.extend(batch_losses.tolist())
        return results
    
    def norm_imp_score(self, imp_score):
        min_v = min(imp_score)
        max_v = max(imp_score)
        if max_v != min_v:
            imp_score = [(item-min_v)/(max_v-min_v) for item in imp_score]
        else:
            imp_score = [1/len(imp_score)] * len(imp_score)
        imp_score = [item/sum(imp_score) for item in imp_score]
        return imp_score
    
    def get_text_list_w_imp(self, text_list, label_list):
        all_w_imp_list = []
        all_word_list = []
        for i in range(len(text_list)):
            sent = text_list[i]
            label = label_list[i]
            w_imp_sent_list = [sent]
            w_imp_label_list = [label]
            word_list = sent.split()
            if len(word_list) > 1:
                w_imp_label_list += [label] * len(word_list)
                for i in range(len(word_list)):
                    sent_tmp = ' '.join(word_list[:i] + word_list[i+1:])
                    w_imp_sent_list.append(sent_tmp)
            loss_res = self.get_sentiment_loss(w_imp_sent_list, w_imp_label_list)
            imp_score = [abs(loss_i - loss_res[0]) for loss_i in loss_res[1:]]
            all_w_imp_list.append(self.norm_imp_score(imp_score))
            all_word_list.append(word_list)
        return all_word_list, all_w_imp_list 