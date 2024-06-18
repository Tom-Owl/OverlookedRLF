import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, utils

class T5_SA:
    def __init__(self, 
                 model_name="mrm8488/t5-base-finetuned-imdb-sentiment", 
                 output_dir = "",
                 batch_size=512,
                 load_best = False,
                 output_attentions = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.output_dir = output_dir
        self.load_best = load_best
        self.output_attentions = output_attentions
        if not self.load_best:
            print('Loading pre-trained model: ', self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, output_attentions=True) # Configure model to return attention values
        else:
            if len(self.output_dir) == 0:
                print('error: output_dir!')
                return
            print('Loading best model from: ', self.output_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir, 
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
        
        # Perform sentiment prediction in batches
        results = []
        num_batches = input_ids.size(0) // self.batch_size
        for i in range(num_batches + 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            end = min((i + 1) * self.batch_size, input_ids.size(0))
            if start < end:
                input_ids_batch = input_ids[start:end]
                attention_mask_batch = attention_mask[start:end]
                with torch.no_grad():
                    outputs = self.model.generate(input_ids_batch, attention_mask=attention_mask_batch)
                batch_results = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in outputs]
                results.extend(batch_results)
        predict_labels = []
        for item in results:
            if item == 'positive':
                predict_labels.append(1)
            else: 
                predict_labels.append(0)
        return predict_labels
    
    def get_sentiment_loss(self, text_list, label_list):
        # Convert labels into strings
        label_texts = ['positive' if int(label) == 1 else 'negative' for label in label_list]
        labels = self.tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        # Tokenize the input text
        inputs = self.tokenizer(text_list, 
                                truncation=True, 
                                padding=True, 
                                return_tensors="pt", 
                                max_length=256)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

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
                
                # Flatten outputs and labels for cross entropy function
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                batch_labels = labels_batch.view(-1)
                batch_losses = loss_function(logits, batch_labels)
                # Reshape to original batch size
                batch_losses = batch_losses.view(labels_batch.size(0), -1).mean(dim=1)
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
