#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:26:34 2022

@author: vanikanjirangat
"""


"""**Training & Testing**"""

import os
import pandas as pd
input_dir = "./generated_data/"

df=pd.read_csv(input_dir+"/train.csv")
df_dev=pd.read_csv(input_dir+"/dev.csv")



import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
#import matplotlib.pyplot as plt
import ast
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

device="cuda"
le = LabelEncoder()



import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModel,AutoConfig,AutoModelForSequenceClassification
class Model:
    def __init__(self,path,f1=0):
        # self.args = args
        self.f1=f1
        print("flag status:(1:sentence pair, 0: single sentence)",f1)
        self.path=path
        self.MAX_LEN=128
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        num_labels=3
        self.config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",num_labels=num_labels)

       
        # if not os.path.isdir(self.opath):
        #     os.makedirs(self.opath)
            
            
    def extract_data(self,name):
        file =self.path+name
        df = pd.read_csv(file)
        #print(df.head())
        df.replace(np.nan,'NIL', inplace=True)
        
        sentences = df.sent.values
        entity=df.word.values
        labels = df.label.values
        span=df.span.values
        doc_id=df.doc_id.values
        
        
        
        return (sentences,entity,labels,doc_id)

    def process_inputs(self,e,sentences,labels):
      #entity=[ast.literal_eval(x) for x in e]
      #triplets=[ast.literal_eval(x) for x in t]
      #triplets_sents=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets]
      if self.f1==1:
        sentences = [self.tokenizer.encode_plus(sent,e[i],add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#sentence-pair
      else:
        sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#single sentence
      
      tags_vals = list(labels)

      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      print(le_name_mapping)
      labels=le.fit_transform(labels)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

      if self.f1==1:
        token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
        token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

        inputs, labels,types = input_ids, labels,token_type_ids
      else:
        inputs,labels=input_ids, labels

      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      if self.f1==1:
        self.types=torch.tensor(types).to(torch.int64)
        self.data = TensorDataset(self.inputs, self.types,self.masks, self.labels)
      else:
        self.data = TensorDataset(self.inputs, self.masks, self.labels)

      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=32)

    def process_inputs_test(self,e,sentences,labels,act_ids,batch_size=1):
      #entity=[ast.literal_eval(x) for x in e]
      #triplets=[ast.literal_eval(x) for x in t]
      #triplets_sents=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets]
      if self.f1==1:
        sentences = [self.tokenizer.encode_plus(sent,e[i],add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#sentence-pair
      else:
        sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#single sentence
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)

      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      print(le_name_mapping)
      labels=le.fit_transform(labels)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

      if self.f1==1:
        token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
        token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

        inputs, labels,types = input_ids, labels,token_type_ids
      else:
        inputs,labels=input_ids, labels

      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      self.torch_idx = torch.tensor(sentence_idx).to(torch.int64)
      self.act_ids = torch.tensor(act_ids).to(torch.int64)
        #self.words=torch.tensor(e).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      
      if self.f1==1:
        self.types=torch.tensor(types).to(torch.int64)
        self.data = TensorDataset(self.inputs, self.types,self.masks, self.labels,self.torch_idx,self.act_ids)
      else:
        self.data = TensorDataset(self.inputs, self.masks, self.labels,self.torch_idx,self.act_ids)

      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=batch_size)

      # return (self.inputs,self.labels,self.masks,self.types)
    def train_save_load(self,path_,train=1):
      self.model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",config=self.config)
     
      #self.model = BertForSequenceClassification.from_pretrained(path_, num_labels=3)
      self.model.cuda()
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                                                                                                                     'weight_decay_rate': 0.0}]
      #optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
      optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=2e-5)
      
      WEIGHTS_NAME = "CMED_clinicalsingle10.bin"
      OUTPUT_DIR = "./"
      output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      train_loss_set = []
      epochs = 4
      import time
      start_time = time.time()
      if train==1:
        for _ in trange(epochs, desc="Epoch"):
          # Trainin
          # Set our model to training mode (as opposed to evaluation mode
          self.model.train()
          # Tracking variables
          tr_loss = 0
          nb_tr_examples, nb_tr_steps = 0, 0
          # Train the data for one epoch
          for step, batch in enumerate(self.dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            if self.f1==1:
              b_input_ids, b_token_type,b_input_mask, b_labels = batch
            else:
              b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            # loss = model(b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
            if self.f1==1:
              loss,logits= self.model(b_input_ids, token_type_ids=b_token_type, attention_mask=b_input_mask, labels=b_labels,return_dict=False)
            else:
              loss,logits= self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels,return_dict=False)

            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
          print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("--- %s seconds ---" % (time.time() - start_time)) 
        torch.save(self.model.state_dict(), output_model_file)
      else:
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict) 
    def test(self):
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels = [], []
      # Predict 
      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        if self.f1==1:
          b_input_ids,b_type, b_input_mask, b_labels = batch
        else:
          b_input_ids,b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          if self.f1==1:
            outputs = self.model(b_input_ids, token_type_ids=b_type, attention_mask=b_input_mask)
          else:
            outputs = self.model(b_input_ids, attention_mask=b_input_mask)
        logits=outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        self.predictions.append(logits)
        self.true_labels.append(label_ids)
    def compute(self):
      flat_true_labels1=[]
      # Flatten the predictions and true values 
      flat_predictions = [item for sublist in self.predictions for item in sublist]
      flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
      flat_true_labels = [item for sublist in self.true_labels for item in sublist]
      print('Classification Report')
      labels_test=flat_predictions
      print(classification_report(flat_true_labels,flat_predictions))
      print(confusion_matrix(flat_true_labels,flat_predictions))

    def test_id(self,sents,words,doc_id):
        batch_size=32
        # Put model in evaluation mod
        self.model.eval()
        # Tracking variables 
        self.predictions ,self.true_labels,self.sents,self.actsents = [], [],[],[]
        output_dicts=[]
      
        
        for batch in self.dataloader:
          # Add batch to GPU
          batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          if self.f1==1:
            b_input_ids,b_type, b_input_mask, b_labels = batch
          else:
            b_input_ids,b_input_mask, b_labels = batch
          # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
          with torch.no_grad():
            # Forward pass, calculate logit predictions
            if self.f1==1:
              outputs = self.model(b_input_ids, token_type_ids=b_type, attention_mask=b_input_mask)
            else:
              outputs = self.model(b_input_ids, attention_mask=b_input_mask)
            logits=outputs[0]
            

            for j in range(logits.size(0)):
              probs = F.softmax(logits[j], -1)
              output_dict = {
                  # 'index': batch_size * i + j,
                  'true': b_labels[j].cpu().numpy().tolist(),
                  'pred': logits[j].argmax().item(),
                  'conf': probs.max().item(),
                  'logits': logits[j].cpu().numpy().tolist(),
                  'probs': probs.cpu().numpy().tolist(),
                  'actsent_ids'   : b_ids[j].cpu().numpy().tolist(),
                  'sent_ids'   : b_index[j].cpu().numpy().tolist(),
                  'sents' : sents[b_index[j]],
                  'words': words[b_index[j]],
                  'doc_id': doc_id[b_index[j]]
              }
              output_dicts.append(output_dict)
        y_true = [output_dict['true'] for output_dict in output_dicts]
        y_pred = [output_dict['pred'] for output_dict in output_dicts]
        y_conf = [output_dict['conf'] for output_dict in output_dicts]

        accuracy = accuracy_score(y_true, y_pred) * 100.
        fscore = f1_score(y_true, y_pred, average='macro') * 100.
        confidence = np.mean(y_conf) * 100.

        results_dict = {
            'accuracy': accuracy_score(y_true, y_pred) * 100.,
            'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
            'confidence': np.mean(y_conf) * 100.,
        }
        #print(results_dict)
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        return output_dicts

path="./"
m = Model(path,f1=0)

sentences_train,entity_train,labels_train,doc_id_train=m.extract_data('train.csv')


print(len(sentences_train),len(entity_train),len(labels_train))

m.process_inputs(entity_train,sentences_train,labels_train)


m.train_save_load(path_,train=1)

#m.train_save_load(path_,train=0)



sentences_dev,entity_dev,labels_dev,doc_id_dev=m.extract_data('dev.csv')
act_ids=[]

#process_inputs_test(self,e,sentences,labels,act_ids,batch_size=1)
for i,sent in enumerate(sentences_dev):
  act_ids.append(i)

m.process_inputs_test(entity_dev,sentences_dev,labels_dev,act_ids)

out=m.test_id(sentences_dev,entity_dev,doc_id_dev)




