#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:15:13 2022

@author: vanikanjirangat
"""




#PRE-PROCESSING

# """FOR ALL!"""

# root_dir = "/content/gdrive/My Drive/CMED_Harvard_Shared_task//trainingdata_v3/"

# root_dir = "/content/gdrive/My Drive/CMED_Harvard_Shared_task//trainingdata_v3/"
# import os
# input_dir=os.path.join(root_dir,'train')

# input_dir

# root_dir = "/content/gdrive/My Drive/CMED_Harvard_Shared_task//trainingdata_v3/"
# import os
# input_dir1=os.path.join(root_dir,'dev')

# import spacy
# nlp = spacy.load('en_core_web_sm')

# attr=["Certainty","Action","Actor","Temporality","Negation"]
# order = {key: i for i, key in enumerate(attr)}
# print(order)

# import glob
# import re
# #Marking the entities
# import re


# def preprocess(dir):
#   attr=["Certainty","Actor","Action","Temporality","Negation"]
#   order = {key: i for i, key in enumerate(attr)}
#   print(order)
#   items=[]
#   c1=glob.glob(dir+"/*.ann")
#   for c in c1:
#     with open(c) as f1:
#       e_map={}
#       p=c.split(".")[0]
#       m=f1.read()
#       m1=m.split("\n")
#       m2=[k.split("\t") for k in m1]
#       m3=[(k[0],k[1],k[-1]) for k in m2 if len(k)==3]
#       spans=[re.findall('\d+', s[1]) for s in m3]
#       labels=[re.split('\d+', s[1])[0] for s in m3]
#       #m4=[(k[0],labels[i],spans[i],k[-1]) for i,k in enumerate(m3)]
#       #items.append((p,m4))
#       m5=[k[0].replace("T","E") for k in m2 if len(k)==3]
#       e_=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("E"))]
#       for (e1,e2) in e_:
#         e21=e2.split(":")[1]
#         e22 = re.sub(' +', '', e21)
#         e_map[e22]=e1

#       #print(p)
#       #print("emap",e_map)
#       #print(m3)
#       m4=[(k[0],labels[i],spans[i],k[2],e_map[k[0]]) for i,k in enumerate(m3)]
#       #print("@@@",m4)
#       #m4=[(k[0],labels[i],spans[i],k[2],m5[i]) for i,k in enumerate(m3)]
#       # ['A1', 'Certainty E10 Hypothetical'],
#       att=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("A"))]
      
      
#       for m in m4:
#         fg=0
#         #print(item)
#         z=[]
#         z1=[]
#         z2=[0,0,0,0,0]
#         for (a,v) in att:
#           v=v.split(" ")
#           if m[4]==v[1]:
#             z1.append(a)
#             z.append((a,v[0],v[2]))
            
#             fg+=1
          
          
#         p1=[m[0],m[1],m[2],m[3],m[4]]
#         diff=[i for i, j in enumerate(attr) if j not in z1]
#         same=[i for i, j in enumerate(attr) if j in z1]
#         #print(z1)
        
#         if len(z1)>0:
#           a1=[]
#           if len(z1)==5:# all 5 attributes present
#             za=sorted(z, key=lambda d: order[d[1]])
#             #print(za)
#             za1=list(sum(za,()))
#             #print(za1)
#             a=[tuple(p1)+tuple(za1)]
#             items.append((p,a))
#             #print("Yes")
#           else:
#             print("Attributes partially present?")
#             # for i in diff:
#             #   z2.insert(i,(attr[i],"nil","nil"))
#             # for i,j in enumerate(same):
              
#             #   z2.insert(i,z[i])
#             # a=p1+z2
#             # items.append((p,a))
#           #print(p)
          
#         if fg==0:
#           q=tuple(["nil"]*15)
#           #print("no attributes")
#           a=[(m[0],m[1],m[2],m[3],m[4])+q]
#           items.append((p,a))
#         if len(a[0])!=20:
#           print(a)
#       #   #print("a",a)
#       # v=[x[1] for x in items]
#       # for i in v:
#       #   print("v",len(i))


#       # for item in m4:
#       #   fg=0
#       #   #print(item)
#       #   for (a,v) in att:
#       #     v=v.split(" ")
#       #     if item[4]==v[1]:
#       #       #print(v)
#       #       a=[(item[0],item[1],item[2],item[3],item[4],a,v[0],v[2])]
#       #       fg+=1
#       #       #print(p)
#       #       items.append((p,a))
#       #   if fg==0:
#       #     #print("no attributes")
#       #     a=[(item[0],item[1],item[2],item[3],item[4],"nil","nil","nil")]
#       #     items.append((p,a))

#   txtfiles=[]
#   c2=glob.glob(dir+"/*.txt")
#   for c in c2:
#     with open(c) as f1:
#       p=c.split(".")[0]
#       text=f1.read()
#       txtfiles.append((p,text))
#   #Adjusting the path ids
#   textdata=[]
#   ptext=[g[0] for g in txtfiles]
#   print("ptext",len(ptext))
#   for i,item in enumerate(items):
#     k={}
#     #print(i)
#     path=item[0]
    
#     if path in ptext:
#       idx=ptext.index(path)

#     else:
#       print("Path Not Found",path)
#       #print(idx) 
#     #print(idx) 
#     #print(ptext[idx])
#     text=txtfiles[idx][1]
#     textdata.append(text)
#   #PROCESSING
#   print("********Processing Starts******")
#   data=[]
#   #textdata=[]
#   ptext=[g[0] for g in txtfiles]
#   ct=0
#   #print("items",len(items))
#   #print(items[:3])
#   for i,item in enumerate(items):
#     ct+=1
#     k={}
#     #print(i)
#     path=item[0]
#     #print(path)
#     if path in ptext:
#       idx=ptext.index(path)
#     doc_id=path.split("/")[-1]
#     #print(idx) 
#     #print(ptext[idx])
#     text=txtfiles[idx][1]
#     s=text
#     doc = nlp(text)
#     #('T11','Disposition ',['2330', '2343'],'ACE inhibitor','E11','A6','Actor','Physician'),
#     #print(doc_id)
#     #print(item)
#     #print(item[0])
#     #print("x",item[1])
#     for x in item[1]:
#       #('T1', 'Disposition ', ['119', '131'], 'Procardia XL', 'E1', 'A1', 'Certainty', 'Certain', 'A2', 'Actor', 'Physician', 'A3', 'Action', 'Start', 'A4', 'Temporality', 'Past', 'A24', 'Negation', 'NotNegated')
#       #print("item",x)
#       define_words=x[3]
#       span=int(x[2][0]),int(x[2][1])
#       label=x[1]
#       w=x[3]
#       word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
#       attr1=x[6]
#       att_val1=x[7]
#       attr2=x[9]
#       att_val2=x[10]
#       attr3=x[12]
#       att_val3=x[13]
#       attr4=x[15]
#       att_val4=x[16]
#       attr5=x[18]
#       att_val5=x[19]
#       #if str(word)!=define_words:
#         #print((idx,word,define_words))
#       if word is not None:#if spacy detects the word
#         sent = word.sent#get the sentence related to word span
#       else:
#         sent=re.findall(r"([^\n\n]*?%s.*?.[$\n\n])" % define_words,text)#else use a regular expression to get the sent/context related to the span
#       if len(sent)==0:#if no context/ sent is retrieved for the time being set it as the word/entity name itself (**work on this**)
#         sent=w
#       sent=str(sent).replace(w,"@"+w+"@")
#       data.append((sent,w,span,label,doc_id,attr1,att_val1,attr2,att_val2,attr3,att_val3,attr4,att_val4,attr5,att_val5))
     
#   print("********Processing Ends******")
#   print("no. of files processed is %s"%(ct))
#   print("no. of sentences/context extracted is %s"%(len(data)))
#   return data

# input_dir

# k=[(0,1),(3,4),(9,10)]
# g=[]
# for i in k:
#   g.append(i)
# g

# #sorted(g, key=lambda d: order[d['value']])

# g1=[12,13,14]

# g2=[tuple(g1)+tuple(g)]
# g2

# m1=[]
# m1.append(("a",g2))
# m1

# train_data=preprocess(input_dir)

# dev_data=preprocess(input_dir1)

# import pandas as pd

# import pandas as pd
# #df = pd.DataFrame(train_data,columns=["sent","word","span","label","doc_id","attribute","att_val"])

# train_data[2]

# df = pd.DataFrame(train_data,columns=["sent","word","span","label","doc_id","attr1","att_val1","attr2","att_val2","attr3","att_val3","attr4","att_val4","attr5","att_val5"])

# import pandas as pd
# df_dev = pd.DataFrame(dev_data,columns=["sent","word","span","label","doc_id","attr1","att_val1","attr2","att_val2","attr3","att_val3","attr4","att_val4","attr5","att_val5"])

# df_dev.head()

# s=df["sent"].values

# import pandas as pd

# input_dir2 = "/content/gdrive/My Drive/CMED_Harvard_Shared_task//generated_data/"

# #df.to_csv(input_dir2+"/train_multi_attribute.csv")

# #df_dev.to_csv(input_dir2+"/dev_multi_attribute.csv")

# df=pd.read_csv(input_dir2+"/train_multi_attribute.csv")

# df_dev=pd.read_csv(input_dir2+"/dev_multi_attribute.csv")

"""

**Training & Testing**
"""

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel,AutoConfig,AutoModelForSequenceClassification



input_dir = "./generated_data/"
device="cuda"
le = LabelEncoder()
# df=pd.read_csv(input_dir+"/train_multi_attribute.csv.csv")
# df_dev=pd.read_csv(input_dir+"/dev_multi_attribute.csv.csv")

# df=pd.read_csv(input_dir2+"/train_multi_attribute.csv")




# print(len(df_disp))

# print(df_disp["attr1"].unique())
# print(df_disp["attr2"].unique())
# print(df_disp["attr3"].unique())
# print(df_disp["attr4"].unique())
# print(df_disp["attr5"].unique())

# print(df_disp["att_val1"].unique())
# print(df_disp["att_val2"].unique())
# print(df_disp["att_val3"].unique())
# print(df_disp["att_val4"].unique())
# print(df_disp["att_val5"].unique())

#df=df_disp

# #print(df.head())
# df.replace(np.nan,'NIL', inplace=True)

# sentences = df.sent.values
# entity=df.word.values
# labels = df.label.values
# span=df.span.values
# doc_id=df.doc_id.values
# certain=df.att_val1.values
# actor=df.att_val2.values
# action=df.att_val3.values
# temporal=df.att_val4.values
# neg=df.att_val5.values




# print(len(df_dev))
# df_disp_dev=df_dev[df_dev.label=='Disposition ']
# print(len(df_disp_dev))
# df_dev=df_disp_dev

# print(df_disp["att_val1"].unique())
# print(df_disp["att_val2"].unique())
# print(df_disp["att_val3"].unique())
# print(df_disp["att_val4"].unique())
# print(df_disp["att_val5"].unique())

# print(df_disp_dev["att_val1"].unique())
# print(df_disp_dev["att_val2"].unique())
# print(df_disp_dev["att_val3"].unique())
# print(df_disp_dev["att_val4"].unique())
# print(df_disp_dev["att_val5"].unique())

# a1=df_disp["att_val1"].unique()
# a2=df_disp["att_val2"].unique()
# a3=df_disp["att_val3"].unique()
# a4=df_disp["att_val4"].unique()
# a5=df_disp["att_val5"].unique()

# b1=df_disp_dev["att_val1"].unique()
# b2=df_disp_dev["att_val2"].unique()
# b3=df_disp_dev["att_val3"].unique()
# b4=df_disp_dev["att_val4"].unique()
# b5=df_disp_dev["att_val5"].unique()

# c1=list(set(a1)-set(b1))
# c2=list(set(a2)-set(b2))
# c3=list(set(a3)-set(b3))
# c4=list(set(a4)-set(b4))
# c5=list(set(a5)-set(b5))

# print(c1)
# print(c2)
# print(c3)
# print(c4)
# print(c5)

# len(df)





#######################################
### --------- Setup BERT ---------- ###


from transformers import AutoTokenizer, AutoModel,AutoConfig,AutoModelForSequenceClassification
class Model:
    def __init__(self,path,f1=0):
        # self.args = args
        self.f1=f1
        print("flag status:(1:sentence pair, 0: single sentence)",f1)
        self.path=path
        self.MAX_LEN=128
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        num_labels=7
    
        self.config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",num_labels=num_labels)

       
        # if not os.path.isdir(self.opath):
        #     os.makedirs(self.opath)
            
            
    def extract_data(self,name):
        file =self.path+name
        df = pd.read_csv(file)
        #print(df.head())
        df.replace(np.nan,'NIL', inplace=True)
        print(df.columns)
        df=df[["sent","word","label","span","doc_id","attr3","att_val3"]]
        #df=df[["sent","word","label","span","doc_id","attr2","att_val2"]]
        #df=df[["sent","word","label","span","doc_id","attr5","att_val5"]]
        #df=df[["sent","word","label","span","doc_id","attr4","att_val4"]]
        #df=df[["sent","word","label","span","doc_id","attr1","att_val1"]]
        df=df[df.label=='Disposition ']
        
        sentences = df.sent.values
        entity=df.word.values
        labels = df.att_val3.values
        
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

    def process_inputs_test(self,e,sentences,labels1,labels2,act_ids,batch_size=1):
      #entity=[ast.literal_eval(x) for x in e]
      #triplets=[ast.literal_eval(x) for x in t]
      #triplets_sents=[('_'.join(x[0:2]),'_'.join(x[2:4])) if len(x)==4 else ('_'.join(x[0:2]),'_'.join(x[1:3])) for x in triplets]
      if self.f1==1:
        sentences = [self.tokenizer.encode_plus(sent,e[i],add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#sentence-pair
      else:
        sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]#single sentence
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      
         
     
      tags_vals = list(labels2)
      
          
      le.fit(labels1)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      print(le_name_mapping)
      
      labels=le.transform(labels2)
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
    def train_save_load(self,train=1):
      
      self.model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",config=self.config)
     
      #self.model = BertForSequenceClassification.from_pretrained(path_, num_labels=3)
      self.model.cuda()
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                                                                                                                     'weight_decay_rate': 0.0}]
      #optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
      optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=2e-5)
      
      WEIGHTS_NAME = "CMED_action.bin"
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
          #print(len(batch))
          #print(batch)
          batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          if self.f1==1:
            b_input_ids,b_type, b_input_mask, b_labels,b_index,b_ids = batch
          else:
            b_input_ids,b_input_mask, b_labels,b_index,b_ids  = batch
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

input_dir

path=input_dir
m = Model(path,f1=0)

sentences_train,entity_train,labels_train,doc_id_train=m.extract_data('train_multi_attribute.csv')



print(len(sentences_train),len(entity_train),len(labels_train))



m.process_inputs(entity_train,sentences_train,labels_train)

#path_=os.path.join(root_dir1,'biobert_T')

#Train the model from scratch
m.train_save_load(train=1)

#m.train_save_load(train=0)

#m.train_save_load(path_,train=1)#single sentence

#to load the model directly set train!=1 else train=1
#m.train_save_load(path_,train=0)

sentences_dev,entity_dev,labels_dev,doc_id_dev=m.extract_data('dev_multi_attribute.csv')
act_ids=[]

#process_inputs_test(self,e,sentences,labels,act_ids,batch_size=1)
for i,sent in enumerate(sentences_dev):
  act_ids.append(i)

m.process_inputs_test(entity_dev,sentences_dev,labels_train,labels_dev,act_ids)

out=m.test_id(sentences_dev,entity_dev,doc_id_dev)

