#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:23:23 2022

@author: vanikanjirangat
"""








"""FOR ALL!"""



"""Sentence level Extraction"""

import glob
import re
#Marking the entities
import re


def preprocess(dir):
  attr=["Certainty","Actor","Action","Temporality","Negation"]
  order = {key: i for i, key in enumerate(attr)}
  print(order)
  items=[]
  c1=glob.glob(dir+"/*.ann")
  for c in c1:
    with open(c) as f1:
      e_map={}
      p=c.split(".")[0]
      m=f1.read()
      m1=m.split("\n")
      m2=[k.split("\t") for k in m1]
      m3=[(k[0],k[1],k[-1]) for k in m2 if len(k)==3]
      spans=[re.findall('\d+', s[1]) for s in m3]
      labels=[re.split('\d+', s[1])[0] for s in m3]
      #m4=[(k[0],labels[i],spans[i],k[-1]) for i,k in enumerate(m3)]
      #items.append((p,m4))
      m5=[k[0].replace("T","E") for k in m2 if len(k)==3]
      e_=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("E"))]
      for (e1,e2) in e_:
        e21=e2.split(":")[1]
        e22 = re.sub(' +', '', e21)
        e_map[e22]=e1

      #print(p)
      #print("emap",e_map)
      #print(m3)
      m4=[(k[0],labels[i],spans[i],k[2],e_map[k[0]]) for i,k in enumerate(m3)]
      #print("@@@",m4)
      #m4=[(k[0],labels[i],spans[i],k[2],m5[i]) for i,k in enumerate(m3)]
      # ['A1', 'Certainty E10 Hypothetical'],
      att=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("A"))]
      
      
      for m in m4:
        fg=0
        #print(item)
        z=[]
        z1=[]
        z2=[0,0,0,0,0]
        for (a,v) in att:
          v=v.split(" ")
          if m[4]==v[1]:
            z1.append(a)
            z.append((a,v[0],v[2]))
            
            fg+=1
          
          
        p1=[m[0],m[1],m[2],m[3],m[4]]
        diff=[i for i, j in enumerate(attr) if j not in z1]
        same=[i for i, j in enumerate(attr) if j in z1]
        #print(z1)
        
        if len(z1)>0:
          a1=[]
          if len(z1)==5:# all 5 attributes present
            za=sorted(z, key=lambda d: order[d[1]])
            #print(za)
            za1=list(sum(za,()))
            #print(za1)
            a=[tuple(p1)+tuple(za1)]
            items.append((p,a))
            #print("Yes")
          else:
            print("Attributes partially present?")
            # for i in diff:
            #   z2.insert(i,(attr[i],"nil","nil"))
            # for i,j in enumerate(same):
              
            #   z2.insert(i,z[i])
            # a=p1+z2
            # items.append((p,a))
          #print(p)
          
        if fg==0:
          q=tuple(["nil"]*15)
          #print("no attributes")
          a=[(m[0],m[1],m[2],m[3],m[4])+q]
          items.append((p,a))
        if len(a[0])!=20:
          print(a)
      #   #print("a",a)
      # v=[x[1] for x in items]
      # for i in v:
      #   print("v",len(i))


      # for item in m4:
      #   fg=0
      #   #print(item)
      #   for (a,v) in att:
      #     v=v.split(" ")
      #     if item[4]==v[1]:
      #       #print(v)
      #       a=[(item[0],item[1],item[2],item[3],item[4],a,v[0],v[2])]
      #       fg+=1
      #       #print(p)
      #       items.append((p,a))
      #   if fg==0:
      #     #print("no attributes")
      #     a=[(item[0],item[1],item[2],item[3],item[4],"nil","nil","nil")]
      #     items.append((p,a))

  txtfiles=[]
  c2=glob.glob(dir+"/*.txt")
  for c in c2:
    with open(c) as f1:
      p=c.split(".")[0]
      text=f1.read()
      txtfiles.append((p,text))
  #Adjusting the path ids
  textdata=[]
  ptext=[g[0] for g in txtfiles]
  print("ptext",len(ptext))
  for i,item in enumerate(items):
    k={}
    #print(i)
    path=item[0]
    
    if path in ptext:
      idx=ptext.index(path)

    else:
      print("Path Not Found",path)
      #print(idx) 
    #print(idx) 
    #print(ptext[idx])
    text=txtfiles[idx][1]
    textdata.append(text)
  #PROCESSING
  print("********Processing Starts******")
  data=[]
  #textdata=[]
  ptext=[g[0] for g in txtfiles]
  ct=0
  #print("items",len(items))
  #print(items[:3])
  for i,item in enumerate(items):
    ct+=1
    k={}
    #print(i)
    path=item[0]
    #print(path)
    if path in ptext:
      idx=ptext.index(path)
    doc_id=path.split("/")[-1]
    #print(idx) 
    #print(ptext[idx])
    text=txtfiles[idx][1]
    s=text
    doc = nlp(text)
    #('T11','Disposition ',['2330', '2343'],'ACE inhibitor','E11','A6','Actor','Physician'),
    #print(doc_id)
    #print(item)
    #print(item[0])
    #print("x",item[1])
    for x in item[1]:
      #('T1', 'Disposition ', ['119', '131'], 'Procardia XL', 'E1', 'A1', 'Certainty', 'Certain', 'A2', 'Actor', 'Physician', 'A3', 'Action', 'Start', 'A4', 'Temporality', 'Past', 'A24', 'Negation', 'NotNegated')
      #print("item",x)
      define_words=x[3]
      span=int(x[2][0]),int(x[2][1])
      label=x[1]
      w=x[3]
      word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
      attr1=x[6]
      att_val1=x[7]
      attr2=x[9]
      att_val2=x[10]
      attr3=x[12]
      att_val3=x[13]
      attr4=x[15]
      att_val4=x[16]
      attr5=x[18]
      att_val5=x[19]
      #if str(word)!=define_words:
        #print((idx,word,define_words))
      if word is not None:#if spacy detects the word
        sent = word.sent#get the sentence related to word span
      else:
        sent=re.findall(r"([^\n\n]*?%s.*?.[$\n\n])" % define_words,text)#else use a regular expression to get the sent/context related to the span
      if len(sent)==0:#if no context/ sent is retrieved for the time being set it as the word/entity name itself (**work on this**)
        sent=w
      sent=str(sent).replace(w,"@"+w+"@")
      data.append((sent,w,span,label,doc_id,attr1,att_val1,attr2,att_val2,attr3,att_val3,attr4,att_val4,attr5,att_val5))
     
  print("********Processing Ends******")
  print("no. of files processed is %s"%(ct))
  print("no. of sentences/context extracted is %s"%(len(data)))
  return data

"""Extracting Section wise content with attribute values"""

'''
If you need to explicitly mark the entities section wise: ues the script below
'''

import glob
import re
#Marking the entities
def preprocess(dir):
  attr=["Certainty","Actor","Action","Temporality","Negation"]
  order = {key: i for i, key in enumerate(attr)}
  print(order)
  items=[]
  c1=glob.glob(dir+"/*.ann")
  for c in c1:
    with open(c) as f1:
      e_map={}
      p=c.split(".")[0]
      m=f1.read()
      m1=m.split("\n")
      m2=[k.split("\t") for k in m1]
      m3=[(k[0],k[1],k[-1]) for k in m2 if len(k)==3]
      spans=[re.findall('\d+', s[1]) for s in m3]
      labels=[re.split('\d+', s[1])[0] for s in m3]
      #m4=[(k[0],labels[i],spans[i],k[-1]) for i,k in enumerate(m3)]
      #items.append((p,m4))
      m5=[k[0].replace("T","E") for k in m2 if len(k)==3]
      e_=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("E"))]
      for (e1,e2) in e_:
        e21=e2.split(":")[1]
        e22 = re.sub(' +', '', e21)
        e_map[e22]=e1

      #print(p)
      #print("emap",e_map)
      #print(m3)
      m4=[(k[0],labels[i],spans[i],k[2],e_map[k[0]]) for i,k in enumerate(m3)]
      #print("@@@",m4)
      #m4=[(k[0],labels[i],spans[i],k[2],m5[i]) for i,k in enumerate(m3)]
      # ['A1', 'Certainty E10 Hypothetical'],
      att=[(k[0],k[1]) for k in m2 if (len(k)==2 and k[0].startswith("A"))]
      
      
      for m in m4:
        fg=0
        #print(item)
        z=[]
        z1=[]
        z2=[0,0,0,0,0]
        for (a,v) in att:
          v=v.split(" ")
          if m[4]==v[1]:
            z1.append(a)
            z.append((a,v[0],v[2]))
            
            fg+=1
          
          
        p1=[m[0],m[1],m[2],m[3],m[4]]
        diff=[i for i, j in enumerate(attr) if j not in z1]
        same=[i for i, j in enumerate(attr) if j in z1]
        #print(z1)
        
        if len(z1)>0:
          a1=[]
          if len(z1)==5:# all 5 attributes present
            za=sorted(z, key=lambda d: order[d[1]])
            #print(za)
            za1=list(sum(za,()))
            #print(za1)
            a=[tuple(p1)+tuple(za1)]
            items.append((p,a))
            #print("Yes")
          else:
            print("Attributes partially present?")
            # for i in diff:
            #   z2.insert(i,(attr[i],"nil","nil"))
            # for i,j in enumerate(same):
              
            #   z2.insert(i,z[i])
            # a=p1+z2
            # items.append((p,a))
          #print(p)
          
        if fg==0:
          q=tuple(["nil"]*15)
          #print("no attributes")
          a=[(m[0],m[1],m[2],m[3],m[4])+q]
          items.append((p,a))
        if len(a[0])!=20:
          print(a)
     

  txtfiles=[]
  c2=glob.glob(dir+"/*.txt")
  for c in c2:
    with open(c) as f1:
      p=c.split(".")[0]
      text=f1.read()
      txtfiles.append((p,text))
  #Adjusting the path ids
  textdata=[]
  ptext=[g[0] for g in txtfiles]
  print("ptext",len(ptext))
  for i,item in enumerate(items):
    k={}
    #print(i)
    path=item[0]
    
    if path in ptext:
      idx=ptext.index(path)

    else:
      print("Path Not Found",path)
      #print(idx) 
    #print(idx) 
    #print(ptext[idx])
    text=txtfiles[idx][1]
    textdata.append(text)
  #PROCESSING
  print("********Processing Starts******")
  data=[]
  #textdata=[]
  ptext=[g[0] for g in txtfiles]
  ct=0
  for i,item in enumerate(items):
    ct+=1
    k={}
    #print(i)
    path=item[0]
    #print(path)
    if path in ptext:
      idx=ptext.index(path)
    doc_id=path.split("/")[-1]
    #print(idx) 
    #print(ptext[idx])
    text=txtfiles[idx][1]
    s=text
    #Getting sections
    start="\n"
    end=":"
    result = re.findall('%s(.*)%s' % (start, end), s)
    #print(result)
    result[:] = [x for x in result if not x.startswith('-')]
    w11=[k[-1] for k in item[1]]
    for j in w11:
      for i in result:
        if j in i:
          #print(i)
          result.remove(i)
    
    doc = nlp(text) # use your raw text here
    n=len(text)-1
    n1=len(text)-25 #heuristic
    result.append(text[n1:n])#including the last part of sentence as final section
    #results-->the various sections in clinical note
    wr=[]
    c={}
    for x in item[1]:
      define_words=x[3]
      span=int(x[2][0]),int(x[2][1])
      label=x[1]
      w=x[3]
      word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
      attr1=x[6]
      att_val1=x[7]
      attr2=x[9]
      att_val2=x[10]
      attr3=x[12]
      att_val3=x[13]
      attr4=x[15]
      att_val4=x[16]
      attr5=x[18]
      att_val5=x[19]
      
      # define_words=x[-1]
      # span=int(x[2][0]),int(x[2][1])
      # label=x[1]
      # w=x[-1]
      # word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
      #if str(word)!=define_words:
        #print((idx,word,define_words))
      if word is not None:#if spacy detects the word
        sent = word.sent#get the sentence related to word span
      else:
        sent=re.findall(r"([^\n\n]*?%s.*?.[$\n\n])" % define_words,text)#else use a regular expression to get the sent/context related to the span
      if len(sent)==0:#if no context/ sent is retrieved for the time being set it as the word/entity name itself (**work on this**)
        sent=w
      if w not in c.keys():#This is to track the position of the entity mention
        #r=0
        c[w]=[w]
        r=len(c[w])-1
      else:
        c[w].append(w)
        r=len(c[w])-1# this length gives the counter of the entity mention, needed if it occurs multiple times
        #print("multiword",(r,w))

    
      para=[]# retrieve the context in terms of sections in which the entity spanned sentence/phrase is present (could be a paragrah)
      fg=0
      for i in range(len(result)-1):#all sections
        start=result[i]
        end=result[i+1]
      
      #result=s[s.find(start):s.rfind(end)]
      #if f==0:
        ctxt=text[text.find(start):text.rfind(end)]# get the context between the given sections
      
        #print(len(para))
        if str(sent) in ctxt:#check if the retrieved sentence is in the section/context
          fg+=1
          #print((start,end))
          #print(p)
          
          sent=str(sent)
          p=p.replace(w,"@"+w+"@")
          p=p.replace(sent,"@"+sent+"@")
          para.append((ctxt,start))#append the context/section of retrived sentence
          #para.append((p,start))
        
      if fg==0:
        sent=str(sent)
        sent=sent.replace(w,"@"+w+"@")
        #p=p.replace(sent,"@"+sent+"@")
        para.append((sent,start))
      if len(para)>1:
        #print("para_length",len(para))
        #print((w,doc_id,r))
        #print(c)
        if len(para)<=r:
          sent=str(sent)
          sent=sent.replace(w,"@"+w+"@")
          p1=(sent,start)
        else:
          p1=para[r]#get the context based on the counter values of the entity mention
        #print("para>1",p1)
        data.append((p1[0],p1[1],w,span,label,doc_id,attr1,att_val1,attr2,att_val2,attr3,att_val3,attr4,att_val4,attr5,att_val5,sent))

        #data.append((p1[0],p1[1],w,span,label,doc_id,sent))
      elif len(para)==1:
        p1=para[0]
        #print("para=1",p1)
        data.append((p1[0],p1[1],w,span,label,doc_id,attr1,att_val1,attr2,att_val2,attr3,att_val3,attr4,att_val4,attr5,att_val5,sent))
        #data.append((p1[0],p1[1],w,span,label,doc_id,sent))
      else:
        #print("para=0",(w, span))
        p1=sent
        data.append((p1[0],p1[1],w,span,label,doc_id,attr1,att_val1,attr2,att_val2,attr3,att_val3,attr4,att_val4,attr5,att_val5,sent))
        #data.append((p1[0],p1[1],w,span,label,doc_id,sent))
        #data.append((sent,define_words,span,label,doc_id))
      #if x[-1] not in k.keys():
        #k[x[-1]]=t
    #data.append(k)
    #textdata.append(text)
  print("********Processing Ends******")
  print("no. of files processed is %s"%(ct))
  print("no. of sentences/context extracted is %s"%(len(data)))
  return data



# train_data=preprocess(input_dir)

# dev_data=preprocess(input_dir1)

# import pandas as pd

# train_data[2]

# df = pd.DataFrame(train_data,columns=["context","section","word","span","label","doc_id","attr1","att_val1","attr2","att_val2","attr3","att_val3","attr4","att_val4","attr5","att_val5","sent"])


# df_dev = pd.DataFrame(dev_data,columns=["context","section","word","span","label","doc_id","attr1","att_val1","attr2","att_val2","attr3","att_val3","attr4","att_val4","attr5","att_val5","sent"])



# input_dir2 = "/content/gdrive/My Drive/CMED_Harvard_Shared_task//generated_data/"

# df.to_csv(input_dir2+"/train_multi_attribute_section.csv")

# df_dev.to_csv(input_dir2+"/dev_multi_attribute_section.csv")

# df=pd.read_csv(input_dir2+"/train_multi_attribute_section.csv")

# df_dev=pd.read_csv(input_dir2+"/dev_multi_attribute_section.csv")

"""MULTI-LABEL MULTI-CLASS

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
df=pd.read_csv(input_dir+"/train_multi_attribute_section.csv")

df_dev=pd.read_csv(input_dir+"/dev_multi_attribute_section.csv")
df_disp=df[df.label=='Disposition ']
df=df_disp
print(len(df_disp))
df_disp_dev=df_dev[df_dev.label=='Disposition ']
print(len(df_disp_dev))
df_dev=df_disp_dev

le = LabelEncoder()
device="cuda"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#from tensorflow.keras import backend as K
#K.tensorflow_backend._get_available_gpus()
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split




#######################################
### --------- Setup BERT ---------- ###

from transformers import AutoTokenizer, AutoModel,AutoConfig,AutoModelForSequenceClassification

# Name of the BERT model to use
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#model_name = 'bert-base-uncased'
model_name="emilyalsentzer/Bio_ClinicalBERT"
# Max length of tokens
max_length = 128

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config,from_pt=True)


#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
#inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
certain= Dense(units=len(df.att_val1.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='certainty')(pooled_output)
actor= Dense(units=len(df.att_val2.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='actor')(pooled_output)
action= Dense(units=len(df.att_val3.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='action')(pooled_output)
temporal= Dense(units=len(df.att_val4.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='temporality')(pooled_output)
negation= Dense(units=len(df.att_val5.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='negation')(pooled_output)


#issue = Dense(units=len(data.Issue_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='issue')(pooled_output)
#product = Dense(units=len(data.Product_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='product')(pooled_output)
#outputs = {'issue': issue, 'product': product}
outputs = {'certainty': certain, 'actor': actor,'action': action,'temporality': temporal,'negation': negation}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()

le = LabelEncoder()

def label_convert(train_labels,test_labels):
  tags_vals = list(train_labels)
  le.fit(train_labels)
  le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
  print(le_name_mapping)
  train_labels=le.transform(train_labels)
  test_labels=le.transform(test_labels)
  return train_labels,test_labels
  #print(labels)
  #print(to_categorical(labels))



#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
#loss = {'issue': CategoricalCrossentropy(from_logits = True), 'product': CategoricalCrossentropy(from_logits = True)}
#metric = {'issue': CategoricalAccuracy('accuracy'), 'product': CategoricalAccuracy('accuracy')}
loss = {'certainty': CategoricalCrossentropy(from_logits = True), 'actor': CategoricalCrossentropy(from_logits = True),
        'action': CategoricalCrossentropy(from_logits = True),'temporality': CategoricalCrossentropy(from_logits = True),
        'negation': CategoricalCrossentropy(from_logits = True)}
metric = {'certainty': CategoricalAccuracy('accuracy'), 'actor': CategoricalAccuracy('accuracy'),'action': CategoricalAccuracy('accuracy'),
          'temporality': CategoricalAccuracy('accuracy'),'negation': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
#y_issue = to_categorical(data['Issue'])
#y_product = to_categorical(data['Product'])
data=df
# Ready test data
data_test=df_dev

y_certain,test_y_certain = label_convert(data['att_val1'].values,data_test['att_val1'].values)
y_actor,test_y_actor = label_convert(data['att_val2'].values,data_test['att_val2'].values)
y_action,test_y_action = label_convert(data['att_val3'].values,data_test['att_val3'].values)
y_temp,test_y_temp = label_convert(data['att_val4'].values,data_test['att_val4'].values)
y_neg,test_y_neg = label_convert(data['att_val5'].values,data_test['att_val5'].values)

def cat(l):
  return to_categorical(l)

y_certain=cat(y_certain)
y_actor=cat(y_actor)
y_action=cat(y_action)
y_temp=cat(y_temp)
y_neg=cat(y_neg)

test_y_certain=cat(test_y_certain)
test_y_actor=cat(test_y_actor)
test_y_action=cat(test_y_action)
test_y_temp=cat(test_y_temp)
test_y_neg=cat(test_y_neg)

# test_y_certain = to_categorical(le_certain.transform(data_test['att_val1'].values))
# test_y_actor = to_categorical(le_actor.transform(data_test['att_val2'].values))
# test_y_action = to_categorical(le_action.transform(data_test['att_val3'].values))
# test_y_temp = to_categorical(le_temp.transform(data_test['att_val4'].values))
# test_y_neg = to_categorical(le_neg.transform(data_test['att_val5'].values))




# Tokenize the input (takes some time)
x = tokenizer(
    text=data["sent"].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

test_x = tokenizer(
    text=data_test['sent'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


# attention_masks = []
# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#   seq_mask= [float(i>0) for i in seq]
#   attention_masks.append(seq_mask)

# Fit the model
out_path="./multiclasslabel_Section"
import os
if not os.path.exists(out_path):
    with tf.device('/gpu:0'):
        history = model.fit(
        #x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
        x={'input_ids': x['input_ids']},
        y={'certainty': y_certain,'actor': y_actor,'action': y_action,'temporality': y_temp,'negation': y_neg},
        #validation_split=0.2,
        batch_size=8,
        epochs=5)

    #root_dir1

    model.save("./multiclasslabel_Section")


    model.summary()

#######################################
### ----- Evaluate the model ------ ###





#{'Certain': 0, 'Conditional': 1, 'Hypothetical': 2, 'Unknown': 3}
#{'Patient': 0, 'Physician': 1, 'Unknown': 2}
# {'Decrease': 0, 'Increase': 1, 'OtherChange': 2, 'Start': 3, 'Stop': 4, 'UniqueDose': 5, 'Unknown': 6}
# {'Future': 0, 'Past': 1, 'Present': 2, 'Unknown': 3}
# {'Negated': 0, 'NotNegated': 1}


#SEE THAT THE TEST AND TRAIN LABELS DIFFER....IMPPPPPPPP


#{'Decrease': 0, 'Increase': 1, 'OtherChange': 2, 'Start': 3, 'Stop': 4, 'UniqueDose': 5, 'Unknown': 6}

# Run evaluation
#model_eval = model.evaluate(
    #x={'input_ids': test_x['input_ids']},
    #y={'certainty': test_y_certain,'actor': test_y_actor,'action': test_y_action,'temporality': test_y_temp,'negation': test_y_neg})

#action_accuracy: 0.4664 - 
#actor_accuracy: 0.9013 - 
#certainty_accuracy: 0.8386 - 
#negation_accuracy: 0.9821 - 
#temporality_accuracy: 0.7444

data_test=df_dev



from tensorflow import keras
if os.path.exists(out_path):
    model = tf.keras.models.load_model("./multiclasslabel_Section")

    model.summary()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#y_pred = model.predict([test_x['input_ids'],test_x['attention_mask']])#with attention
y_pred = model.predict(test_x['input_ids'])
#confusion_matrix = confusion_matrix(test_y_certain, np.rint(y_pred))



#ANALYSIS_PART
sent=data_test["sent"].values

pred_val=np.argmax(y_pred["certainty"],axis=1)
act_val=np.argmax(test_y_certain,axis=1)

#cert=[]
#for i,pred in enumerate(pred_val):
  #cert.append((sent[i],pred,act_val[i]))
#df_cert=pd.DataFrame(cert,columns=["sent","predicted","actual"])
#labels={0:'Certain', 1:'Conditional', 2:'Hypothetical', 3:'Unknown'}
#df_cert=df_cert.replace({"predicted": labels,"actual": labels})
#df_cert[df_cert["predicted"]!=df_cert["actual"]]

#EVALUATIONS

#With the large class performing better than the small ones, you would expect to see the micro average being higher than the macro average.
#“Is micro-average is preferable if the class is imbalanced”. It depends on what’s the objective. If you care about overall data not prefer any class, ‘micro’ is just fine. 
#However, let’s say, class A is rare, but it’s way important, ‘macro’ should be a better choice because it treats each class equally. ‘micro’ is better if we care more about the accuracy overall. ‘micro’ is closer to ‘accuracy’, while ‘macro’ is a bit different when it’s not dominated by prevalent class. 
#In a multi-class classification setup, micro-average is preferable if you suspect there might be a class imbalance

#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt

# print("Attention")
# labels={'Certain': 0, 'Conditional': 1, 'Hypothetical': 2, 'Unknown': 3}
# l1=list(labels.keys())
# c_certain = confusion_matrix(np.argmax(test_y_certain,axis=1), np.rint(np.argmax(y_pred["certainty"],axis=1)))
# print(c_certain)
# #disp = ConfusionMatrixDisplay(confusion_matrix=c_certain, display_labels=labels)

# #disp.plot(cmap=plt.cm.Blues)
# #plt.show()
# print("Classification Report")
# print(classification_report(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1)))
# print("\n Accuracy")
# print(accuracy_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1)))
# print("Macro-Fscore")
# print(f1_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1), average='macro'))
# print("Micro-Fscore")
# print(f1_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1), average='micro'))
print("Certainty")
labels={'Certain': 0, 'Conditional': 1, 'Hypothetical': 2, 'Unknown': 3}
l1=list(labels.keys())

print(labels)
c_certain = confusion_matrix(np.argmax(test_y_certain,axis=1), np.rint(np.argmax(y_pred["certainty"],axis=1)))
print(c_certain)
#disp = ConfusionMatrixDisplay(confusion_matrix=c_certain, display_labels=labels)

#disp.plot(cmap=plt.cm.Blues)
#plt.show()
print("Classification Report")
print(classification_report(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1)))
print("\n Accuracy")
print(accuracy_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1)))
print("Macro-Fscore")
print(f1_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1), average='macro'))
print("Micro-Fscore")
print(f1_score(np.argmax(test_y_certain,axis=1),np.argmax(y_pred["certainty"],axis=1), average='micro'))

# print("Attention")
# labels={'Patient': 0, 'Physician': 1, 'Unknown': 2}
# l1=list(labels.keys())
# c_actor = confusion_matrix(np.argmax(test_y_actor,axis=1), np.rint(np.argmax(y_pred["actor"],axis=1)))
# print(c_actor)
# disp = ConfusionMatrixDisplay(confusion_matrix=c_actor, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
# print("Classification Report")
# print(classification_report(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1)))
# print("\n Accuracy")
# print(accuracy_score(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1)))

#y_pred = model.predict(test_x['input_ids'])
print("Actor")
labels={'Patient': 0, 'Physician': 1, 'Unknown': 2}
l1=list(labels.keys())
print(labels)
c_actor = confusion_matrix(np.argmax(test_y_actor,axis=1), np.rint(np.argmax(y_pred["actor"],axis=1)))
print(c_actor)
#disp = ConfusionMatrixDisplay(confusion_matrix=c_actor, display_labels=l1)
#disp.plot(cmap=plt.cm.Blues)
#plt.show()
print("Classification Report")
print(classification_report(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1)))
print("\n Accuracy")
print(accuracy_score(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1)))
print("Macro-Fscore")
print(f1_score(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1), average='macro'))
print("Micro-Fscore")
print(f1_score(np.argmax(test_y_actor,axis=1),np.argmax(y_pred["actor"],axis=1), average='micro'))

# print("Attention")
# c_action = confusion_matrix(np.argmax(test_y_action,axis=1), np.rint(np.argmax(y_pred["action"],axis=1)))
# print(c_action)
# labels={'Decrease': 0, 'Increase': 1, 'OtherChange': 2, 'Start': 3, 'Stop': 4, 'UniqueDose': 5, 'Unknown': 6}
# l1=list(labels.keys())
# disp = ConfusionMatrixDisplay(confusion_matrix=c_action, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
# print("Classification Report")
# print(classification_report(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1)))
# print("\n Accuracy")
# print(accuracy_score(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1)))
print("Action")
c_action = confusion_matrix(np.argmax(test_y_action,axis=1), np.rint(np.argmax(y_pred["action"],axis=1)))
print(c_action)

labels={'Decrease': 0, 'Increase': 1, 'OtherChange': 2, 'Start': 3, 'Stop': 4, 'UniqueDose': 5, 'Unknown': 6}
print(labels)
l1=list(labels.keys())
# disp = ConfusionMatrixDisplay(confusion_matrix=c_action, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
print("Classification Report")
print(classification_report(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1)))
print("\n Accuracy")
print(accuracy_score(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1)))
print("Macro-Fscore")
print(f1_score(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1), average='macro'))
print("Micro-Fscore")
print(f1_score(np.argmax(test_y_action,axis=1),np.argmax(y_pred["action"],axis=1), average='micro'))





# print("Attention")
# c_temp = confusion_matrix(np.argmax(test_y_temp,axis=1), np.rint(np.argmax(y_pred["temporality"],axis=1)))
# print(c_temp)
# labels={'Future': 0, 'Past': 1, 'Present': 2, 'Unknown': 3}
# l1=list(labels.keys())
# disp = ConfusionMatrixDisplay(confusion_matrix=c_temp, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
# print("Classification Report")
# print(classification_report(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1)))
# print("\n Accuracy")
# print(accuracy_score(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1)))
print("Temporality")
c_temp = confusion_matrix(np.argmax(test_y_temp,axis=1), np.rint(np.argmax(y_pred["temporality"],axis=1)))
print(c_temp)
labels={'Future': 0, 'Past': 1, 'Present': 2, 'Unknown': 3}
l1=list(labels.keys())
print(labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=c_temp, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
print("Classification Report")
print(classification_report(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1)))
print("\n Accuracy")
print(accuracy_score(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1)))
print("Macro-Fscore")
print(f1_score(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1), average='macro'))
print("Micro-Fscore")
print(f1_score(np.argmax(test_y_temp,axis=1),np.argmax(y_pred["temporality"],axis=1), average='micro'))

# print("Attention")
# c_neg = confusion_matrix(np.argmax(test_y_neg,axis=1), np.rint(np.argmax(y_pred["negation"],axis=1)))
# print(c_neg)
# labels={'Negated': 0, 'NotNegated': 1}
# l1=list(labels.keys())
# disp = ConfusionMatrixDisplay(confusion_matrix=c_neg, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
# print("Classification Report")
# print(classification_report(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1)))
# print("\n Accuracy")
# print(accuracy_score(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1)))
print("negation")
c_neg = confusion_matrix(np.argmax(test_y_neg,axis=1), np.rint(np.argmax(y_pred["negation"],axis=1)))
print(c_neg)
labels={'Negated': 0, 'NotNegated': 1}
l1=list(labels.keys()
        )
print(labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=c_neg, display_labels=l1)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
print("Classification Report")
print(classification_report(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1)))
print("\n Accuracy")
print(accuracy_score(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1)))
print("Macro-Fscore")
print(f1_score(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1), average='macro'))
print("Micro-Fscore")
print(f1_score(np.argmax(test_y_neg,axis=1),np.argmax(y_pred["negation"],axis=1), average='micro'))

##

