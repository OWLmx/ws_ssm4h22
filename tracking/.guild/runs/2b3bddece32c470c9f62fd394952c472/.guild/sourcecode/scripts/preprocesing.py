#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:33:26 2022

@author: vanikanjirangat
"""

import spacy
nlp = spacy.load('en_core_web_sm')

root_dir = "../CMED_Harvard_Shared_task//trainingdata_v3/"
input_dir=os.path.join(root_dir,'train')
input_dir1=os.path.join(root_dir,'dev')

#Preprocessing-part
import glob
import re
def preprocess(dir):
  items=[]
  c1=glob.glob(dir+"/*.ann")
  for c in c1:
    with open(c) as f1:
      p=c.split(".")[0]
      m=f1.read()
      m1=m.split("\n")
      m2=[k.split("\t") for k in m1]
      m3=[(k[0],k[1],k[-1]) for k in m2 if len(k)==3]
      spans=[re.findall('\d+', s[1]) for s in m3]
      labels=[re.split('\d+', s[1])[0] for s in m3]
      m4=[(k[0],labels[i],spans[i],k[-1]) for i,k in enumerate(m3)]
      items.append((p,m4))

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
  for i,item in enumerate(items):
    k={}
    #print(i)
    path=item[0]
    #print(path)
    if path in ptext:
      idx=ptext.index(path)
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
      define_words=x[-1]
      span=int(x[2][0]),int(x[2][1])
      label=x[1]
      w=x[-1]
      word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
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
          para.append(ctxt)#append the context/section of retrived sentence
        
      if fg==0:
        para.append(sent)
      if len(para)>1:
        #print("para_length",len(para))
        #print((w,doc_id,r))
        #print(c)
        if len(para)<=r:
          p1=sent
        else:
          p1=para[r]#get the context based on the counter values of the entity mention
        #print("para>1",p1)
        data.append((p1,w,span,label,doc_id,sent))
      elif len(para)==1:
        p1=para[0]
        #print("para=1",p1)
        data.append((p1,w,span,label,doc_id,sent))
      else:
        #print("para=0",(w, span))
        p1=sent
        data.append((p1,w,span,label,doc_id,sent))
        #data.append((sent,define_words,span,label,doc_id))
      #if x[-1] not in k.keys():
        #k[x[-1]]=t
    #data.append(k)
    #textdata.append(text)
  print("********Processing Ends******")
  print("no. of files processed is %s"%(ct))
  print("no. of sentences/context extracted is %s"%(len(data)))
  return data

train_data=preprocess(input_dir)

dev_data=preprocess(input_dir1)


import pandas as pd
df = pd.DataFrame(train_data,columns=["context","word","span","label","doc_id","sent"])
df_dev = pd.DataFrame(dev_data,columns=["context","word","span","label","doc_id","sent"])



df.to_csv(save_dir+"/train_para.csv")

df_dev.to_csv(save_dir+"/dev_para.csv")

'''
If you need to explicitly mark the entities: ues the script below
'''

import glob
import re
#Marking the entities
def preprocess(dir):
  items=[]
  c1=glob.glob(dir+"/*.ann")
  for c in c1:
    with open(c) as f1:
      p=c.split(".")[0]
      m=f1.read()
      m1=m.split("\n")
      m2=[k.split("\t") for k in m1]
      m3=[(k[0],k[1],k[-1]) for k in m2 if len(k)==3]
      spans=[re.findall('\d+', s[1]) for s in m3]
      labels=[re.split('\d+', s[1])[0] for s in m3]
      m4=[(k[0],labels[i],spans[i],k[-1]) for i,k in enumerate(m3)]
      items.append((p,m4))

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
  for i,item in enumerate(items):
    k={}
    #print(i)
    path=item[0]
    #print(path)
    if path in ptext:
      idx=ptext.index(path)
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
      define_words=x[-1]
      span=int(x[2][0]),int(x[2][1])
      label=x[1]
      w=x[-1]
      word = doc.char_span(span[0], span[1])#get the word based on the entity span
      
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
        data.append((p1[0],p1[1],w,span,label,doc_id,sent))
      elif len(para)==1:
        p1=para[0]
        #print("para=1",p1)
        data.append((p1[0],p1[1],w,span,label,doc_id,sent))
      else:
        #print("para=0",(w, span))
        p1=sent
        data.append((p1[0],p1[1],w,span,label,doc_id,sent))
        #data.append((sent,define_words,span,label,doc_id))
      #if x[-1] not in k.keys():
        #k[x[-1]]=t
    #data.append(k)
    #textdata.append(text)
  print("********Processing Ends******")
  print("no. of files processed is %s"%(ct))
  print("no. of sentences/context extracted is %s"%(len(data)))
  return data


