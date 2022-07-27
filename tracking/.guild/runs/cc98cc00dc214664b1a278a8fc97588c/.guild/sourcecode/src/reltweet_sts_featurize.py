import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from plibs.features.featurizers import STSEmbedder


t10sec = True
columns = ['misconception_id', 'misconception', 'tweet_text_clean', 'label']
tgt_columns = ['misconception', 'tweet_text_clean']

transformer = STSEmbedder(SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1"))

dataset_file = "misconception-tweet_isrelated_processed.pkl"
df = pd.read_pickle(dataset_file)[:(10 if t10sec else None)]

print("STS Transforming... ")
rs = transformer.fit_transform(df[tgt_columns])

df.to_pickle('dataset.pkl') # normalize source dataset to later steps
for c in tgt_columns:
    torch.save(rs[c], f'{c}.pt')
