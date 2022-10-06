from argparse import ArgumentParser
from operator import mod
from os import path
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from plibs.utils.transformers_utils import TransformerClassifierInferenceWrapper
# from plibs.estimators.sequence_classifier_plusfeats import SequenceClassifierPlusFeats as PtlModel
from plibs.estimators.sequence_classifier_mtl import SequenceClassifierMTL as PtlModel



import logging

logger = logging.getLogger(__name__)

AVAIL_GPUS = min(1, torch.cuda.device_count())
estimator = None

def predict(tgt_data:pd.DataFrame, model_path:str, model_type:str='distilbert-base-uncased', input_feats = ['text'], label_mapper=None, rs_prefix="", output_logits=False, extra_features='', **kwargs):
    
    if rs_prefix and not rs_prefix.endswith("_"):
        rs_prefix += '_'

    # load  model trained on fakeddit
    logger.info(f"\n=========  Loading model [{model_type}::{model_path}]  ==========")    
    global estimator
    if model_path.endswith('.ckpt'):
        predictor = estimator if estimator else TransformerClassifierInferenceWrapper(model_path, tokenizer_type=model_type, loaded_model=PtlModel.load_from_checkpoint(model_path), **kwargs)
    else:
        predictor = estimator if estimator else TransformerClassifierInferenceWrapper(model_path, tokenizer_type=model_type, **kwargs)
    estimator = predictor

    # test the trained model
    logger.info(f"\n=========  Infering on [{tgt_data.shape}] using features: [{input_feats}]  ==========")    
    output_yhat = predictor.predict(tgt_data, features=input_feats, output_logits=output_logits, extra_features=extra_features.split(',') if extra_features else [] ) 

    if type(output_yhat) is dict:
        for target in output_yhat:
            prefixi = f'{rs_prefix}{target}_' if len(output_yhat) > 1 else rs_prefix
            yhat = output_yhat[target]
            if output_logits:
                yhat_logits = np.concatenate(np.array(yhat, dtype=object)[:,1])
                yhat = np.array(yhat, dtype=object)[:,0].tolist()
                tgt_data = pd.concat([tgt_data, pd.DataFrame(yhat_logits, columns=[ f"{prefixi}logits_{i}" for i in range(yhat_logits.shape[1])]).set_index(tgt_data.index) ], ignore_index=False, axis=1 )        
            
            tgt_data[f'{prefixi}yhat'] = yhat
    else:
        yhat = output_yhat
        if output_logits:
            yhat_logits = np.concatenate(np.array(yhat, dtype=object)[:,1])
            yhat = np.array(yhat, dtype=object)[:,0].tolist()
            tgt_data = pd.concat([tgt_data, pd.DataFrame(yhat_logits, columns=[ f"{rs_prefix}logits_{i}" for i in range(yhat_logits.shape[1])]).set_index(tgt_data.index) ], ignore_index=False, axis=1 )        
        
        tgt_data[f'{rs_prefix}yhat'] = yhat

    print(tgt_data)

    if label_mapper:        
        if isinstance(label_mapper, LabelEncoder):
            tgt_data[f'{rs_prefix}yhat_label'] = label_mapper.inverse_transform (tgt_data[f'{rs_prefix}yhat']) # label to text
        else:
            tgt_data[f'{rs_prefix}yhat_label'] = tgt_data[f'{rs_prefix}yhat'].map(label_mapper) # label to text


    return tgt_data


def main(tgtdata_path, model_path:str, model_type:str, input_features:str, label_mapper:str, rs_prefix:str, output_logits, tgtdata_format="", t10sec:bool=False, **kwargs):

    logger.info(f"Available GPUs: {AVAIL_GPUS}")

    # load label_encoder if specified
    if label_mapper:
        if path.isfile(label_mapper):
            label_mapper = joblib.load(label_mapper)
        elif type(label_mapper) is str and '|' in label_mapper:
            label_mapper = { int(k.split('|')[0]): k.split('|')[1] for k in label_mapper.split(',')}
            
    for tgtdata_pathx in tgtdata_path.split(','):

        # load data to predict
        if not tgtdata_format:
            tgtdata_format = path.basename(tgtdata_pathx).split('.')[-1]
        
        if tgtdata_format == 'csv':
            tgt_data = pd.read_csv(tgtdata_pathx) 
        elif tgtdata_format == 'tsv':
            tgt_data = pd.read_csv(tgtdata_pathx, sep='\t') 
        elif tgtdata_format == 'pickle' or tgtdata_format == 'pkl':
            tgt_data = pd.read_pickle(tgtdata_pathx)         
        else:
            tgt_data = pd.read_pickle(tgtdata_pathx)         


        if t10sec:
            tgt_data = tgt_data.sample(10).copy()

        # field transformations (new feature) the same transformation that was applied on training # TODO better to be abstracted in a library (for later)
        transformations_applied=False
        if 'fields_transformations' in kwargs and kwargs['fields_transformations'] == 'text|mask_term':
            logger.info(f"Applying transformation to [text]")
            transformation = lambda e: e.text[:e.off_ini] + '[MASK]' + e.text[e.off_end:]
            tgt_data['tmp']=tgt_data['text'] # temp back up for reverting the process
            tgt_data['text']= tgt_data.apply(transformation, axis=1)
            transformations_applied=True
        kwargs.pop('fields_transformations', None)



        # do prediction
        input_feats = [inx.split('|') if '|' in inx else inx for inx in input_features.split(',')] # handle nested lists (firt level , second level |)
        rs = predict(tgt_data, model_path, model_type, input_feats, label_mapper=label_mapper, rs_prefix=rs_prefix, output_logits=output_logits, **kwargs)

        # rever transformation for chained invocations that do not need transformed text
        if transformations_applied:
            rs['text']=rs['tmp']

        prefix = path.basename(tgtdata_pathx).split('.')[0]+'_' if len(tgtdata_path.split(',')) > 1 else ""
        logger.info(f"... Saving predictions to [{prefix}predictions.tsv]")
        rs.to_csv(f'{prefix}predictions.tsv', sep='\t', index=False)



if __name__ == "__main__":
  
    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster
 
    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)") 
    parser.add_argument('--tgtdata_path', type=str, default="") 
    parser.add_argument('--model_path', type=str, default="") 
    parser.add_argument('--model_type', type=str, default="distilbert-base-uncased") 
    parser.add_argument('--num_labels', type=int, default=3) 
    parser.add_argument('--input_features', type=str, default="text", help="Features in the data that will be passed as input to the model") 
    parser.add_argument('--label_mapper', type=str, default="") 
    parser.add_argument('--output_logits', type=bool, default=False, help="Add also the output model logits") 
    parser.add_argument('--rs_prefix', type=str, default="", help="Useful when several predictors are applied in the same dataset (e.g. Experiment name, model name, etc.") 
    parser.add_argument('--tgtdata_format', type=str, default="", help="expected format of the tgtdata: csv | tsv | pickle (if empyty inferred from extension)") 
    parser.add_argument("--fields_transformations", type=str, default="", help="Dictionary where key is the field and the value the transformation to be applied")
    parser.add_argument("--extra_features", type=str, default="", help="")

    args = parser.parse_args()        
     
    main(**vars(args))