from argparse import ArgumentParser
from os import path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


import logging

logger = logging.getLogger(__name__)


def predict(tgt_data:pd.DataFrame, model_path:str, model_type:str, input_feats:list=['text'], label_mapper=None, rs_prefix="", output_logits=False, **kwargs):
    
    if rs_prefix and not rs_prefix.endswith("_"):
        rs_prefix += '_'

    # load  model trained on fakeddit
    logger.info(f"\n=========  Loading model [{model_type}::{model_path}]  ==========")
    predictor = joblib.load(model_path)

    # test the trained model
    logger.info(f"\n=========  Infering on [{tgt_data.shape}] using features: [{input_feats}]  ==========")    
    yhat = predictor.predict(tgt_data[input_feats].values)
    tgt_data[f'{rs_prefix}yhat'] = yhat

    if label_mapper:        
        if isinstance(label_mapper, LabelEncoder):
            tgt_data[f'{rs_prefix}yhat_label'] = label_mapper.inverse_transform (tgt_data[f'{rs_prefix}yhat']) # label to text
        else:
            tgt_data[f'{rs_prefix}yhat_label'] = tgt_data[f'{rs_prefix}yhat'].map(label_mapper) # label to text


    return tgt_data

def main(tgtdata_path, model_path:str, model_type:str, input_features:str, label_mapper:str, rs_prefix:str, tgtdata_format="", t10sec:bool=False, **kwargs):


    # load data to predict
    if not tgtdata_format:
        tgtdata_format = path.basename(tgtdata_path).split('.')[-1]
    
    if tgtdata_format == 'csv':
        tgt_data = pd.read_csv(tgtdata_path) 
    elif tgtdata_format == 'tsv':
        tgt_data = pd.read_csv(tgtdata_path, sep='\t') 
    elif tgtdata_format == 'pickle' or tgtdata_format == 'pkl':
        tgt_data = pd.read_pickle(tgtdata_path)         
    else:
        tgt_data = pd.read_pickle(tgtdata_path)         


    if t10sec:
        tgt_data = tgt_data.sample(10).copy()

    # load label_encoder if specified
    if label_mapper:
        if path.isfile(label_mapper):
            label_mapper = joblib.load(label_mapper)
        elif type(label_mapper) is str and '|' in label_mapper:
            label_mapper = { int(k.split('|')[0]): k.split('|')[1] for k in label_mapper.split(',')}

    # do prediction
    input_feats = [inx.split('|') if '|' in inx else inx for inx in input_features.split(',')] # handle nested lists (firt level , second level |)
    rs = predict(tgt_data, model_path, model_type, input_feats, label_mapper=label_mapper, rs_prefix=rs_prefix, **kwargs)

    logger.info(f"... Saving predictions to [predictions.csv]")
    rs.to_csv('predictions.tsv', sep='\t', index=False)



if __name__ == "__main__":
  
    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster
 
    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)") 
    parser.add_argument('--tgtdata_path', type=str, default="") 
    parser.add_argument('--tgtdata_format', type=str, default="", help="expected format of the tgtdata: csv | tsv | pickle (if empyty inferred from extension)") 
    parser.add_argument('--model_path', type=str, default="", help="Trained estimator to be used") 
    parser.add_argument('--model_type', type=str, default="") 
    parser.add_argument('--num_labels', type=int, default=3) 
    parser.add_argument('--input_features', type=str, default="text", help="Features in the data that will be passed as input to the model") 
    parser.add_argument('--label_mapper', type=str, default="") 
    # parser.add_argument('--ou tput_logits', type=bool, default=False, help="Add also the output model logits") 
    parser.add_argument('--rs_prefix', type=str, default="", help="Useful when several predictors are applied in the same dataset (e.g. Experiment name, model name, etc.") 


    args = parser.parse_args()        
     
    main(**vars(args))