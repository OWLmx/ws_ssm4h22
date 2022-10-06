"""Dummy EVENT classifier

Task: EVENT classification
Approach: dummy cls for pipeline sanity check
Scope: Train & Evaluate

"""
from os import getcwd
from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame
from os import makedirs, path

from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
import eval_script

import logging

logger = logging.Logger(__name__)

X_feats =['text']
y_feats = 'event'

def generate_pred_files(preds:DataFrame, gs_dir:str, use_gs_anns=['T'], out_dir:str=''):
    # from predictions in a DF generate bratt files
    
    for fname, df in preds.groupby(by=['file']):
        rs = [ f"E{i+1}\t{r.yhat}:{r.tgt}\n"  for i,r in enumerate(df.itertuples()) if r.tgt.startswith('T') ]

        # use (include) annotations from GoldStandard
        if use_gs_anns:
            with open(path.join(gs_dir, fname+'.ann'), 'r') as f_gs:
                gs_anns = f_gs.readlines()
            rs += [e for e in gs_anns if e[0] in use_gs_anns ]

        with open(path.join(out_dir, fname+'.ann'), 'w') as f:
            f.writelines(rs)
    

def main(train_data, valid_data, test_data, test_gs_dir, **kwargs):

    df_train:pd.DataFrame = pd.read_pickle(train_data)
    df_valid = pd.read_pickle(valid_data)
    df_test = pd.read_pickle(test_data)

    cls = DummyClassifier(strategy='most_frequent')

    cls.fit(df_train[X_feats], df_train[y_feats] )

    logger.info("*** Evaluation on VALID split")
    yhat = cls.predict(df_valid[X_feats])
    print(classification_report(y_true=df_valid[y_feats], y_pred=yhat))

    logger.info("*** Evaluation on TEST split")
    yhat = cls.predict(df_test[X_feats])
    df_test['yhat'] = yhat
    print(classification_report(y_true=df_test[y_feats], y_pred=yhat))



    print("*** Official EVAL")
    stage=2
    test_preds_dir = f"yhat_stage{stage}"
    makedirs(test_preds_dir, exist_ok=True)
    generate_pred_files(df_test, gs_dir=test_gs_dir, use_gs_anns=['T'], out_dir=test_preds_dir)
    eval_script.main(test_gs_dir, test_preds_dir, verbose=False)


if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--train_data', type=str, default="data/input/trainingdata_v3/train", help="Dataframe with prepared examples") 
    parser.add_argument('--valid_data', type=str, default="data/input/trainingdata_v3/dev", help="Dataframe with prepared examples") 
    parser.add_argument('--test_data', type=str, default="data/input/trainingdata_v3/dev", help="Dataframe with prepared examples") 
    parser.add_argument('--test_gs_dir', type=str, default="data/input/trainingdata_v3/dev", help="Dir with GS files for official evaluation ") 

    args = parser.parse_args()    
    
 
    main(**vars(args))
