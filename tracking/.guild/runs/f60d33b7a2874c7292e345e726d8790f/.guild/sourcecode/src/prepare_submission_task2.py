import joblib
from os import getcwd
from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame
from os import makedirs, path
import logging
from sklearn.preprocessing import LabelEncoder

import eval_script

logger = logging.getLogger(__name__)


def prepare_submission_2a(df_predictions, original_data, yhat_field = 'yhat', **kwargs):

    df = df_predictions[['id', 'text', 'claim', yhat_field]]
    # val = val.drop(columns=['labels', 'Premise']).rename(columns={'predictions': 'Stance'})
    df = df.rename(columns={'claim': 'Claim', yhat_field: 'Stance'})

    if original_data:
        # check consistency
        df_original = pd.read_csv(original_data, sep='\t')
        assert len(df) == len(df_original), f"Inconsistent number of records: [{len(df)} != {len(df_original)}] " 

        missing_tweets = set(df_original['id']).difference(set(df['id']))
        assert len(missing_tweets)==0, f"There are missing tweets: [{missing_tweets}]"


    # expected cols: id	text	Claim	Stance              
    df.to_csv('stance_predictions.tsv', sep='\t', index=False)

    return df


prepare_data = {
    '2a': prepare_submission_2a,
    }



def main(prediction_data, subtask,  **kwargs):

    logger.info(f"*** Preparing data for submission for subtask: {subtask} [{prediction_data}] ")
    df_preds = pd.read_csv(prediction_data, sep='\t' if prediction_data.endswith('.tsv') else ',')

    df = prepare_data[subtask](df_predictions = df_preds, **kwargs)
    


if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--prediction_data', type=str, default="", help="Path to Dataframe with predictions")
    parser.add_argument('--subtask', type=str, default="2a", help="Subtask for which data will be prepared")
    parser.add_argument('--original_data', type=str, default="/home/owlmx/research/comps/SMM4H22/data/input/task2/official_test/test.tsv", help="Usef for consistency checks") 
    parser.add_argument('--yhat_field', type=str, default="yhat", help="Field name with yhat value")
    
    args = parser.parse_args()        
 
    main(**vars(args))

