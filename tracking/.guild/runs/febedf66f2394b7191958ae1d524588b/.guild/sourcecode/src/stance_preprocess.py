from argparse import ArgumentParser

import pandas as pd
import os
from os import path, makedirs

# from scipy.sparse import data
from plibs.utils import lang, tweeter, datahandle
# from plibs.utils import tweeter, datahandle
from sklearn.model_selection import train_test_split

data_path = 'data/input/stancecovid_kglandt/'
dataset_type = 'kglandt'
tweet_text_column = "tweet_text"
target_lang = '__label__en'
splits = None
split_stratify = ['Stance', 'Claim']

def normalize_fieldnames(df):
    df.rename(lambda col: '_'.join(col.split(' ')).lower(), axis=1, inplace=True) # lowercase and remove spaces    


def prepare_data_kglandt(data_path, out_path, **kwargs):
    file_names = [f for f in os.listdir(data_path) if f.endswith('.tsv') and '_noisy' not in f]

    dfs_valid = pd.DataFrame()
    for f in file_names:
        df = pd.read_csv(f"{os.path.join(data_path, f)}", sep='\t', encoding = "utf-8")
        df = df.astype('U')
        df = preprocess_tweets(df, tweet_text_column='Tweet')
        df['labels'] = df['Stance'].map({"AGAINST": 0, "FAVOR": 1, "NONE": 2})

        # df.rename(columns={'Tweet': 'tweet', 'Claim': 'claim'  'Stance': 'stance'}, inplace=True)
        # df['Opinion'] = df['Opinion Towards'].apply(lambda e: e[:1])
        # df['split'] = f[:-4].split('_')[-1]
        # df['successfully_hydrated'] = ~df.tweet_text.isna()
        
        # dfs.append(df)

        # df = pd.concat(dfs, ignore_index=True)
        # df = df[df.successfully_hydrated] # only stats of hydrated

        # create statement -> positive expression of the Target 
        target2statement = { 
            "fauci": "Fauci is doing a good job.",
            "face_masks": "Face masks help to protect us.",
            "stay_at_home_orders": "Stay at home is a needed measure.",
            "school_closures": "Schools need to remain closed.",

            "face masks": "Face masks help to protect us.",
            "stay at home orders": "Stay at home is a needed measure.",
            "school closures": "Schools need to remain closed."
        }    
        df['Claim2'] = df['Claim'].map(target2statement) 

        # normalize field names    
        # normalize_fieldnames(df)
        df['groupkey'] = df[split_stratify[0]] +' | '+ df[split_stratify[1]]
        print(df.groupkey.value_counts())
        df, df_val_ = train_test_split(df, test_size=0.05 if 'train' in f else 0.125, stratify= df[['groupkey']]  )
        dfs_valid = dfs_valid.append(df_val_)

        makedirs(out_path, exist_ok=True)
        df.to_csv(path.join(out_path, "task2_train.tsv" if 'train' in f else "task2_test.tsv"), sep='\t')
    
    dfs_valid.to_csv(path.join(out_path, "task2_valid.tsv"), sep='\t')

    return df


def prepare_data_covilies(data_path, out_path, **kwargs):
    print("\n\n**** preprocessing COVIDLIES")
    # merge hydrated tweets
    file_names = [f for f in os.listdir(data_path) if f.endswith('.tsv') and '_noisy' not in f and 'tokenized' in f]
    df_tweets = pd.DataFrame()
    for f in file_names:
        df = pd.read_csv(f"{os.path.join(data_path, f)}", sep='\t', encoding = "utf-8")
        df = df.astype('U')
        # df = preprocess_tweets(df, tweet_text_column='tweet_text')

        df_tweets = df_tweets.append(df)

    df_tweets = df_tweets.drop_duplicates(subset=['tweet_id'])
    df_tweets = df_tweets.set_index(['tweet_id'])
    tweet_dict = df_tweets[['tweet_text']].to_dict('index')

    df_covidlies = pd.read_csv(path.join(data_path, "covid_lies.csv"))
    print(df_covidlies.columns)
    df_covidlies['Tweet'] = df_covidlies['tweet_id'].map(lambda v: tweet_dict.get(str(v), {'tweet_text': None})['tweet_text'])
    df_covidlies = df_covidlies[df_covidlies['Tweet'].notnull()]
    print(df_covidlies)
    df_covidlies = preprocess_tweets(df_covidlies, tweet_text_column='Tweet')
    df_covidlies['labels'] = df_covidlies['label'].map({"neg": 0, "pos": 1, "na": 2})


    # df_covidlies['groupkey'] = df_covidlies['label'] +' | '+ df_covidlies['misconception']
    df_covidlies['groupkey'] = df_covidlies['label']
    print(df_covidlies.groupkey.value_counts())
    df, df_val_ = train_test_split(df_covidlies, test_size=0.10, stratify= df_covidlies[['groupkey']]  )
    # dfs_valid = dfs_valid.append(df_val_)

    makedirs(out_path, exist_ok=True)
    df.to_csv(path.join(out_path, "covidlies_train.tsv"), sep='\t')
    df_val_.to_csv(path.join(out_path, "covidlies_valid.tsv"), sep='\t')

    return df_covidlies


prepare_data = {
    'kglandt': prepare_data_kglandt,
    'covidlies': prepare_data_covilies,
    }

# --------------------------------------

def preprocess_tweets(df, tweet_text_column='Tweet', target_lang=None):
    langId = lang.LanguageIdentification()
    df['tweet_text'] = df[tweet_text_column]
    df['tweet_text_clean'] = df.tweet_text.apply(tweeter.clean_tweet)
    df['tgt_lang'] = True if target_lang is None else df.tweet_text_clean.apply(lambda e: langId.is_language(e, target_lang, min_ratio_wrt_first=0.4)[0])

    return df[df['tgt_lang']]
    # return df
    
    
# =====================================





def main(t10sec, in_path, dataset_type, **kwargs):

    # consolidate data, filter-out non hydrated tweets, etc (dataset specific)
    df = prepare_data[dataset_type](data_path = in_path, **kwargs)
    # df.to_pickle(f"{dataset_type}.pkl")
    # print(df)

    # # preprocess (clean tweets, lang id, etc..)
    # df = preprocess_tweets(df)
    # # df.to_pickle(f"{dataset_type}_preprocessed.pkl" )
    # df.to_pickle(f"df_stance_preprocessed.pkl" )
    # print(df)



    # # generate splits
    # if splits:
    #     _splits = [(s.split(':')) for s in splits]
    #     _rs_splits = datahandle.splitify_dataframe(df, splits=[float(s[1]) for s in _splits], stratify=split_stratify, random_state=1212)
    #     for s in zip(_rs_splits, _splits):
    #         print(f"Dumping split [{s[1]}] => [{s[0].shape}]")
    #         s[0].to_pickle(f"df_stance_preprocessed_{s[1][0]}.pkl")


    pass

if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--in_path', type=str, default="", help="Input path") 
    parser.add_argument('--out_path', type=str, default="", help="Output")     
    parser.add_argument('--dataset_type', type=str, default="kglandt", help="Typ[e of dataset to process")     
    

    args = parser.parse_args()    
    

    main(**vars(args))
