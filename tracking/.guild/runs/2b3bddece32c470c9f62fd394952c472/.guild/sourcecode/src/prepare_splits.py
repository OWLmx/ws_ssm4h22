"""Generate metasummary (dataframe) of files and assign a split to each one 

Output: splits.tsv
"""

from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame
from os import path, listdir, makedirs
from tqdm import tqdm 

from collections import Counter
from sklearn.model_selection import train_test_split

import logging


logger = logging.Logger(__name__)

def get_files_metainfo(dirpaths, dirs_split_suffix={'train':None, 'valid':None, 'test':'test'}) -> DataFrame:
    """Generate a summary of labels in each file

    Dataframe with one row per file where columns are events and attributes
    and the values are the number of times that that even or attr appear
    in the file

    Useful for stratified split

    Includes the test split already assigned based on the suffix provided for the test directory
    default: if the dir ends with 'test' thoise files are assigned to the 'test' split

    Args:
        dirpath (_type_): Path of the dir with the files

    Returns:
        DataFrame: 
    """

    dfs = []
    for dirpath in dirpaths:
        split_name = ""
        for k in dirs_split_suffix:
            if dirs_split_suffix[k] and dirpath.endswith(dirs_split_suffix[k]):
                split_name = k
                break
        for i, fname in tqdm(enumerate(listdir(dirpath))):
            if fname.endswith('.ann'):
                try:                
                    df = pd.read_csv(path.join(dirpath, fname), sep="\t", header=None)
                    if not df.empty:
                        rs = {'file': fname.split('.')[0], 'split': split_name  }
                        rs.update( Counter(df[df[0].str.startswith('E')][1].map(lambda v: v.split(':')[0])) ) # num events
                        rs.update( Counter(df[df[0].str.startswith('A')][1].map(lambda v: v.split(' ')[0])) ) # num attributes
                        dfs.append(rs)
                except Exception:
                    logger.warn(f"Error on processing file [{fname}]")
                    pass

    dfs = pd.DataFrame(dfs)
    dfs = dfs.fillna(0)
    dfs = pd.concat([ dfs,  pd.DataFrame( list( dfs.apply(lambda e: {f"has_{k}": int(e[k]) > 0 for k in e.keys() if not k in ['file', 'split']}, axis=1) ) ) ], axis='columns')

    return dfs 

def split_data(files_metainfo:DataFrame,  splits=[('train', -1), ('valid', 50), ('test', 50)], override=False) -> DataFrame:
    # creates dataframe where each file is assigned to one split (or use already assigned through source dir)

    split_names = [s[0] for s in splits]
    missing_splits = set(split_names) if override else set(split_names).difference( set(files_metainfo.split.unique()) )
    missing_splits = [s for s in splits if s[0] in missing_splits and s[0]!= 'train']

    df = files_metainfo.copy() if override else files_metainfo[files_metainfo.split =='']

    split_ids= {}
    needed_exs = sum([s[1] for s in missing_splits ])
    train_, rest_ = train_test_split(df, test_size=needed_exs, stratify=df[[c for c in df.columns if c.startswith('has_')]] )
    split_ids['train'] = train_.file.values
    if 'valid' in missing_splits and 'test' in missing_splits: # two splits        
        valid_, test_ = train_test_split(rest_, test_size=missing_splits[1][1], stratify=rest_[[c for c in df.columns if c.startswith('has_')]] )
        split_ids['valid'] = valid_.file.values
        split_ids['test'] = test_.file.values
    else: # one split is enough
        split_ids[missing_splits[0][0]] = rest_.file.values
    
    # register assigned split
    for split in split_ids:
        files_metainfo.loc[files_metainfo.file.isin(split_ids[split]),'split'] = split

    return files_metainfo

def prepare(in_dirs:str) -> DataFrame:

    files_metainfo = get_files_metainfo(dirpaths=[d.strip() for d in in_dirs.split(',')])
    files_metainfo = split_data(files_metainfo) 

    return files_metainfo

def main(out_dir, t10sec, **kwargs):

    rs = prepare(**kwargs)
    makedirs(out_dir, exist_ok=True)
    rs.to_csv(path.join(out_dir, "splits.tsv"), sep='\t')

if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check")     
    parser.add_argument('--in_dirs', type=str, default="data/input/trainingdata_v3/train, data/input/trainingdata_v3/dev") 
    parser.add_argument('--out_dir', type=str, default="data/prepared", help="Output dir") 

    args = parser.parse_args()    
    
 
    main(**vars(args))
