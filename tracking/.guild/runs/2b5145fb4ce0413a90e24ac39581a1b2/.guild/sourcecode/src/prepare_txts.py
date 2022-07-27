"""Preprocess text files (clean, segment, ...)

Task: For all tasks
Approach: Segment text and put all segments in a single TSV with file-name as reference
        also all annotations are pre-interpreted and put in a single TSV

Output: segments.tsv, annotations.tsv

"""

from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame
from os import makedirs, path, listdir
from tqdm import tqdm 
from random import sample
import logging

from plibs import text_preprocessors as tpp

logger = logging.Logger(__name__)



def read_ann(in_dir:str, fname:str) -> DataFrame:
    """Reads an Ann file and returns a pre-interpreted Pandas DF

    Interpret types T E and A and assign values to columns
    ['id', 'attrs', 'term', 'etype', 'file', 'idx_ini', 'idx_end', 'event',
       'tgt', 'attr', 'attr_value'],

    Args:
        in_dir (str): Dir where file exists
        fname (str): filename (no extension)

    Returns:
        DataFrame: Pre-interpreted DF
    """
    try:
        if fname.endswith(".ann"):
            fname = fname[:-4] 
        ann = pd.read_csv(path.join(in_dir, fname) + ".ann", sep="\t", header=None, names=['id', 'attrs', 'term'])

        ann['etype'] = ann.id.map(lambda v: v[0])
        ann['file'] = fname
        ann['off_ini'] = None
        ann['off_end'] = None
        ann['event'] = None
        ann['tgt'] = None

        for row in ann.itertuples():
            if row.etype == 'T':
                v = row.attrs.split(' ')
                ann.at[row.Index, "off_ini"] = int(v[1])
                ann.at[row.Index, "off_end"] = int(v[2])
            elif row.etype == 'E':
                v = row.attrs.split(':')
                ann.at[row.Index, "event"] = v[0].strip()
                ann.at[row.Index, "tgt"] = v[1].strip()
            elif row.etype == 'A':
                v = row.attrs.split(' ')
                ann.at[row.Index, "attr"] = v[0].strip()
                ann.at[row.Index, "tgt"] = v[1].strip()
                ann.at[row.Index, "attr_value"] = v[2].strip()

        return ann
    except Exception:
        logger.warn(f"Error on reading {in_dir}/{fname}")
        return None


def read_txt(in_dir, fname):
    with open(path.join(in_dir, fname +  ("" if fname.endswith(".txt") else ".txt")), 'r') as f:
        lines = f.readlines()
        txt = ''.join(lines)
    return txt

def prepare(in_dir:str, text_preprocessor:tpp.TextPreprocessor, limit:int=None, **kwargs):    
    """apply indicated preprocessing strategy to all texts in the in_dir

    The main purpose of the text's preprocessing step is segmentation

    Args:
        in_dir (str): directory with all txt and ann files 
        text_preprocess_strategy (_type_): implemented preprocessing strategy
        limit (int, optional): max number of files to be processed Defaults to None.

    Returns:
        tuple(DataFrame, DataFrame): preprocessed_texts, preprocessed_annotations
    """
    # apply indicated preprocessing strategy to all texts in the in_dir
    # retur

    fnames = [f[:-4] for f in listdir(in_dir) if f.endswith('.txt')]
    if limit:
        fnames = sample(fnames, limit)

    annots = pd.DataFrame()
    segments = pd.DataFrame()
    for i, fname in tqdm(enumerate(fnames)):
        txt = read_txt(in_dir, fname)
        ann = read_ann(in_dir, fname)

        if not txt or ann is None or ann.empty:
            logger.warn(f"Error preparing [{fname}] annotatio    # get preprocens or text are empty.")
        else:
            annots = annots.append(ann, ignore_index=True)
            try:
                segmenti = text_preprocessor.process(txt, ann)
                segments = segments.append(segmenti, ignore_index=True)
            except Exception as err:
                logger.warn(f"Error proprocessing file [{fname}]", err)

    return segments, annots

def main(out_dir:str, strategy:str, t10sec:bool = False, **kwargs):

    # initialize preprocessing strategy
    if strategy not in tpp.strategies:
        logger.error(f"Text preprocessing steategy [{strategy}] not found.")
    text_preprocessor = tpp.strategies[strategy](**kwargs)

    # generate tgt dir if not exists
    makedirs(out_dir, exist_ok=True)
    prepated_txts, prepared_anns = prepare(
        text_preprocessor=text_preprocessor, 
        limit=10 if t10sec else None, 
        **kwargs)

    # save DFs: all anns (interepreted), segments, hydrated_examples, ...
    prepated_txts.to_csv(path.join(out_dir, "segments.tsv"), sep='\t')
    prepared_anns.to_csv(path.join(out_dir, "annotations.tsv"), sep='\t')
    
    

if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--in_dir', type=str, default="data/input/trainingdata_v3", help="Dir with all text and ann files") 
    parser.add_argument('--out_dir', type=str, default="data/prepared", help="Output dir") 
    parser.add_argument('--strategy', type=str, default="segmentbysentence", help="Name of the strategy to be used for segmentation") 
    parser.add_argument('--spacy_model', type=str, default="en_core_web_sm", help="Spacy model type to be used") 
    parser.add_argument('--using_char_span', type=bool, default="True", help="Use spacy's char_span to detect sentences, if False uses a more lenient boundary search") 

    args = parser.parse_args()
 
    main(**vars(args))
