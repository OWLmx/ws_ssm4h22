"""Generate Dataframes with hydrated examples from previously processed splits, texts, annotations   

Task: For all tasks
Approach: Associate text segments with labels (accordingly to stage) and put it in a DFF


Output: DataFrames (pickle) one per split per stage
"""

from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame
from os import path, makedirs
import logging

logger = logging.Logger(__name__)

def hydrate_examples(segments:DataFrame, anns:DataFrame, stage=2) -> DataFrame:
    """Format examples entries for training based on preprocessed text and annotations

    Using preprocessed texts (i.e., segmented) and annotations, generate a DataFrame
    with entries prepared for training classiers.

    The type of format (and the data hydrated) depends on the target stage (stage param) 

    Args:
        segments (DataFrame): DataFrame with one line per segment associated with an Term (T)
        anns (DataFrame): preprocessed annotations (pre-interpreted)
        stage (int, optional): Target stage. Defaults to 2.

    Returns:
        DataFrame: Entries with the text, the event, attribute, attribute value, etc. 
    """

    # based on preprocessed texts and annotations, hydrate examples for event training 

    rss = []
    if stage>=1:
        rs = []
        rss.insert(0, ('stage1', segments))    
    if stage>=2:
        rs = []
        for rann in anns[anns.etype=='E'].itertuples():
            try:
                rtxt = segments[(segments.file==rann.file) & (segments.id==rann.tgt)]
                if rtxt is None or rtxt.empty:
                    logger.warn(f"Segment [{rann.file} & {rann.tgt}] not found")
                else:
                    rtxt = rtxt.iloc[0]
                    rs.append( {**rtxt, 'id': rann.id, 'event': rann.event, 'tgt': rann.tgt })
            except Exception as err:
                logger.warn(f"Error hydrating[{stage}] {rann}", err)        
        rss.insert(0, ('stage2', pd.DataFrame(rs)) )
    if stage==3:
        evs = rss[0][1] # from previous stage (above if)
        print(evs)
        rs = []
        for rann in anns[anns.etype=='A'].itertuples():
            try:
                revs = evs[(evs.file==rann.file) & (evs.id==rann.tgt)]
                if revs is None or revs.empty:
                    logger.warn(f"Event [{rann.file} & {rann.tgt}] not found")
                else:
                    revs = revs.iloc[0]
                    rs.append( {**revs, 'id': rann.id, 'attr' : rann.attr, 'attr_value' : rann.attr_value, 'tgt': rann.tgt}  )
            except Exception as err:
                logger.warn(f"Error hydrating[{stage}] {rann}", err)
        rss.insert(0, ('stage3', pd.DataFrame(rs)) )

    return tuple(rss)

def prepare(assigned_splits:DataFrame, annotations:DataFrame, prepared_txts:DataFrame, stage:int=3, out_dir:str=""):
    # hydrate examples for each split and for each stage

    for split_name in assigned_splits.split.unique():
        split_files = assigned_splits[assigned_splits.split==split_name].file.values
        logger.info(f"Hydrating split: {split_name} [{len(split_files)} files] ... ")

        rss = hydrate_examples(
            segments=prepared_txts[prepared_txts.file.isin(split_files)], 
            anns=annotations[annotations.file.isin(split_files)], stage=stage)
        for stage_name, rs in rss:
            rs.to_pickle(path.join(out_dir, f"{stage_name}_{split_name}.pkl"), protocol=4 ) # protocol 4 for compatibility


def main(prepared_splits:str, prepared_texts:str, prepared_annots:str, out_dir:str, stage:int, t10sec:bool = False, **kwargs):

    # save DFs: all anns (interepreted), segments, hydrated_examples, ...
    annotations = pd.read_csv(prepared_annots, sep='\t')
    texts = pd.read_csv(prepared_texts, sep='\t')
    splits = pd.read_csv(prepared_splits, sep='\t')

    makedirs(out_dir, exist_ok=True)
    prepare(assigned_splits=splits, prepared_txts=texts, annotations=annotations, out_dir=out_dir, stage=stage)
    

if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--prepared_splits', type=str, default="data/prepared/splits.tsv", help="TSV with split assigned per file") 
    parser.add_argument('--prepared_annots', type=str, default="data/prepared/annotations.tsv", help="TSV with prepared texts (segmentation, ...)")     
    parser.add_argument('--prepared_texts', type=str, default="data/prepared/segments.tsv", help="TSV with prepared texts (segmentation, ...)")     
    parser.add_argument('--out_dir', type=str, default="data/prepared", help="Output dir") 
    parser.add_argument('--stage', type=int, default=3, help="Stage data is inteded for") 


    args = parser.parse_args()    
    
 
    main(**vars(args))
