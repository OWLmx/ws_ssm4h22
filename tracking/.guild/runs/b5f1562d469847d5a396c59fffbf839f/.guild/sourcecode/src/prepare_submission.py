"""Prepare data for submission

Task: EVENT & ATTR

"""
from posixpath import split
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

def generate_pred_files(preds:DataFrame, previous_stage_anns_dir:str, include_prev_anns=['T'], out_dir:str='', event_label_mapper=None, columIdT='id', subtask=2, **kwargs):
    # from predictions in a DF generate bratt files
    
    if columIdT!='id':
        preds['id_bak'] = preds['id']
        preds['id'] = preds[columIdT]

    used_predictions = {}
    predicting_events=False
    if 'event_pred' in kwargs and kwargs['event_pred'] and subtask==2: # predicting events
        predicting_events=True
        used_predictions['event_pred'] = kwargs['event_pred']
        preds['event_yhat_label'] = preds[kwargs['event_pred']]
        if event_label_mapper:
            if path.isfile(event_label_mapper):
                event_label_mapper = joblib.load(event_label_mapper)
            elif type(event_label_mapper) is str and '|' in event_label_mapper:
                print(f"------------- INTERPRETING {event_label_mapper}")
                event_label_mapper = { int(k.split('|')[0]): k.split('|')[1] for k in event_label_mapper.split(',')}
            print(f"-=========== {event_label_mapper} ====================")
            if isinstance(event_label_mapper, LabelEncoder):
                preds['event_yhat_label'] = event_label_mapper.inverse_transform (preds['event_yhat_label']) # label to text
            else:
                preds['event_yhat_label'] = preds['event_yhat_label'].map(event_label_mapper) # label to text
    else: #not predicting events (given)
        predicting_events=False
        logger.info("Using given events... (not infering)")
        preds['event_yhat_label'] = preds['event']
        pass


    t2e_map = {}
    for fname, df in preds.groupby(by=['file']):
        # rs = [ f"{i+1}\t{r.final_yhat}:{r.id}\n"  for i,r in enumerate(df.itertuples()) if r.id.startswith('T') ]

        rs = []

        # prepare events
        if predicting_events:
            for i,r in enumerate(df.itertuples()):
                if r.id.startswith('T'):
                    tgtE = f"E{i+1}"
                    t2e_map[r.id] = tgtE # map from event top T.... used un next step for attrs 
                    rs.append(f"{tgtE}\t{r.event_yhat_label}:{r.id}\n")                


        # prepare contexts (attributes)
        attrs = [(karg, kwargs[karg], karg.split('_')[1]) for karg in kwargs if karg.startswith('attr_') and karg.endswith('_pred')]
        attr_label_mappers = {}
        for kattr, vattr, attrname in attrs:
            used_predictions[kattr] = vattr
            mapper_name = f"{kattr[:-len('_pred')]}_label_mapper"            
            if mapper_name in kwargs:
                attr_mapper = kwargs[mapper_name]
                if type(attr_mapper) is str and '|' in attr_mapper:
                    # print(f"------------- INTERPRETING {attr_mapper}")
                    attr_mapper = { int(k.split('|')[0]): k.split('|')[1] for k in attr_mapper.split(',')}
                    attr_label_mappers[vattr] = attr_mapper
        attri = 0
        for i,(idx,r) in enumerate(df.iterrows()):
            if r.event_yhat_label.lower() == 'disposition': # only disposition events
                for kattr, vattr, attrname in attrs:
                    attri += 1
                    tgtE = t2e_map[r.id] if predicting_events else r.id_y # TODO hardcoded logic (id_y resulted from manually joining segments with EVENT.annotations where this have the suffix _y (misc.ipynb) )
                    if vattr in attr_label_mappers and attr_label_mappers[vattr]:
                        rs.append(f"A{attri}\t{attrname.capitalize()} {tgtE} {attr_label_mappers[vattr][r[vattr]]}\n")            
                    else:
                        rs.append(f"A{attri}\t{attrname.capitalize()} {tgtE} {r[vattr]}\n")            
            pass


        # use (include) annotations from GoldStandard
        if include_prev_anns:
            with open(path.join(previous_stage_anns_dir, fname+'.ann'), 'r') as f_gs:
                gs_anns = f_gs.readlines()
            rs += [e for e in gs_anns if e[0] in include_prev_anns ]

        with open(path.join(out_dir, fname+'.ann'), 'w') as f:
            f.writelines(rs)

    return used_predictions




def main(previous_stage_anns_dir, prediction_data, test_gs_dir, subtask,  **kwargs):

    logger.info(f"*** Preparing data for submission for subtask: {subtask} [{prediction_data}] ")
    df_preds = pd.read_csv(prediction_data, sep='\t' if prediction_data.endswith('.tsv') else ',')

    ## stage 2
    stage=2
    submission_dir = f"submission_stage{stage}"
    makedirs(submission_dir, exist_ok=True)    
    used_predictions = generate_pred_files(df_preds, previous_stage_anns_dir=previous_stage_anns_dir, 
    include_prev_anns=['T', 'E'] if subtask==3 else ['T'], 
    out_dir=submission_dir, subtask=subtask, **kwargs)

    if test_gs_dir:
        # logger.info(f"Predictions used: \n\t{ ''.join(['\n\t' + k + '=' + {v} for k,v in used_predictions.items()])}")        
        logger.info(f"Predictions used: { used_predictions }")        
        logger.info(f"\n\n===== Oficial evaluation =====")
        eval_script.main(test_gs_dir, submission_dir, verbose=False)


if __name__ == '__main__':

    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Fast sanity check") 
    parser.add_argument('--previous_stage_anns_dir', type=str, default="data/input/trainingdata_v3/dev", help="Dir with generated annotations of the previous stage") 
    parser.add_argument('--prediction_data', type=str, default="data/input/trainingdata_v3/train", help="Path to Dataframe with predictions")
    parser.add_argument('--subtask', type=int, default=2, help="Subtask for which data will be prepared")
    parser.add_argument('--test_gs_dir', type=str, default="data/input/trainingdata_v3/dev", help="Dir with GS files for official evaluation, eval is skipped if not provided ") 
    
    parser.add_argument('--event_pred', type=str, default="", help="prediction to use as event")
    parser.add_argument('--event_label_mapper', type=str, default="", help="Dict to map predictions or a Label encoder")
    parser.add_argument('--attr_actor_pred', type=str, default="actor_yhat_label", help="prediction to use as actor")
    parser.add_argument('--attr_actor_label_mapper', type=str, default="", help="label mapper for the column")
    parser.add_argument('--attr_action_pred', type=str, default="action_yhat_label", help="prediction to use as action")
    parser.add_argument('--attr_action_label_mapper', type=str, default="", help="label mapper for the column")
    parser.add_argument('--attr_certainty_pred', type=str, default="certainty_yhat_label", help="prediction to use as certainty")
    parser.add_argument('--attr_certainty_label_mapper', type=str, default="", help="label mapper for the column")
    parser.add_argument('--attr_negation_pred', type=str, default="negation_yhat_label", help="prediction to use as negation")
    parser.add_argument('--attr_negation_label_mapper', type=str, default="", help="label mapper for the column")
    parser.add_argument('--attr_temporality_pred', type=str, default="temporality_yhat_label", help="prediction to use as temporality")
    parser.add_argument('--attr_temporality_label_mapper', type=str, default="", help="label mapper for the column")
    parser.add_argument('--columIdT', type=str, default="id", help="column name of the T ids")

    args = parser.parse_args()        
 
    main(**vars(args))

