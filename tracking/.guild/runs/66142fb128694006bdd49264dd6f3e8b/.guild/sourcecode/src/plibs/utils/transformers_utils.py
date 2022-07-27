from cmath import isnan
from os import path, listdir
from pyexpat import model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from functools import partial
from typing import Union
from pandas import DataFrame
import logging
# import numpy as np
import math 

logger = logging.getLogger(__name__)

class TransformerClassifierInferenceWrapper():
    """Wrapper for doing ONLY inference using a trained Classifier Transformer model
    """

    def __init__(self, model_path:str, tokenizer_type='self', custom_model_arch = None, use_gpu_if_available:bool=True, **kwargs) -> None:
        """

        Args:
            model_path (str): path to the trained model (where .bin and config.json are)
            tokenizer_type (str, optional): the tokenizer to use (e.g., bert-base-uncvased, if 'self' loads from model path, ). Defaults to 'self'.
        """
        if 'loaded_model' in kwargs and not kwargs['loaded_model'] is None: # model already loaded and passed
            logger.debug(f"Using already provided loaded model.")
            assert tokenizer_type!='self', "A tokenizer type has to be explicitly specified when using a custom modiel architecture"
            
            self.model = kwargs['loaded_model']
            self.model.eval()
        elif custom_model_arch:
            assert tokenizer_type!='self', "A tokenizer type has to be explicitly specified when using a custom modiel architecture"
                
            model_pathx = [path.join(model_path, f) for f in listdir(model_path) if f.endswith('.pt')][0] if not path.isfile(model_path) else model_path
            logger.debug(f"Loading model from: {model_pathx}")
            self.model = custom_model_arch
            self.model.load_state_dict(torch.load(model_pathx))
            self.model.eval()
        else:
            print(f"{model_path} => {kwargs}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)


        if use_gpu_if_available:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        logger.info(f"Using [{self.device}] device for inferring")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path if tokenizer_type=='self' else tokenizer_type, use_fast=True)                                        
        self.kwargs = kwargs

    def process_model_output(self, rs, output_probs, output_logits):
        # print(f"--> output :: \n {rs}")
        if rs.dim() ==1 :
            rs = rs.unsqueeze(0)
        # print(f"1************** {rs} --> {rs.shape}")
        rs = torch.softmax(rs, dim=1)
        # print(f"2************** {rs}")
        if output_probs:
            prob = torch.max(rs, dim=1)
            yhat = prob.indices.detach().to('cpu').numpy()[0]
            prob = prob.values.detach().to('cpu').numpy()[0]
            output = [yhat, prob]
        else:            
            yhat = torch.argmax(rs, dim=1)
            yhat = yhat.detach().to('cpu').numpy()[0]
            output = [yhat]

        if output_logits:
            output.append(rs.detach().to('cpu').numpy()) 
        return tuple(output) if len(output) > 1 else output[0]        
        

    def predict_one(self, inputs, output_probs=False, output_logits=False, max_length=128, **kwargs) -> Union[int, tuple]:
        """Predict class using model

        Args:
            inputs (text, tuple(text1, text2)): input for the estimator
            output_probs (bool, optional): If True the prediction "probability" is included. Defaults to False.

        Returns:
            (int, tuple): predicted class or tuple(predicted class, probability)
        """
        features = self.tokenizer.encode_plus(inputs, max_length=max_length, padding='max_length', return_tensors="pt", truncation=True).to(self.device)
        try:
            features.update(kwargs) # include extra args (e.g. extra feats exf_n_ or exf_c_ )
            with torch.no_grad():
                # print(f"--> kwargs:: \n\n{features}")
                # rs = self.model(**features)[0]
                
                rs = self.model(**features)
                if type(rs) is dict: # used in multi-target classification (multiple predictions -> multiple logits sets)
                    rs_dict = {}
                    for target in rs:
                        if 'logits' in target:
                            rs_dict[target.split('_')[1] if '_' in target else 'target'] = self.process_model_output(rs[target], output_probs, output_logits)
                    return rs_dict
                else:
                    rs = self.model(**features)[0]
                    return self.process_model_output(rs, output_probs, output_logits)

                # # print(f"--> output :: \n {rs}")
                # if rs.dim() ==1 :
                #     rs = rs.unsqueeze(0)
                # # print(f"1************** {rs} --> {rs.shape}")
                # rs = torch.softmax(rs, dim=1)
                # # print(f"2************** {rs}")
                # if output_probs:
                #     prob = torch.max(rs, dim=1)
                #     yhat = prob.indices.detach().to('cpu').numpy()[0]
                #     prob = prob.values.detach().to('cpu').numpy()[0]
                #     output = [yhat, prob]
                # else:            
                #     yhat = torch.argmax(rs, dim=1)
                #     yhat = yhat.detach().to('cpu').numpy()[0]
                #     output = [yhat]

                # if output_logits:
                #     output.append(rs.detach().to('cpu').numpy()) 
                # return tuple(output) if len(output) > 1 else output[0]
                

        except Exception as err:
            logger.debug(features)
            logger.debug('----------------------------------')
            logger.debug(err)
            return None

    def coerce_valid_input_(self, input):
        if not input or input is None:
            return ""
        elif not type(input) is str and math.isnan(input):
            return ""
        else:
            return input
        pass

    def build_text_from_fields_(self, batch, fields:Union[str,list]) -> list:
        if isinstance(fields, str):
            return self.coerce_valid_input_(batch[fields])
        elif (isinstance(fields, list) and len(fields)==1):
            return self.coerce_valid_input_(batch[fields[0]])
        else: #multiple fields
            return ' '.join([self.coerce_valid_input_(batch[f]) for f in fields])


    def predict(self, X:Union[list, DataFrame], features=[0], extra_features=[], output_probs=False, output_logits=False, max_length=128, **kwargs) -> list:
        """Apply the model to predict each entry

        Args:
            X (list, DataFrame): input to be predicted
            features (list): feature indices (for list) or names (for DFs, column names)
            output_probs (bool, optional): _description_. Defaults to False.

        Returns:
            list: predictions
        """
        if len(features) < 1 or len(features) > 2:
            raise Exception("1 or 2 features should be specified")
        
        # get_input = (lambda e: (e[features[0]], e[features[1]]) ) if len(features) == 2 else (lambda e: e[features[0]] if isinstance(X, DataFrame) else lambda e: e )
        get_input = (lambda e: (e[features[0]], self.build_text_from_fields_(e, features[1]) ) ) if len(features) == 2 else (lambda e: e[features[0]] if isinstance(X, DataFrame) else lambda e: e )

        extra_feat_cols = []
        if isinstance(X, DataFrame) and extra_features:
            extra_feat_cols = [c for c in X.columns if c.startswith('exf_c_') or c.startswith('exf_n_')]
        # get_extra = lambda e: e[extra_features]
        get_extra = lambda e: ({f'exf_n_{ef}': [e[ef]] for ef in extra_features} if extra_features else {}) #TODO: consider non numerical fts

        # tqdm.pandas()
        if isinstance(X, list):
            yhat = [self.predict_one( get_input(e)  , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **{**kwargs, **get_extra(e)}) for e in tqdm(X, total=len(X))] # or tuple = (e['text1'], e['text2'])
        elif isinstance(X, DataFrame):
            yhat =  [self.predict_one( get_input(e) , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **{**kwargs, **get_extra(e)}) for i,e in tqdm(X.iterrows(), total=len(X))] # or tuple = (e['text1'], e['text2'])
        else:
            raise Exception("Unhandled X's type")

        # unroll predictions if nested in dicts (happens on multitarget inference)
        # print(f"\n\n--> predict:: {type(yhat)} :: {yhat}")
        if all(type(e) is dict for e in yhat):
            # print(f"\n\n--> predict2:: ")
            yhat_dict = {}
            for e in yhat:
                for k,v in e.items():
                    yhat_dict.setdefault(k, [])
                    yhat_dict[k].append(v)
            # print(f"\n\n--> predict3:: {type(yhat_dict)} :: {yhat_dict}")
            return yhat_dict

        return yhat    
