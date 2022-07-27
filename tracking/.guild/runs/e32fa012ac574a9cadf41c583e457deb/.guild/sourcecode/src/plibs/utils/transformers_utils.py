from cmath import isnan
from os import path, listdir
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
        
        if custom_model_arch:
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
                rs = self.model(**features)[0]
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
        get_extra = lambda e: e[extra_features]

        # tqdm.pandas()
        if isinstance(X, list):
            yhat = [self.predict_one( get_input(e)  , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **kwargs) for e in tqdm(X, total=len(X))] # or tuple = (e['text1'], e['text2'])
        elif isinstance(X, DataFrame):
            yhat =  [self.predict_one( get_input(e) , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **kwargs) for i,e in tqdm(X.iterrows(), total=len(X))] # or tuple = (e['text1'], e['text2'])
        else:
            raise Exception("Unhandled X's type")

        return yhat    

        
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from tqdm import tqdm
# from functools import partial
# from typing import Union
# from pandas import DataFrame
# from os import path, listdir

# class TransformerClassifierInferenceWrapper():
#     """Wrapper for doing ONLY inference using a trained Classifier Transformer model
#     """

#     def __init__(self, model_path:str, tokenizer_type='self', custom_model_arch = None, **kwargs) -> None:
#         """

#         Args:
#             model_path (str): path to the trained model (where .bin and config.json are)
#             tokenizer_type (str, optional): the tokenizer to use (e.g., bert-base-uncvased, if 'self' loads from model path, ). Defaults to 'self'.
#         """
        
#         if custom_model_arch:
#             assert tokenizer_type!='self', "A tokenizer type has to be explicitly specified when using a custom modiel architecture"
                
#             model_pathx = [path.join(model_path, f) for f in listdir(model_path) if f.endswith('.pt')][0] if not path.isfile(model_path) else model_path
#             print(f"Loading model from: {model_pathx}")
#             self.model = custom_model_arch
#             self.model.load_state_dict(torch.load(model_pathx))
#             self.model.eval()
#         else:
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)

#         self.tokenizer = AutoTokenizer.from_pretrained(model_path if tokenizer_type=='self' else tokenizer_type, use_fast=True)                                        
#         self.kwargs = kwargs

#     def predict_one(self, inputs, output_probs=False, output_logits=False, max_length=128, **kwargs) -> Union[int, tuple]:
#         """Predict class using model

#         Args:
#             inputs (text, tuple(text1, text2)): input for the estimator
#             output_probs (bool, optional): If True the prediction "probability" is included. Defaults to False.

#         Returns:
#             (int, tuple): predicted class or tuple(predicted class, probability)
#         """
#         features = self.tokenizer.encode_plus(inputs, max_length=max_length, padding='max_length', return_tensors="pt", truncation=True)
#         try:
#             features.update(kwargs) # include extra args (e.g. extra feats exf_n_ or exf_c_ )
#             with torch.no_grad():
#                 rs = self.model(**features)[0]
#                 if rs.dim() ==1 :
#                     rs = rs.unsqueeze(0)
#                 # print(f"1************** {rs} --> {rs.shape}")
#                 rs = torch.softmax(rs, dim=1)
#                 # print(f"2************** {rs}")
#                 if output_probs:
#                     prob = torch.max(rs, dim=1)
#                     yhat = prob.indices.detach().numpy()[0]
#                     prob = prob.values.detach().numpy()[0]
#                     output = [yhat, prob]
#                 else:            
#                     yhat = torch.argmax(rs, dim=1)
#                     yhat = yhat.numpy()[0]
#                     output = [yhat]

#                 if output_logits:
#                     output.append(rs.detach().numpy()) 
#                 return tuple(output) if len(output) > 1 else output[0]
#         except Exception as err:
#             print(features)
#             print('----------------------------------')
#             print(err)
#             return None

#     def predict(self, X:Union[list, DataFrame], features=[0], extra_features=[], output_probs=False, output_logits=False, max_length=128, **kwargs) -> list:
#         """Apply the model to predict each entry

#         Args:
#             X (list, DataFrame): input to be predicted
#             features (list): feature indices (for list) or names (for DFs, column names)
#             output_probs (bool, optional): _description_. Defaults to False.

#         Returns:
#             list: predictions
#         """
#         if len(features) < 1 or len(features) > 2:
#             raise Exception("1 or 2 features should be specified")
        
#         get_input = (lambda e: (e[features[0]], e[features[1]]) ) if len(features) == 2 else (lambda e: e[features[0]] if isinstance(X, DataFrame) else lambda e: e )

#         extra_feat_cols = []
#         if isinstance(X, DataFrame) and extra_features:
#             extra_feat_cols = [c for c in X.columns if c.startswith('exf_c_') or c.startswith('exf_n_')]
#         get_extra = lambda e: e[extra_features]

#         # tqdm.pandas()
#         if isinstance(X, list):
#             yhat = [self.predict_one( get_input(e)  , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **kwargs) for e in tqdm(X, total=len(X))] # or tuple = (e['text1'], e['text2'])
#         elif isinstance(X, DataFrame):
#             yhat =  [self.predict_one( get_input(e) , output_probs=output_probs, output_logits=output_logits, max_length=max_length, **kwargs) for i,e in tqdm(X.iterrows(), total=len(X))] # or tuple = (e['text1'], e['text2'])
#         else:
#             raise Exception("Unhandled X's type")

#         return yhat    