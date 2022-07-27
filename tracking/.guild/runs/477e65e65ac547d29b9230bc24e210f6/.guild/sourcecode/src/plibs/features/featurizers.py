from numpy import ma
import pandas as pd
import numpy as np
from sentence_transformers.SentenceTransformer import SentenceTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_random_state


class BasicFeaturizer(BaseEstimator, TransformerMixin):


    def __init__(self, transformer, random_state=None, fields_to_keep=[]):
        
        self.transformer = ColumnTransformer( [ 
             ('label', 'passthrough', ['label']) # keep target value for training step
        ], remainder='drop')

        if fields_to_keep: # If other fields were specified to be kept in the features
            self.transformer.transformers.extend( [(c.strip(), 'passthrough', [c.strip()]) for c in fields_to_keep if c.strip()] )

        print(self.transformer)
        self.random_state = random_state
    
    def fit(self, X, y=None):
        #Nothing to do here, 
        #self.transformer.fit(self._prepare( X ) )
        self.random_state_ = check_random_state(self.random_state)
        return self
    
    def transform(self, X): 

        if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            rs = self.transformer.fit_transform(self._prepare( X ) )
        else:
            return None

        # assign column names to transformed data
        cols = [  transformer_name for transformer_name, _, _ in self.transformer.transformers_]
        rs = pd.DataFrame(rs, columns = cols[:rs.shape[1]])

        return rs

    def _prepare(self, X):
        # Needed composed columns for some trasnfromations
        pass    

        return X

class STSEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, transformer:SentenceTransformer) -> None:
        super().__init__()
        self.transformer = transformer

    def fit(self, X, y=None):
        #Nothing to do here, 
        # self.random_state_ = check_random_state(self.random_state_)
        return self

    def transform(self, X): 
        
        rs = None
        if isinstance(X, pd.DataFrame):
            rs = {}
            for c in X.columns:
                embs = self.transformer.encode(list(X[c]), convert_to_tensor=True, show_progress_bar=True)
                rs[c] = embs            
        elif isinstance(X, list) and X and 'str' in str(type(X[0])): # single element list expected
            embs = self.transformer.encode(list(X), convert_to_tensor=True, show_progress_bar=True)
            rs = embs
        else:
            raise Exception("Not accepted format")

        return rs        