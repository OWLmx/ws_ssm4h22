from ast import Try
from typing import Union
import logging as log
from os import path, sep
import pandas as pd
import numpy as np
import datasets
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch
from transformers import ( AutoTokenizer )

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

import logging

logger = logging.getLogger(__name__)

class StanceDataModule(LightningDataModule):

    task_text_field_map = {
        "base": ["Claim", "tweet_text_clean"],
        "sentclaim": ["Claim2", "tweet_text_clean"],
        "covidlies": ["misconception", "tweet_text_clean"]
    }

    task_label_field_map = {
        "base": ["labels"],
        "sentclaim": ["labels"],
        "covidlies": ["labels"],
    }    

    task_extrafeat_cat = {
    }

    task_extrafeat_num = {
        # "text_extrafeat_pos10": ["i10th_pos"],
    }

    task_num_labels = {
        "base": 3, # 'AGAINST', 'FAVOR', 'NONE'
        "sentclaim": 3,
        "covidlies": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "base",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        stratified_batch_sampling = False, # include replacement -> oversampling
        fields_transformations = {}, # column_name : func to transform its data
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.text_fields = self.task_text_field_map[task_name]
        self.label_fields = self.task_label_field_map[task_name] if task_name in self.task_label_field_map else list(self.task_label_field_map.values())[0]
        self.extrafeat_cat = self.task_extrafeat_cat[task_name] if task_name in self.task_extrafeat_cat else []
        self.extrafeat_num = self.task_extrafeat_num[task_name] if task_name in self.task_extrafeat_num else []
        self.tokenizer = kwargs.get("tokenizer", AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True))
        self.stratified_batch_sampling = stratified_batch_sampling
        self.fields_transformations =  {} if fields_transformations is None or len(fields_transformations.strip())==0 else fields_transformations

        self.t10sec = kwargs.get("t10sec", False) # flag for mini test (all working)
        if self.t10sec:
            print("*** t10sec")
            logger.setLevel(logging.DEBUG)
            if self.t10sec==1 or not isinstance(self.t10sec, (int, float)): # default num value if t10sec = True
                self.t10sec = 100

        # define label encoder & number of classes
        self.dataset = kwargs.get("dataset", None) # if dataset is provided (already loaded)
        self.data_dirpath = kwargs.get("data_dirpath", "")
        self.data_filename_prefix = kwargs.get("data_filename_prefix", "")

        _train_data = self.read_data(train=self.hparams.data_split_train, val=None, test=None)['train'] # preload for checking size and labels
        self.train_data_size = len(_train_data)
        if self.t10sec:
            self.train_data_size = self.t10sec
        if "label_encoder" in kwargs: #.get("label_encoder", None):
            self.label_encoder = kwargs.get("label_encoder", None)
        else:
            self.label_encoder = LabelEncoder()            
            # self.label_encoder.fit( self.read_data(train=None, val=self.hparams.data_split_valid, test=None)['validation'][self.label_fields[0]]) # no need to load full train split
            self.label_encoder.fit( _train_data[self.label_fields[0]]) 

        # label encoders for categorical extra features 
        self.extrafeat_cat_labelencoder = {} 
        if self.extrafeat_cat:       
            for f in self.extrafeat_cat:
                self.extrafeat_cat_labelencoder[f] = LabelEncoder()
                self.extrafeat_cat_labelencoder[f].fit(_train_data[f])

        # scalers for numerical data (put everything in the [0..1] range for proper gradient comp.)
        self.extrafeat_num_scaler = {}
        if self.extrafeat_num:
            for f in self.extrafeat_num:
                self.extrafeat_num_scaler[f] = MinMaxScaler()
                self.extrafeat_num_scaler[f].fit(np.reshape(_train_data[f], (-1, 1)) )

        if self.label_encoder:
            self.num_labels = len(self.label_encoder.classes_) if self.label_encoder else (self.task_num_labels[task_name] if task_name in self.task_num_labels else list(self.task_num_labels.values())[0])
        else:
            try:
                self.num_labels = len(set(_train_data[self.label_fields[0]])) 
            except Exception as err:
                self.num_labels = (self.task_num_labels[task_name] if task_name in self.task_num_labels else list(self.task_num_labels.values())[0])
                pass



    def read_data(self, train='train', val='val', test='test', sample_size=-1):
        if self.dataset and \
            all([
                ((train and 'train' in self.dataset) or train is None),
                ((val and 'val' in self.dataset) or val is None),
                ((test and 'test' in self.dataset) or test is None)
            ]) : # already loaded (possibly passed in init)
            return self.dataset

        print("Reading dataset.")        
        # log.info(f"Reading dataset ...")
        dataset = datasets.DatasetDict()
        data_files = {}
        if train:
            data_files['train']= path.join(self.data_dirpath, f"{self.data_filename_prefix}{train}.{self.hparams.data_filename_type}")
        if val:
            data_files['validation']= path.join(self.data_dirpath, f"{self.data_filename_prefix}{val}.{self.hparams.data_filename_type}")
        if test:
            data_files['test']= path.join(self.data_dirpath, f"{self.data_filename_prefix}{test}.{self.hparams.data_filename_type}")
        
        # dataset = datasets.load_dataset('csv', data_files=data_files) # load as hugginface arrow datasets with cache handling
        dataset = datasets.DatasetDict({ k : datasets.Dataset.from_pandas( self.read_dataset(data_files[k], sample_size=sample_size) ) for k in data_files }) # load as pandas dataframe                

        # print(dataset)
        return dataset

    def read_dataset(self, path: str, as_dataframe=True, sample_size=-1):
        """ Reads a comma separated value file.
        :param path: path to a csv file.
        
        :return: List of records as dictionaries
        """
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t')            
        elif path.endswith('.pkl'):
            df = pd.read_pickle(path)
        else:
            raise("Unhandled format")

        # common preprocess
        # df = df.dropna().sample(n=sample_size) if sample_size > 0 else df.dropna()
        df = df.sample(n=sample_size) if sample_size > 0 else df # now there are columns with Nones

        # apply transformations that need other fields later discarded
        for field, transformation in self.fields_transformations.items():
            # if field in df.columns: # to just transform existimg fields
            logger.info(f"Applying transformation to [{field}]")
            df[field]= df.apply(transformation, axis=1)

        text_fields_flat_ = sum([f if type(f) is list else [f] for f in self.text_fields], []) # flattize nested fields (for multi field input building)
        df = df[text_fields_flat_ + self.label_fields +  self.extrafeat_cat + self.extrafeat_num]
        for f in (text_fields_flat_):
            df[f] = df[f].astype(str)
        if as_dataframe:
            return df
        else:
            return df.to_dict("records")
        
    def setup(self, stage: str):
        log.info(f"... Setup [{stage}]")
        if stage == 'test':
            self.dataset = self.read_data(train=None, val=None, test=self.hparams.data_split_test, sample_size=self.t10sec if self.t10sec else -1)            
        else:
            self.dataset = self.read_data(train=self.hparams.data_split_train, val=self.hparams.data_split_valid, test=None, sample_size=self.t10sec if self.t10sec else -1)            

        for split in self.dataset.keys():
            log.info(f"\n---- Mapping split [{split}] -----")
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns= self._cols_to_remove(self.dataset[split]), ## ['label']
                # load_from_cache_file=False, keep_in_memory=False # No cache
            )
            # self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns or (c.startswith('exf_c_') or c.startswith('exf_n_')) ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if any(v in x for v in ["validation", "test"]) ]

    def _cols_to_remove(self, ds):
        # labels columns (original or renamed whatever exists)
        return list(set(ds.column_names).intersection(set(self.task_label_field_map[self.task_name] + ['labels'])))        

    def prepare_data(self):
        # self.dataset = datasets.load_dataset('csv', data_files={'train': base_path + 'test.csv', 'validation': base_path + 'val.csv', 'test': base_path + 'test.csv'})
        # self.dataset = self.read_data(sample_size=100 if self.t10sec else -1)
        # AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        pass

    def train_dataloader(self):
        if self.stratified_batch_sampling:
            logger.info("Using stratified batch sampling for training DataLoader.")
            # weighted sampler with replacement for stratisfied oversample
            y_train = self.dataset["train"]['labels'].numpy()
            count=Counter(y_train)
            count = [count[k] for k in range(len(count))]    
            class_count=np.array(count)
            weight=1./class_count
            samples_weight = np.array([weight[t] for t in y_train])
            samples_weight=torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

            # return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, num_workers=self.hparams.loader_workers)
            return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, num_workers=self.hparams.loader_workers, sampler=sampler)
        else:
            return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, num_workers=self.hparams.loader_workers)


    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.hparams.loader_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.hparams.loader_workers) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.hparams.loader_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.hparams.loader_workers) for x in self.eval_splits]

    def build_text_from_fields_(self, batch, fields:Union[str,list]) -> list:
        if isinstance(fields, str):
            return batch[fields]
        elif (isinstance(fields, list) and len(fields)==1):
            return batch[fields[0]]
        else: #multiple fields
            # rs = []
            # for p in zip(*[batch[f] for f in fields]):
            #     rs.append( ' '.join([str(t) if not t is None else "" for t in p]) ) # joined text
            rs = [ ' '.join([str(t) if not t is None else "" for t in p]) for p in zip(*[batch[f] for f in fields]) ] # joined text
            return rs

    def convert_to_features(self, example_batch, indices=None):

        # logger.debug(f"converting to features --> {example_batch}")

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            # texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]],  self.build_text_from_fields_(example_batch, self.text_fields[1])  ) )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )    

        # Rename label feat to labels to make it easier to pass to model forward
        features["labels"] = example_batch[self.task_label_field_map[self.task_name][0]]

        # pass extra features (categorical & numerical) # normalize extra features so within 0-1 range
        for f in self.extrafeat_cat:
            features[f"exf_c_{f}"] = int(self.extrafeat_cat_labelencoder[f].transform( example_batch[f] )) / len(self.extrafeat_cat_labelencoder[f].classes_)
        for f in self.extrafeat_num:
            features[f"exf_n_{f}"] = self.extrafeat_num_scaler[f].transform( np.reshape( example_batch[f], (-1,1)) ).reshape(-1).round(2)


        if self.label_encoder: # if a label encoder was provided use it to encode the labels
            features["labels"] = self.label_encoder.transform(features["labels"])


        # logger.debug(f"\t --> {features}")

        return features

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--t10sec",default=-1,type=int,help="10 sec minitest for sanity check")        
        parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
        parser.add_argument("--task_name", default='base', type=str, help="Task from where config will be used")        
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--data_dirpath", type=str, default="", help="Path of the directory where the datafiles are located")
        parser.add_argument("--data_filename_prefix", type=str, default="", help="Datafile's name without the suffix related to the split identification.")
        parser.add_argument("--data_filename_type", type=str, default="csv", help="CSV | TSV | PKL (dataframe) ")
        parser.add_argument("--data_split_train", type=str, default="train", help="Suffix that identfies the split, if None the split won't be used")
        parser.add_argument("--data_split_valid", type=str, default="valid", help="Suffix that identfies the split, if None the split won't be used")
        parser.add_argument("--data_split_test", type=str, default="test", help="Suffix that identfies the split, if None the split won't be used")
        parser.add_argument("--loader_workers",default=8,type=int, help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.")
        parser.add_argument("--stratified_batch_sampling", type=bool, default=False, help="Uses stratified batch smapling with replacement for training")
        parser.add_argument("--fields_transformations", type=str, default="", help="Dictionary where key is the field and the value the transformation to be applied")

        return parent_parser        

