from ast import arg
import imp
import os
from os import path, makedirs, symlink
from argparse import ArgumentParser
import pandas as pd
import torch
import pytorch_lightning as pl
import torchmetrics as metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, model_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import pandas as pd
import datetime

# from plibs.estimators.base_sequence_classifier import SequenceClassifierTransformer as Model
from plibs.estimators.sequence_classifier_plusfeats import SequenceClassifierPlusFeats as Model
from plibs.estimators.stance_datamodule import StanceDataModule as Data


AVAIL_GPUS = min(1, torch.cuda.device_count())

def dump_raw_transformer(pl_model_path) -> str:
    print(f"Converting to raw PT [{pl_model_path}]")
    _model = Model.load_from_checkpoint(pl_model_path)
    # _tgtpath = f"{pl_model_path[:-5]}_pt"
    _tgtpath = path.join(path.dirname(pl_model_path), 'best')
    makedirs(_tgtpath, exist_ok=True)
    if hasattr(_model, 'model'): # direct transformer pretrained 
        _model.model.save_pretrained(_tgtpath)
    else: # custom model
        torch.save(_model.state_dict(), path.join(_tgtpath, "pytorch_model.bin")) # "pytorch_model.pt"))
        _model.config.to_json_file(path.join(_tgtpath, 'config.json')) # TODO: this is the backbone config, not actually the right final config 

    return _tgtpath


def main(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # avoid HuggingFace warnings on Parallelization of Dataloaders

    pl.seed_everything(42)

    # pretrained_model = "distilbert-base-uncased"

    # load Data & Model
    dict_args = vars(args)    
    kwargs = {}    

    # hydrate params
    if 'fields_transformations' in dict_args:
        if dict_args['fields_transformations'] == 'text|mask_term':
            # the correct way to get the corrsponding mask token is from teh corresponding tokenizer : tokenizer.mask_token -> '[MASK]'
            dict_args['fields_transformations'] = {'text': lambda e: e.text[:e.off_ini] + '[MASK]' + e.text[e.off_end:]}

    # files bases on fold arguments
    if 'fold_data_file' in dict_args and dict_args['fold_data_file'] and 'fold_test' in dict_args and 'fold_valid' in dict_args:
        df = pd.read_csv(path.join(args.data_dirpath, args.fold_data_file))
        df[df.kfold==args.fold_test].to_pickle(path.join(args.data_dirpath, f"{args.data_filename_prefix}{args.data_split_test}.{args.data_filename_type}"))
        df[df.kfold==args.fold_valid].to_pickle(path.join(args.data_dirpath, f"{args.data_filename_prefix}{args.data_split_valid}.{args.data_filename_type}"))
        df[(df.kfold!=args.fold_valid) & (df.kfold!=args.fold_test)].to_pickle(path.join(args.data_dirpath, f"{args.data_filename_prefix}{args.data_split_train}.{args.data_filename_type}"))

    data = Data(**dict_args)
    print(f"**Training Data size: {data.train_data_size}")

    dict_args['num_labels']=data.num_labels
    try:
        dict_args['num_extra_features']= len(data.extrafeat_cat) + len(data.extrafeat_num)
        print(f"Prepared for [{dict_args['num_extra_features']}] extra features")
    except Exception as err:
        print(f"No extra features.", err)
    dict_args['train_data_size']=data.train_data_size # explicitly passed to compute num steps

    if 'use_weights_of' in dict_args and dict_args.get('use_weights_of'):
        print(f"--> reusing model from {dict_args.get('use_weights_of')}")
        model = Model.load_from_checkpoint(dict_args.get('use_weights_of'))
    else:
        model = Model(**dict_args)

    if 'resume_from_ckpt' in dict_args and dict_args.get('resume_from_ckpt') and len(dict_args.get('resume_from_ckpt')) > 0:
        kwargs['resume_from_checkpoint'] = dict_args.get('resume_from_ckpt')

    # Configure Trainer & its callbacks
    callbacks=[]
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='model_{epoch:02d}_{step:02d}_{validation_loss:.2f}', # fixed for automated scripts
        auto_insert_metric_name=False, # not include metric name because of undesired '=' symbols
        monitor='validation_loss', verbose=True,
        save_last=False, save_top_k=1, save_weights_only=False,
        mode='min', every_n_epochs=1) 

    callbacks.append( checkpoint_callback)
    callbacks.append( EarlyStopping(monitor='validation_loss', patience=dict_args.get('patience') ) )    
    callbacks.append( LearningRateMonitor(logging_interval='step') )

    # Custom logger
    # get current run's id (if run from guild)
    run_name = None
    if ".guild/runs/" in os.getcwd():
        try:
            run_name = os.getcwd().partition(".guild/runs/")[2].split(path.sep)[0][:8] # shortid used to name the tensorboard version
            run_name = f"{datetime.datetime.utcnow().strftime('%m%d%H%M%S')}_{run_name}" # add time stamp to ensure proper order in TB
        except:
            pass
    
    pl_logger_path = os.path.join(os.getenv("PRJ_HOME", "."), dict_args.get("logs_path"))
    print(f"Configuring TensorLogger output to --> {pl_logger_path}")
    makedirs(pl_logger_path, exist_ok=True) # now it complains if does not exist
    logger = TensorBoardLogger(pl_logger_path, name="evcls", version=run_name) # log path relative to the Project's home

    # trainer = pl.Trainer(max_epochs=1, callbacks=callbacks, gpus=AVAIL_GPUS)
    trainer = pl.Trainer.from_argparse_args(args, gpus=AVAIL_GPUS, callbacks=callbacks, logger=logger, **kwargs)

    # --------- Find a good LR ----------
    if args.auto_lr_find:
        print(" *** tuning trainer, Finding LR ***")
        # also possible to do
        # lr_finder = trainer.tune(model, datamodule=data)["lr_find"]  #  update_attr is set to True by default  
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, datamodule=data, update_attr=True)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("auto_lr_find.png")
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(f"Auto_LR: {new_lr}")
        # update hparams of the model # Updated by the tunner so optimizers are also upodated
        # model.hparams.lr = new_lr
        # model.hparams.learning_rate = new_lr
    # // --------- Find a good LR ----------


    # Do training
    trainer.fit(model, data) # ckpt_path (Optional[str]) – Path/URL of the checkpoint from which training is resumed

    # SAVE:: convert best checkpoint to raw Torch model to be used with HugginFace without PL
    _tgtpath = dump_raw_transformer(checkpoint_callback.best_model_path)    
    # symlink(checkpoint_callback.best_model_path, path.join(path.basename(path.normpath(checkpoint_callback.best_model_path)), 'best') ) # alias to best ckpt
    symlink(checkpoint_callback.best_model_path, path.join(path.dirname(checkpoint_callback.best_model_path), 'best.ckpt') ) # alias to best ckpt for script automation

    # dump label_encoder
    if data.label_encoder:
        with open(f"{_tgtpath}/label_encoder.pkl", 'wb') as pickle_file:
            pickle.dump(data.label_encoder, pickle_file )

    # test the trained model
    print("\n=========  TESTING  ==========")    
    # rs = trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=data, verbose=True)
    # rs = trainer.test()
    rs = trainer.test(datamodule=data) # now data has to be explicitly passed
    for r in rs:
        for k,v in r.items():
            print(f"{k}: {v}")
    
    print(f"... Saving predictions to [{'test_predictions.csv'}]")
    # pd.DataFrame(model.test_predictions).to_csv('test_predictions.csv', index=False)
    preds= pd.DataFrame(model.test_predictions)
    if data.label_encoder: # apply label encoder if provided
        preds['ytrue_label'] = data.label_encoder.inverse_transform( preds.ytrue.values )
        preds['yhat_label'] = data.label_encoder.inverse_transform( preds.yhat.values )        
    preds.to_csv('test_predictions.csv', index=False)
    # print(f"f1: {metrics.functional.f1(preds=model.test_predictions['yhat'], target=model.test_predictions['ytrue'], average='weighted', num_classes=data.num_labels).detach().cpu().numpy()}")



if __name__ == "__main__":
  
    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster
    # parser = ArgumentParser(conflict_handler='resolve', add_help=False) # add_help=False to be used in cluster
 
    # parser.add_argument('--training_portion', type=float, default=0.9)    
    # parser.add_argument('--strategy', type=str, default="ddp") # for use in cluster
    parser.add_argument('--logs_path', type=str, default="logs/") 
    parser.add_argument('--use_weights_of', type=str, default="") 
    parser.add_argument('--patience', type=int, default=3) 
    parser.add_argument('--auto_lr_find', type=bool, default=False, help="Enable PyLighting auto find learning-rate") 

    parser.add_argument('--resume_from_ckpt', type=str, default="", help="If specified training resume from this checkpoint") 

    # parser.add_argument('--fold_data_file', type=str, default="", help="Data file with kfolds, generate splits on the fly if provided") 
    # parser.add_argument('--fold_test', type=int, default=6) 
    # parser.add_argument('--fold_valid', type=int, default=4) 
 
    parser = Data.add_argparse_args(parser)    
    parser = Model.add_argparse_args(parser) 
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()        
    

    if args.t10sec is not None and (args.t10sec or args.t10sec > 1):
        print("==> Setting [max_epochs] to 1 due to t10sec var")
        args.max_epochs = 1
 
    main(args)