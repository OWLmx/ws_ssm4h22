from datetime import datetime
from typing import Optional
from argparse import ArgumentParser

import datasets

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import (
    AdamW,
    AutoConfig, AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from transformers import (
    BertModel,
    DistilBertModel
)

from pytorch_lightning import LightningModule

import torchmetrics as metrics



from .model_seqcls_mtl import ModelForSequenceClassificationMtlPlusFeatures

import logging

loggerx = logging.getLogger(__name__)
loggerx.setLevel(logging.DEBUG)

import inspect

def _find_device(current_frame):
    frame = current_frame.f_back.f_back

    device = None
    while device is None:
        values = inspect.getargvalues(frame)
        device = values.locals.get("device", None)
        frame = frame.f_back

    return device

default_device = torch.device('cuda:0')

class SequenceClassifierMTL(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-3,
        warmup_steps: int = 0.0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        train_data_size = 100, # now have to explicitilly be provided because it is not longer possible to access train_loader
        num_extra_features = 0, # numerical or categorical features that are directly passed to the classification header
        **kwargs,
        # nlabels_per_task='task1|2',
    ):
        super().__init__()

        print(f"\n========= Classifier:: NumLabels [{num_labels}] ================")

        self.save_hyperparameters()
        self.num_extra_features = num_extra_features
        self.learning_rate = learning_rate
        self.t10sec = kwargs.get("t10sec", False) # flag for mini test (all working)


        self.tasks = {'task': num_labels} if num_labels.isnumeric() else {num_labels.split('|')[0]: int(num_labels.split('|')[1]) for num_labels in num_labels.split(',')}
        self.custom_model = ModelForSequenceClassificationMtlPlusFeatures(model_name_or_path=model_name_or_path, num_labels=num_labels, tasks=self.tasks, num_extra_features=num_extra_features)
        self.config = self.custom_model.config
        # Metrics
        # self.metric = datasets.load_metric("f1", experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        # self.metric_train_acc = metrics.Accuracy()    
        # self.metric_train_f1 = metrics.F1(num_classes=num_labels, average='macro') # for this task is indicated to use macro

        # self.metric_val_acc = metrics.Accuracy()    
        # self.metric_val_f1 = metrics.F1(num_classes=num_labels, average='macro') #  weighted

        # self.metric_test_acc = metrics.Accuracy()    
        # self.metric_test_f1 = metrics.F1(num_classes=num_labels, average='macro')

        self.valid_metrics = {}
        self.test_metrics = {}
        for task in self.tasks:
            metricsi = metrics.MetricCollection([metrics.Accuracy(), metrics.F1(num_classes=self.tasks[task], average='macro')])
            self.valid_metrics[task] = metricsi.clone(prefix=f'{task}_valid_')
            self.valid_metrics[task].to(default_device)
            self.test_metrics[task] = metricsi.clone(prefix=f'{task}_test_')
            self.test_metrics[task].to(default_device)

        self.train_data_size = train_data_size

        # freeze backbone N epochs
        self.nr_frozen_epochs = self.hparams.get('nr_frozen_epochs', 0)
        if self.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    # def _apply(self, fn):
    #     # to handle automated devics asignation on nested modules
    #     self._device = torch.device(_find_device(inspect.currentframe()))
    #     for module in self.children():
    #         module._apply(fn)

    def get_yhat_ytrue(self, task, batch, outputs):
        # ytrue = batch[f'labels_{task}'] if len(self.tasks) > 1 else batch['labels']
        ytrue = outputs[f'labels_{task}'] if len(self.tasks) > 1 else outputs['labels'] # use labels already used to compute the loss
        logits = outputs[f'logits_{task}'] if len(self.tasks) > 1 else outputs['logits']
        return (torch.argmax(logits, dim=1) if self.tasks[task] >= 1 else logits.squeeze(), ytrue)

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            loggerx.info(f"\n-- Encoder model fine-tuning")
            # for param in self.bert.parameters():
            for param in self.custom_model.backbone_model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        # for param in self.bert.parameters():
        for param in self.custom_model.backbone_model.parameters():
            param.requires_grad = False
        self._frozen = True


    def forward(self, **inputs):
        # loggerx.debug(f"***** Running forward:: {list(inputs.keys())}")
        
        # return self.custom_model(**inputs)
        return self.custom_model(**inputs, return_dict=True) # to handle multiple logits


    def training_step(self, batch, batch_idx):
        # loss, logits = self(**batch)[:2]
        outputs = self(**batch)
        loss = outputs['loss']

        self.log("loss", loss, on_epoch=True, sync_dist=False)
        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

        # return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"-----> VALIDATING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss = outputs['loss']

        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=False)
        for task in self.tasks:
            yhat, ytrue = self.get_yhat_ytrue(task, batch, outputs)
            metrics_output = self.valid_metrics[task](yhat, ytrue)
            self.log_dict(metrics_output)

        return loss

    def test_step(self, batch, batch_idx):
        # loggerx.info(f"\n\n-----> TESTING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss = outputs['loss']

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=False)
        collected_ouputs = {} # collect outputs for all the tasks
        for task in self.tasks:
            yhat, ytrue = self.get_yhat_ytrue(task, batch, outputs)
            metrics_output = self.test_metrics[task](yhat, ytrue)
            self.log_dict(metrics_output)

            prefix = f'{task}_' if len(self.tasks) > 1 else ''
            collected_ouputs = {**collected_ouputs, **{f'{prefix}yhat': yhat, f'{prefix}ytrue': ytrue}}

        # return {'yhat': yhat, 'ytrue': ytrue}
        return collected_ouputs

    def test_epoch_end(self, outputs): 
        # consolidate all predictions in case of further analysis outside
        # yhat = torch.cat([x["yhat"] for x in outputs]).detach().cpu()
        # ytrue = torch.cat([x["ytrue"] for x in outputs]).detach().cpu()
        # self.test_predictions = {'yhat': yhat.numpy(), 'ytrue': ytrue.numpy()}

        self.test_predictions = {} # to handle N tasks (have to include yhat or ytrue word)
        for k in outputs:
            if 'yhat' in k or 'ytrue' in k:
                y = torch.cat([x[k] for x in outputs]).detach().cpu()
                self.test_predictions[k] = y.numpy()



    def setup(self, stage=None) -> None:
        loggerx.info(f"--> SetUp Model ** {stage} :: device {self.device} [{type(self.device)}]")
        # self.custom_model.device = self.device # Not updated even at this point https://github.com/Lightning-AI/lightning/issues/13108 
        if stage != "fit":
            return

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.epoch_steps = (self.train_data_size // self.hparams.train_batch_size)
        self.total_steps = self.epoch_steps * self.trainer.max_epochs                

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        loggerx.info(f"*** Prepare optimizer and schedule (linear warmup and decay)  ***")
        # model = self.custom_model
        model = self
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        loggerx.info(f"Optimizer LR: {(self.learning_rate or self.hparams.learning_rate or self.hparams.lr)} EPS: {self.hparams.adam_epsilon}")
        optimizer = AdamW(optimizer_grouped_parameters, lr=(self.learning_rate or self.hparams.learning_rate), eps=self.hparams.adam_epsilon)
        
        if not self.hparams.warmup_steps is None and self.hparams.warmup_steps != 0:
            loggerx.info("Optimizer & Scheduler")
            # warmup_steps = int(-1*(self.hparams.warmup_steps/100) * self.total_steps) if self.hparams.warmup_steps < 0 else self.hparams.warmup_steps
            warmup_steps = int(-1*(self.hparams.warmup_steps/100) * self.total_steps) if self.hparams.warmup_steps < 0 else ((self.hparams.warmup_steps/10) * self.epoch_steps ) if self.hparams.warmup_steps <= 10 else self.hparams.warmup_steps
            loggerx.info(f"warmup_steps_original: {self.hparams.warmup_steps} | {self.total_steps}")
            loggerx.info(f"warmup_steps: {warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]        
        else: # Without scheduler
            loggerx.info(f"Only Optimizer --> {optimizer}")
            return [optimizer]

    # @staticmethod
    @classmethod
    def add_argparse_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        parser.add_argument(
            "--t10sec",
            default=-1,
            type=int,
            help="10 sec minitest, useful to check that things run",
        )        
        parser.add_argument(
            "--model_name_or_path",
            default="distilbert-base-uncased",
            type=str,
            help="Pretrained Transformer to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-5, #3e-05,
            type=float,
            help="Learning rate.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="warmup steps.",
        )                
        parser.add_argument(
            "--num_labels",
            default=2,
            type=str,
            help="How many classes are expected",
        )

        parser.add_argument(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )

        parser.add_argument("--min_epochs", default=None, type=int)
        parser.add_argument("--max_epochs", default=100, type=int)
        parser.add_argument("--adam_epsilon", default=1e-3, type=float, help="For optimizer numerical stability") #official default 1e-8
        parser.add_argument('--auto_lr_find', type=bool, default=True, help="Enable PyLighting auto find learning-rate") 
        # parser.add_argument('--nlabels_per_task', type=str, default='task1|2,task2|3', help="number of labels per task") 
        return parser        