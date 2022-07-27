from distutils.command.config import config
from typing import Optional, Dict
from argparse import ArgumentParser

import torch
from torch import nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
    BertModel,
)
from transformers.modeling_outputs import  TokenClassifierOutput

from pytorch_lightning import LightningModule
import torchmetrics as metrics

# from torchcrf import CRF


class TokenClassifierTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-3,
        warmup_steps: int = 0.0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        train_data_size = 100, # now have to explicitilly be provided because it is not longer possible to access train_loader
        **kwargs,
    ):
        super().__init__()

        print(f"\n=========  NumLabels [{num_labels}] ================")

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.train_data_size = train_data_size

        self.t10sec = kwargs.get("t10sec", False) # flag for mini test (all working)

        # define model
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, add_pooling_layer=False)
        # self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=self.config)        
        # self.backbone_model = BertModel(self.config, add_pooling_layer=False)
        self.backbone_model = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        self.classification_head = nn.Sequential(
            nn.Dropout(0.1), #   self.config.hidden_dropout_prob | self.config.dropout  | 0.1, ),
            nn.Linear(self.config.hidden_size, num_labels)
        )
        # self.backbone_model.init_weights()
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.configure_metrics_()
        

    def configure_metrics_(self) -> None:
        self.metrics = {}
        for mode in ['validation', 'test']:
            self.metrics[f"{mode}_accuracy"] = metrics.Accuracy()
            self.metrics[f"{mode}_precision"] = metrics.Precision(num_classes=self.num_labels)
            self.metrics[f"{mode}_recall"] = metrics.Recall(num_classes=self.num_labels)
            self.metrics[f"{mode}_f1"] = metrics.F1(num_classes=self.num_labels, average='macro')

        self.metric_log_params = {
            'train': {'on_step':True, 'on_epoch':True, 'sync_dist':False},
            'validation': {'on_step':True, 'on_epoch':True, 'sync_dist':False},
            'test': {'on_step':False, 'on_epoch':True, 'sync_dist':False},
        }
        

    def compute_metrics_and_log_(self, loss, mode:str, predictions=None, labels=None, **log_params) -> Dict[str, torch.Tensor]:
        # log loss
        self.log(f"{mode}_loss", loss, {**self.metric_log_params[mode],  **log_params})
        # get other configured metrics
        metrics_configured_for_mode = [(k, metric) for k, metric in self.metrics.items() if k.startswith(f"{mode}_")]
        if metrics_configured_for_mode and not predictions is None and not labels is None :
            # Remove ignored index (special tokens)
            active_labels = labels.view(-1) != -100 # shape (batch_size, seq_len)
            predictions = predictions[active_labels] #[labels != -100]
            labels = labels [active_labels] #[labels != -100]
            # update metric and log
            for k , metric in metrics_configured_for_mode:
                metric(predictions, labels)
                self.log(k, metric, {**self.metric_log_params[mode],  **log_params})

    def forward(self, return_dict=None, **inputs):

        labels = inputs.get('labels')
        attention_mask = inputs.get('attention_mask')

        outputs = self.backbone_model(**inputs)
        sequence_output = outputs[0]
        logits = self.classification_head(sequence_output)

        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                print(f"attention_mask --> {attention_mask}")
                ##active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
                #labels = torch.masked_select(flattened_targets, active_accuracy)
                #predictions = torch.masked_select(flattened_predictions, active_accuracy)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                print(f"active_logits --> {active_logits}")
                print(f"labels --> {labels}")
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.criterion.ignore_index).type_as(labels)
                )
                print(f"active_labels --> {active_labels}")
                loss = self.criterion(active_logits, active_labels)
            else:
                loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )        

    def training_step(self, batch, batch_idx):
        # outputs = self(**batch) # (loss, logits ...)
        loss, logits = self(**batch)[:2]

        self.compute_metrics_and_log_(loss, "training")

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"-----> VALIDATING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        yhat = torch.argmax(logits, dim=1)  #TODO should be in dim 2?
        print(f"computing yhat = [{yhat}] --> {logits}")
        ytrue = batch["labels"]

        self.compute_metrics_and_log_(loss, "validation", yhat, ytrue)

        return loss


    def test_step(self, batch, batch_idx):
        # print(f"\n\n-----> TESTING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        yhat = torch.argmax(logits, dim=1) #TODO should be in dim 2?
        ytrue = batch["labels"]

        self.compute_metrics_and_log_(loss, "test", yhat, ytrue )

        return {'yhat': yhat, 'ytrue': ytrue}


    def test_epoch_end(self, outputs): 
        # consolidate/acumulate all predictions in case of further analysis outside
        yhat = torch.cat([x["yhat"] for x in outputs]).detach().cpu()
        ytrue = torch.cat([x["ytrue"] for x in outputs]).detach().cpu()
        self.test_predictions = {'yhat': yhat.numpy(), 'ytrue': ytrue.numpy()}

    def setup(self, stage=None) -> None:
        print("--> SetUp")
        if stage != "fit":
            return

        # Calculate total steps
        self.epoch_steps = (self.train_data_size // self.hparams.train_batch_size)
        self.total_steps = self.epoch_steps * self.trainer.max_epochs        
        

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        print(f"*** Prepare optimizer and schedule (linear warmup and decay)  ***")
        model = self.backbone_model
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
        print(f"Optimizer LR: {(self.learning_rate or self.hparams.learning_rate or self.hparams.lr)} EPS: {self.hparams.adam_epsilon}")
        optimizer = AdamW(optimizer_grouped_parameters, lr=(self.learning_rate or self.hparams.learning_rate), eps=self.hparams.adam_epsilon)
        
        if not self.hparams.warmup_steps is None and self.hparams.warmup_steps != 0:
            print("Optimizer & Scheduler")
            # warmup_steps = int(-1*(self.hparams.warmup_steps/100) * self.total_steps) if self.hparams.warmup_steps < 0 else self.hparams.warmup_steps
            warmup_steps = int(-1*(self.hparams.warmup_steps/100) * self.total_steps) if self.hparams.warmup_steps < 0 else ((self.hparams.warmup_steps/10) * self.epoch_steps ) if self.hparams.warmup_steps <= 10 else self.hparams.warmup_steps
            print(f"warmup_steps_original: {self.hparams.warmup_steps} | {self.total_steps}")
            print(f"warmup_steps: {warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]        
        else: # Without scheduler
            print("Only Optimizer")
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
            default=100,
            type=int,
            help="warmup steps.",
        )                
        parser.add_argument(
            "--num_labels",
            default=2,
            type=int,
            help="How many classes are expected",
        )
        parser.add_argument("--min_epochs", default=None, type=int)
        parser.add_argument("--max_epochs", default=100, type=int)
        parser.add_argument("--adam_epsilon", default=1e-3, type=float, help="For optimizer numerical stability") #official default 1e-8
        parser.add_argument('--auto_lr_find', type=bool, default=True, help="Enable PyLighting auto find learning-rate") 
        return parser        