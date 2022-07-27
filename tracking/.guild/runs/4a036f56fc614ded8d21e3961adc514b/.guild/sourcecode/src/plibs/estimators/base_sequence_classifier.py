from datetime import datetime
from typing import Optional
from argparse import ArgumentParser

import datasets

import torch
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from pytorch_lightning import LightningModule

import torchmetrics as metrics

class SequenceClassifierTransformer(LightningModule):
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

        print(f"\n========= Classifier:: NumLabels [{num_labels}] ================")

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.t10sec = kwargs.get("t10sec", False) # flag for mini test (all working)

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)        

        # Metrics
        self.metric = datasets.load_metric("f1", experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        self.metric_train_acc = metrics.Accuracy()    
        # self.metric_train_f1 = metrics.F1()
        self.metric_train_f1 = metrics.F1(num_classes=num_labels, average='macro') # for this task is indicated to use macro

        self.metric_val_acc = metrics.Accuracy()    
        self.metric_val_f1 = metrics.F1(num_classes=num_labels, average='macro') #  weighted

        self.metric_test_acc = metrics.Accuracy()    
        self.metric_test_f1 = metrics.F1(num_classes=num_labels, average='macro')

        self.train_data_size = train_data_size

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # outputs = self(**batch) # (loss, logits ...)
        # loss = outputs[0] 
        loss, logits = self(**batch)[:2]

        # log_dict = {'train_loss': loss}
        # return {'loss': loss, 'log': log_dict}
        # self.log("loss", loss) 
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"-----> VALIDATING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            yhat = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            yhat = logits.squeeze()

        ytrue = batch["labels"]

        # in case of parallel mode dp        
        # log_dict = {'val_loss': loss}
        # batch_dict = {"loss": loss, "yhat": yhat, "ytrue": ytrue, 'log': log_dict}
        # return batch_dict
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        acc = self.metric_val_acc(yhat, ytrue)
        self.log("validation_acc", self.metric_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        f1 = self.metric_val_f1(yhat, ytrue)        
        self.log('validation_f1', self.metric_val_f1, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    # def validation_step_end(self, outputs):
    #     #update and log (in case of parallel mode dp)
    #     acc = self.metric_val_acc(outputs['yhat'], outputs['ytrue'])        
    #     self.log('val_acc', self.metric_val_acc)

    #     f1 = self.metric_val_f1(outputs['yhat'], outputs['ytrue'])        
    #     self.log('val_f1', self.metric_val_f1)


    # def validation_epoch_end(self, outputs):
    #     # aggregate outputs of each validation step
    #     yhat = torch.cat([x["yhat"] for x in outputs]).detach().cpu().numpy()
    #     ytrue = torch.cat([x["ytrue"] for x in outputs]).detach().cpu().numpy()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()        

    #     # self.log_dict(self.metric.compute(predictions=preds, references=labels, average="weighted"), prog_bar=True) # HF metrics

    #     self.log("val_loss", loss, prog_bar=True)

    #     self.metric_val_acc(yhat, ytrue)
    #     self.log("val_acc", self.metric_val_acc)
        
    #     return loss

    def test_step(self, batch, batch_idx):
        # print(f"\n\n-----> TESTING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            yhat = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            yhat = logits.squeeze()

        ytrue = batch["labels"]

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.metric_test_acc(yhat, ytrue)
        self.log("test_acc", self.metric_test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.metric_test_f1(yhat, ytrue)        
        self.log('test_f1', self.metric_test_f1, on_step=False, on_epoch=True, sync_dist=True)

        return {'yhat': yhat, 'ytrue': ytrue}


    def test_epoch_end(self, outputs): 
        # consolidate all predictions in case of further analysis outside
        yhat = torch.cat([x["yhat"] for x in outputs]).detach().cpu()
        ytrue = torch.cat([x["ytrue"] for x in outputs]).detach().cpu()
        self.test_predictions = {'yhat': yhat.numpy(), 'ytrue': ytrue.numpy()}

    def setup(self, stage=None) -> None:
        print("--> SetUp")
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        # train_loader = self.train_dataloader() # DEPRECATED , it is no longer possible to directly access it 

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        # self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size
        # self.total_steps = (len(train_loader.dataset) // self.hparams.train_batch_size) * self.trainer.max_epochs
        # self.total_steps = (self.train_data_size // self.hparams.train_batch_size) * self.trainer.max_epochs
        self.epoch_steps = (self.train_data_size // self.hparams.train_batch_size)
        self.total_steps = self.epoch_steps * self.trainer.max_epochs        

        # num_warmup_steps = (total_samples // bs) * 20
        # num_training_steps = (total_samples // bs) * n_epochs        
        
        # print(f"--> setup: self.total_steps = ({len(train_loader.dataset)} // {tb_size}) // {ab_size} => {self.total_steps}")
        # print(f"--> setup: num_training_steps = ({len(train_loader.dataset)} // {self.hparams.train_batch_size}) * {self.trainer.max_epochs} => {(len(train_loader.dataset) // self.hparams.train_batch_size) * self.trainer.max_epochs}")
        

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        print(f"*** Prepare optimizer and schedule (linear warmup and decay)  ***")
        model = self.model
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