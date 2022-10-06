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

class SequenceClassifierPlusFeats(LightningModule):
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
        num_extra_features = 0, # numerical or categorical features that are directly passed to the classification header
        **kwargs,
    ):
        super().__init__()

        print(f"\n========= Classifier:: NumLabels [{num_labels}] ================")

        self.save_hyperparameters()
        self.num_extra_features = num_extra_features

        self.learning_rate = learning_rate

        self.t10sec = kwargs.get("t10sec", False) # flag for mini test (all working)

        self.config = kwargs['config'] if 'config' in kwargs and kwargs['config'] is not None else None
        self.backbone_model = kwargs['backbone_model'] if 'backbone_model' in kwargs and kwargs['backbone_model'] is not None else None        
        self._build_model(model_name_or_path, num_labels=num_labels)

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

        # freeze backbone N epochs
        self.nr_frozen_epochs = self.hparams.get('nr_frozen_epochs', 0)
        if self.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    def _build_model(self, model_name_or_path, num_labels, **kwargs) -> None:

        if not self.config:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        if not self.backbone_model:
            self.backbone_model = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # validd backbone forward params (from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/distilbert/modeling_distilbert.py#L691)
        self.backbone_valid_inputs = ['input_ids', 'attention_mask', 'head_mask', 'inputs_embeds', 'output_attentions', 'output_hidden_states', 'return_dict']
        # TODO: Bert have more accepted params in the forward

        self.num_labels = num_labels

        # if isinstance(self.backbone, DistilBertForSequenceClassification): # check if extracting only encoder (for reuse of previously fine-tuned)
        if isinstance(self.backbone_model, DistilBertModel):
            # self.pre_classifier = nn.Linear(config.dim, config.dim)
            # self.classifier = nn.Linear(config.dim, config.num_labels)
            # self.dropout = nn.Dropout(config.seq_classif_dropout)            
            self.classification_head = nn.Sequential(    
                nn.Linear(self.config.dim + self.num_extra_features, self.config.dim + self.num_extra_features ),
                nn.ReLU(),
                nn.Dropout(self.config.seq_classif_dropout),    
                nn.Linear(self.config.dim + self.num_extra_features, self.num_labels)
            )
            pass
        elif isinstance(self.backbone_model, BertModel):
            # classifier_dropout = (
            #     self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
            # )
            # self.dropout = nn.Dropout(classifier_dropout)
            # self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)            
            # self.classification_head = nn.Sequential (
            #     nn.Linear(self.config.hidden_size + self.num_extra_features, self.num_labels)
            # )
            print(self.config)
            self.classification_head = nn.Sequential(               
                nn.Linear(self.config.hidden_size + self.num_extra_features, self.config.hidden_size + self.num_extra_features ),
                nn.ReLU(),
                nn.Dropout(self.config.hidden_dropout_prob),    
                nn.Linear(self.config.hidden_size + self.num_extra_features, self.num_labels)
            )            
            pass
        else: # generic classification head
            self.classification_head = nn.Sequential(               
                nn.Linear(self.config.hidden_size + self.num_extra_features, self.config.hidden_size + self.num_extra_features ),
                nn.ReLU(),
                nn.Dropout(self.config.seq_classif_dropout),    
                nn.Linear(self.config.hidden_size + self.num_extra_features, self.num_labels)
            )


        self.criterion = CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            # logger.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True


    def forward(self, **inputs):
        # return self.model(**inputs)
        # print("\n\n============ Inputs =================")
        # print(inputs)
        labels = inputs['labels'] if 'labels' in inputs else None

        # print("\n\n============ Outputs =================")
        
        # outputs = self.backbone_model(**inputs)
        outputs = self.backbone_model(**dict(filter(lambda x: x[0] in self.backbone_valid_inputs, inputs.items()))) #only pass valid params to the backbone
        # print(outputs)
        
        if isinstance(self.backbone_model, DistilBertModel):
            # labels = inputs[4]            
            # from transformers ForSeqCls implementation
            # hidden_state = outputs[0]  # (bs, seq_len, dim)
            # pooled_output = hidden_state[:, 0]  # (bs, dim)
            # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            # pooled_output = self.dropout(pooled_output)  # (bs, dim)
            # logits = self.classifier(pooled_output)  # (bs, num_labels)                        
            pass
        elif isinstance(self.backbone_model, BertModel):    
            # labels = inputs[6]            
            # from transformers ForSeqCls implementation
            # pooled_output = outputs[1]
            # pooled_output = self.dropout(pooled_output)
            # logits = self.classifier(pooled_output)     
            pass

        pooled_output = outputs.last_hidden_state[:, 0] # all dims for the first block (i.e. cls_head) => (bs, seq_len, dim) -> (bs, dim) (728) 
        # print("\n\n============ pooled =================")
        # print(type(pooled_output))
        # print(pooled_output.shape)
        # print(pooled_output)

        # include extra_features (exf_n & exf_c) only for the classification layers 
        # print("\n\n============ extra feats =================")
        # print(f"Extra feats --> {inputs.keys()}")
        exfs = []
        for f in [k for k in inputs.keys() if k.startswith('exf_n_') or k.startswith('exf_c_')]:
            # print(type(inputs[f]))
            # print(inputs[f])
            exfs.append(torch.tensor(inputs[f]).unsqueeze(1))

        exfs = torch.cat(exfs, dim=1)
        # print(exfs)        
        # print(f"-->exfs and pooled_output ==> \n{exfs} \n\n{pooled_output}")
        exfs = exfs.to(pooled_output.device) # to be able to concat
        pooled_output_plus = torch.cat(( pooled_output, exfs ), dim=1)
        # print(f"-->pooled_output_plus ==> \n{pooled_output_plus}")
        # print(pooled_output_plus.shape)
        # print(pooled_output_plus)


        # print("\n\n============ cls =================")
        logits = self.classification_head(pooled_output_plus) # (bs, num_labels)
        # print(type(logits))
        # print(logits.shape)
        # print (logits)

        # print("//////////////////////////////////")

        # if self.problem_type == "single_label_classification":
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # elif self.problem_type == "multi_label_classification":
        #     loss_fct = BCEWithLogitsLoss()
        #     loss = loss_fct(logits, labels)                       

        # l = torch.tensor([1,2,0,2]) # ytrue for each sample in batch
        # print( fn_loss(t, l) )
        # print( fn_loss(t, torch.nn.functional.one_hot(l, num_classes=3).type(torch.float) ) ) # is exactly the same result (maybe useful for target probabilities instead of discrete labels)
        if not labels is None:
            return self.criterion(logits, target=labels), logits
        else:
            return logits


    def training_step(self, batch, batch_idx):
        # print(" *** Training step ** ")
        # outputs = self(**batch) # (loss, logits ...)
        # loss = outputs[0] 
        loss, logits = self(**batch)[:2]

        # log_dict = {'train_loss': loss}
        # return {'loss': loss, 'log': log_dict}
        # self.log("loss", loss) 
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

        # return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"-----> VALIDATING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            yhat = torch.argmax(logits, dim=1)
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
        print(f"\n\n-----> TESTING [{batch_idx}] ==> {len(batch)}")
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            yhat = torch.argmax(logits, dim=1)
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
        print(f"--> SetUp Model ** {stage}")
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
        # model = self.model
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
            print(f"Only Optimizer --> {optimizer}")
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
            type=int,
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
        return parser        