
from asyncio import tasks
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
# from transformers import DistilBertConfig, PreTrainedModel, PretrainedConfig
from transformers import (
    AutoConfig, AutoModel
)
# from transformers.modeling_outputs import BaseModelOutputWithPooling

import logging

loggerx = logging.getLogger(__name__)
loggerx.setLevel(logging.DEBUG)

class MyPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# class ModelForSequenceClassificationPlusFeatsConfig(PretrainedConfig):
#     model_type = "resn"
#     pass

# class MultiTaskLossWrapper(nn.Module):
class MultiTaskLossHomoscedasticUncertainty(nn.Module):
    # Cipolla, R., Gal, Y., & Kendall, A. (2018). Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. 
    # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
    def __init__(self, tasks):
        super(MultiTaskLossHomoscedasticUncertainty, self).__init__()
        self.tasks = tasks
        self.itasks = {task: i for i,task in enumerate(self.tasks.keys())} # idx to the task
        self.task_num = len(tasks)
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)))

    # def forward(self, logits, **kwargs):
    def forward(self, **kwargs):
        clsLoss = CrossEntropyLoss()

        losses = []
        # for itask in self.task_num:
        for task in self.tasks:
            lossi = clsLoss(kwargs[f"logits_{task}"],kwargs[f"labels_{task}"])            
            precisioni = torch.exp(-self.log_vars[self.itasks[task]])
            lossi = precisioni*lossi + self.log_vars[self.itasks[task]]
            losses.append(lossi)
        
        return sum(losses)

class ModelForSequenceClassificationMtlPlusFeatures(nn.Module):
    # config_class = ResnetConfig # no need
    config_class=None

    def __init__(self, model_name_or_path='distilbert-base-uncased', tasks={'task1': 2}, num_extra_features=0, add_pooling_layer:bool=True, **kwargs):        
        super().__init__()

        self.tasks = tasks
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=next(iter(tasks.values())), ignore_mismatched_sizes=True, add_pooling_layer=add_pooling_layer)
        self.backbone_model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.backbone_model = self.backbone_model.base_model # ignore classification head if exists
        # validd backbone forward params (from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/distilbert/modeling_distilbert.py#L691)
        self.backbone_valid_inputs = ['input_ids', 'attention_mask', 'head_mask', 'inputs_embeds', 'output_attentions', 'output_hidden_states', 'return_dict']
        # TODO: Bert have more accepted params in the forward

        self.num_extra_features = num_extra_features
        
        # self.pooler = MyPooler(self.config)

        # initialize classification heads for each task/objective
        self.classification_head = {}
        for task in self.tasks:
            self.classification_head[task] = self.get_head_for_classification(self.tasks[task])

        self.classification_head = nn.ModuleDict(self.classification_head)
        self.criterion = MultiTaskLossHomoscedasticUncertainty(tasks)
        # self.criterion.to(self.device)

    def get_head_for_classification(self, num_labels):
        # head based in BERT config (e.g., hidden_size, hidden_dropout_prob)
        cls = nn.Sequential(               
            nn.Linear(self.config.hidden_size + self.num_extra_features, self.config.hidden_size + self.num_extra_features ),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),    
            nn.Linear(self.config.hidden_size + self.num_extra_features, num_labels)
        )    
        # cls.to(self.device)
        return cls               

    def forward(self, return_dict=False, **inputs):
        # loggerx.debug("\n\n============ Inputs =================")
        # loggerx.debug(inputs)
        # labels = inputs['labels'] if 'labels' in inputs else None
        labels = {k: inputs[k] for k in inputs if k.startswith('labels')}
        # print(f"*** labels :: {labels}")

        # loggerx.debug("\n\n============ Outputs =================")        
        # outputs = self.backbone_model(**inputs)
        outputs = self.backbone_model(**dict(filter(lambda x: x[0] in self.backbone_valid_inputs, inputs.items()))) #only pass valid params to the backbone
        # loggerx.debug(outputs)
        

        # loggerx.debug("\n\n============ pooled =================")
        # pooled_output = outputs.pooler_output # outputs[1] # use pooler defined by each backbone_model
        pooled_output = outputs.last_hidden_state[:, 0] # all dims for the first block (i.e. cls_head) => (bs, seq_len, dim) -> (bs, dim) (728) 
        # pooled_output = self.pooler(outputs.last_hidden_state) # using a custom pooler
        # loggerx.debug(f"pooled output:: {type(pooled_output)} | {pooled_output.shape}")
        # loggerx.debug(pooled_output)

        
        # include extra_features (exf_n & exf_c) only for the classification layers 
        # loggerx.debug("\n\n============ extra feats =================")
        # loggerx.debug(f"Extra feats --> {inputs.keys()}")
        exfs = []
        for f in [k for k in inputs.keys() if k.startswith('exf_n_') or k.startswith('exf_c_')]:
            # loggerx.debug(type(inputs[f]))
            # loggerx.debug(inputs[f])
            exfs.append(torch.tensor(inputs[f]).unsqueeze(1))

        if exfs:
            exfs = torch.cat(exfs, dim=1)
            # loggerx.debug(exfs)
            exfs = exfs.to(pooled_output.device) # to be able to concat
            pooled_output_plus = torch.cat(( pooled_output, exfs ), dim=1)
        else:
            pooled_output_plus = pooled_output
        # loggerx.debug(f"Pooled output:: {pooled_output_plus.shape} | {pooled_output_plus.device})")
        # loggerx.debug(pooled_output_plus)


        # loggerx.debug("\n\n============ cls =================")
        # logits = self.classification_head(pooled_output_plus) # (bs, num_labels)
        logits = { (f'logits_{task}' if len(self.tasks)>1 else 'logits'): self.classification_head[task](pooled_output_plus) for task in self.tasks}
        # loggerx.debug(logits.shape)
        # loggerx.debug(logits)

        # l = torch.tensor([1,2,0,2]) # ytrue for each sample in batch
        # print( fn_loss(t, l) )
        # print( fn_loss(t, torch.nn.functional.one_hot(l, num_classes=3).type(torch.float) ) ) # is exactly the same result (maybe useful for target probabilities instead of discrete labels)
        if labels is not None and labels:
            # loss = torch.nn.functional.cross_entropy(logits, labels)
            # loss = self.criterion(logits, labels)
            loss = self.criterion(**logits, **labels)
            # rs = {"loss": loss, "logits": logits}
            rs = {"loss": loss, **logits, **labels}
        else:
            # rs = {"logits": logits}
            rs = logits
        
        return rs if return_dict else tuple(list(rs.values())) # list of values #[(k, v) for k, v in rs.items()] # list of tuples