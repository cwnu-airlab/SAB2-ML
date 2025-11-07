import os
import hydra
from typing import Any, List

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
import transformers
from omegaconf import DictConfig

class Prefix(transformers.PreTrainedModel):
    def __init__(self, base_config, prefix_len):
        self.base_config = base_config
        super().__init__(base_config)
        
        self.preseq = torch.arange(prefix_len)
        
        self.trans = torch.nn.Sequential(
                torch.nn.Embedding(prefix_len, base_config.hidden_size),
                torch.nn.Linear(base_config.hidden_size, base_config.hidden_size//2),
                torch.nn.Tanh(),
                torch.nn.Linear(base_config.hidden_size//2 , base_config.hidden_size//2),
                torch.nn.Tanh(),
                torch.nn.Linear(base_config.hidden_size//2, base_config.num_hidden_layers * 2 * base_config.hidden_size)
        )
         
    def forward(self, batch_size, device):
        preseq = self.preseq.unsqueeze(0).expand(batch_size, -1).to(device)
        preseq = self.trans(preseq)
        preseq = preseq.reshape(batch_size, len(self.preseq), 2*self.base_config.num_hidden_layers, self.base_config.num_attention_heads,int(self.base_config.hidden_size/self.base_config.num_attention_heads))
        past_key_values = preseq.permute(2,0,3,1,4)

        return past_key_values.split(2)


class Classifier(transformers.PreTrainedModel):
    def __init__(self, base_config, num_labels):
        super().__init__(base_config)
        self.bese_config = base_config
        self.cls = torch.nn.Sequential(
                torch.nn.Linear(base_config.hidden_size, base_config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(base_config.hidden_size, base_config.hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.Linear(base_config.hidden_size//2, num_labels)
        )
        
    
    def forward(self, batch):
        output = self.cls(batch)

        return output




class Model(torch.nn.Module):
    def __init__(
        self,
        path =None,
        tokenizer =None,
        **kwargs: dict,
    ):
        super().__init__()

        self.config = DictConfig(kwargs)

        self.tokenizer = tokenizer 

        model = transformers.BertModel
        self.model = model.from_pretrained(path)
        self.model.train()


        self.pad_token_id = 0

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()
        

        # append code to prefix use config
        if self.config.prompt_path == 'None' :
            self.prompt_model = Prefix(self.model.config, prefix_len = self.config.prefix_length)
        else :
            self.prompt_model = Prefix.from_pretrained(self.config.prompt_path, config = self.model.config, prefix_len = self.config.prefix_length)


        if self.config.cls_path == 'None' :
            self.classifier = Classifier(self.model.config, num_labels = self.config.num_labels)
        else :
            self.classifier = Classifier.from_pretrained(self.config.cls_path, config = self.model.config, num_labels= self.config.num_labels)


        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            self.prompt_model = self.prompt_model.to('cuda')
            self.classifier = self.classifier.to('cuda')
        self.device = self.model.device

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    @classmethod
    def set_tokenizer(cls, config):
        tokenizer = hydra.utils.get_class(config._target_)
        tokenizer = tokenizer(**config)
        tokenizer = tokenizer.get_tokenizer()
        return tokenizer

    def set_loss_func(self):
        return torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def set_acc_func(self):
        def acc_func(preds, labels):
            #acc = torch.nn.functional.l1_loss(preds, labels)
            preds = torch.argmax(preds, dim=-1)
            acc = preds == labels
            acc = torch.sum(acc)/acc.size(0)
            return acc
        return acc_func

    def forward(self,
            **kwargs):

        input_ids = kwargs.pop('input_ids', None)
        labels = kwargs.pop('labels', None)
        past_key_values = self.prompt_model(input_ids['input_ids'].shape[0], device = self.model.device)
        output = self.model(input_ids = input_ids['input_ids'], past_key_values = past_key_values)[1]
        output = self.classifier(output)

        ##logits = output.logits
        ##loss = output.loss

        return {'logits':output, 'loss':None}

    def training_step(self, item: Any, batch_idx: int = None):
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        batch = {'input_ids': item['inputs'], 'labels':item['labels']}
        for key in batch:
            batch[key] = batch[key].to(self.device)
        output = self.forward(**batch)
        logits = output['logits']
        loss = output['loss']

        if loss == None:
            loss = self.loss_func(logits, batch['labels'])
        acc = self.acc_func(logits, batch['labels'])

        return {"loss": loss, "logits":logits, "acc":acc}

    def validation_step(self, item: Any, batch_idx: int = None):
        batch = {'input_ids': item['inputs'], 'labels':item['labels']}
        for key in batch:
            batch[key] = batch[key].to(self.device)

        with torch.no_grad():
            output = self.forward(**batch)
            logits = output['logits']
            loss = output['loss']

            if loss == None:
                loss = self.loss_func(logits, batch['labels'])
            acc = self.acc_func(logits, batch['labels'])
        
        return {"loss": loss, "logits":logits, "acc":acc}

    def test_step(self, batch: Any, batch_idx: int =None):
        raise NotImplementedError("test_step is not used.")

    def predict_step(self, item: Any, batch_idx: int =None):

        batch = {'input_ids': item['inputs'], 'labels':item['labels']}
        for key in batch:
            batch[key] = batch[key].to(self.device)

        with torch.no_grad():
            output = self.forward(**batch)
        logits = output['logits']
        predict = torch.argmax(logits, dim=-1)

        labels = batch['labels'].tolist()
        predict = predict.tolist()

        inputs = batch['input_ids']['input_ids']
        inputs = self.tokenizer.batch_decode(inputs)

        softmax = torch.nn.Softmax(dim=1)
        logit = softmax(logits).tolist()

        columns = ['inputs','labels', 'preds','logit']
        result = zip(inputs, labels, predict, logit)

        if 'data' in item:
            raw = item['data']
        else:
            raw = [None for _ in range(len(labels))]

        result = [{'output':dict(zip(columns, x)), 'data':y} for x,y in zip(result, raw)]
        return result


    def configure_optimizers(self, lr=1e-3, **kwargs):
        params = list(self.classifier.parameters()) + list(self.model.parameters()) + list(self.prompt_model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        return optimizer

    def get_model(self):
        return self.model

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("Need model save code for multi-GPU.")
            #self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
            self.prompt_model.save_pretrained(path+"prefix")
            self.classifier.save_pretrained(path+"cls")
        logging.info(f"SAVE {path}")


