import os
import hydra
from typing import Any, List

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)
import torchvision.models as models
import torch.nn as nn

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma: focusing parameter
            alpha: balancing factor per class, tensor of shape [C] or scalar
            reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] 모델 출력 (raw logits)
            targets: [B, C] 멀티라벨 0/1
        """
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        # focal weight 계산
        p_t = targets * prob + (1 - targets) * (1 - prob)
        focal_weight = (1 - p_t) ** self.gamma

        # alpha 적용
        if self.alpha is not None:
            alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_factor

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class Model(torch.nn.Module):
    def __init__(
        self,
        path =None,
        tokenizer =None,
        **kwargs: dict,
    ):
        super().__init__()

        self.config = DictConfig(kwargs)

        #self.tokenizer = tokenizer 

        model = transformers.ResNetForImageClassification
        self.model = model.from_pretrained(path, num_labels = self.config.num_labels, problem_type="multi_label_classification", ignore_mismatched_sizes=True)
        #self.model.classifier = torch.nn.Sequential(
        #                        torch.nn.AdaptiveAvgPool2d(1),  # 공간 차원을 [1,1]로 축소
        #                        torch.nn.Flatten(),  # 텐서를 평탄화
        #                        torch.nn.Linear(2048, self.config.num_labels)
        #                        )
        self.model.train()


        self.pad_token_id = 0

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.device = self.model.device

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    @classmethod
    def set_tokenizer(cls, config):
        tokenizer = hydra.utils.get_class(config._target_)
        tokenizer = tokenizer(**config)
        return tokenizer

    def set_loss_func(self):
        return MultiLabelFocalLoss()

    @torch.no_grad()
    def set_acc_func(self):
        def acc_func(preds, labels):
            #acc = torch.nn.functional.l1_loss(preds, labels)
            acc = 0.0
            preds = torch.argmax(preds, dim=-1)
            for index, label in enumerate(labels) :
                if label[preds[index]] == 1: acc +=1.0
            acc = acc/len(preds)
            return torch.tensor(acc)

            
            return acc
        return acc_func

    def forward(self,
            **kwargs):

        input_ids = kwargs.pop('input_ids', None)
        labels = kwargs.pop('labels', None)
        output = self.model(pixel_values = input_ids)

        logits = output.logits
        ##loss = output.loss

        return {'logits':logits, 'loss':None}

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
        print(logits[0])
        print(batch['labels'][0])

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
        print(logits[0])
        print(batch['labels'][0])

        labels = batch['labels'].tolist()
        predict = predict.tolist()

        inputs = batch['input_ids'].tolist()
        #inputs = self.tokenizer.batch_decode(inputs)

        sigmoid = torch.nn.Sigmoid()
        logit = sigmoid(logits).tolist()

        columns = ['labels', 'preds', 'logit']
        result = zip(labels, predict, logit)

        #if 'data' in item:
        #    raw = item['data']
        #else:
        #    raw = [None for _ in range(len(labels))]

        result = [{'output':dict(zip(columns, x))} for x in result]
        return result


    def configure_optimizers(self, lr=1e-3, **kwargs):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        #optimizer = torch.optim.SGD(                 # ✅ 이렇게 변경
        #params,
        #lr=lr,
        #momentum=0.9,      # VOC-LT 논문 표준
        #weight_decay=1e-4  # 정규화 추가
        #)
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
        logging.info(f"SAVE {path}")


