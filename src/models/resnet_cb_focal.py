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


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, weight, beta=0.9999, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print(weight)
        class_counts = [weight[k] for k in weight]
        print(class_counts)

        class_counts = torch.tensor(class_counts, dtype=torch.float)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        self.weights = weights.to('cuda')

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_factor = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * focal_factor * bce_loss
        else:
            loss = focal_factor * bce_loss

        # 클래스별 가중치 적용
        loss = loss * self.weights.unsqueeze(0)  # (B,C)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class Model(torch.nn.Module):
    def __init__(
        self,
        path =None,
        tokenizer =None,
        weight = None,
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

        self.loss_func = self.set_loss_func(weight = weight)
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

    def set_loss_func(self, weight=None):
        return ClassBalancedFocalLoss(
            weight=weight,
            beta=0.9999,
            alpha=0.25,
            gamma=2.0,
            reduction="mean"
        )

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


