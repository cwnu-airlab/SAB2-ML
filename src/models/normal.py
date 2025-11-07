import os
import hydra
from typing import Any, List

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
import transformers
from omegaconf import DictConfig
import torch.nn.functional as F



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

        model = transformers.BertForSequenceClassification
        self.model = model.from_pretrained(path, num_labels = self.config.num_labels)
        self.model.train()


        self.pad_token_id = 0

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()
        self.softmax_func = torch.nn.Softmax(dim=1)
        

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.device = self.model.device

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    @classmethod
    def set_tokenizer(cls, config):
        tokenizer = hydra.utils.get_class(config._target_)
        tokenizer = tokenizer(**config)
        tokenizer = tokenizer.get_tokenizer()
        return tokenizer

    def set_loss_func(self, weight=None):
        return torch.nn.CrossEntropyLoss(weight=weight)

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
        output = self.model(input_ids = input_ids['input_ids'], labels = labels)

        return {'logits':output.logits, 'loss':output.loss}
    def freelb_training_step(
        self, batch,
        adv_steps=4, adv_max_norm=0.1, adv_lr=0.05, adv_init_mag=1e-5,
        pad_mask_with_attention=True  # True면 PAD 위치 δ를 0으로 묶음(선택)
    ):
        self.train()
        inputs = batch['inputs']

        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        labels         = batch["labels"].to(self.device)

        # 1) 입력 임베딩 고정
        emb_layer = self.model.get_input_embeddings()  # HF 모델 가정
        embeds = emb_layer(input_ids).detach()            # 파라미터 그래프 분리
        delta  = torch.zeros_like(embeds).normal_(0, adv_init_mag).to(self.device)
        delta.requires_grad_(True)

        # 2) PGD 루프 (K-step, L2)
        for _ in range(adv_steps):
            # 선택: PAD 토큰에는 섭동 안 주기
            if pad_mask_with_attention and attention_mask is not None:
                delta = delta * attention_mask.unsqueeze(-1)

            outputs = self.model(
                inputs_embeds = embeds + delta,
                attention_mask= attention_mask,
                token_type_ids= token_type_ids,
            )
            logits = outputs.logits
            loss_inner = F.cross_entropy(logits, labels, reduction="mean")

            # δ에 대한 그라드만 계산 (모델 파라미터 그라드 X)
            grad_delta = torch.autograd.grad(
                loss_inner, delta, retain_graph=True, create_graph=False
            )[0]

            # L2 정규화된 PGD 한 스텝(상승) + ε-ball 프로젝션
            with torch.no_grad():
                g = grad_delta
                g_norm = g.view(g.size(0), -1).norm(p=2, dim=1).clamp(min=1e-12).view(-1,1,1)
                delta.add_(adv_lr * g / g_norm)  # ascent
                d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1,1,1)
                scale = (adv_max_norm / d_norm).clamp(max=1.0)
                delta.mul_(scale)
            delta.requires_grad_(True)

        # 마지막 섭동 상태에서 최종 손실로 파라미터 업데이트
        if pad_mask_with_attention and attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1)

        final_outputs = self.model(
            inputs_embeds = embeds + delta.detach(),
            attention_mask= attention_mask,
            token_type_ids= token_type_ids,
        )
        final_logits = final_outputs.logits
        loss = F.cross_entropy(final_logits, labels, reduction="mean")

        # 정확도 계산(네 규약 유지)
        with torch.no_grad():
            pred = final_logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()

        return {"loss": loss, "acc": acc}

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

    def sampling_step(self, item: Any, batch_idx: int = None):
        batch = {'input_ids': item['inputs'], 'labels':item['labels']}
        for key in batch:
            batch[key] = batch[key].to(self.device)
        with torch.no_grad():
            output = self.forward(**batch)
            logits = output['logits']
            loss = output['loss']

            logits = self.softmax_func(logits)

            if loss == None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits, batch['labels'])
            acc = self.acc_func(logits, batch['labels'])
        
        return {"loss": loss, "logits":logits, "acc":acc}

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
        params = self.model.parameters()
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
        logging.info(f"SAVE {path}")


