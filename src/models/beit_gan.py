import os
import hydra
from typing import Any, Dict, Optional
from collections.abc import Mapping, Sequence

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig


def compute_bsl_loss(logits: torch.Tensor, labels: torch.Tensor, samples_per_cls: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Balanced Softmax Loss from AT-BSL."""
    if not isinstance(samples_per_cls, torch.Tensor):
        samples_per_cls = torch.tensor(samples_per_cls, device=logits.device, dtype=logits.dtype)
    else:
        samples_per_cls = samples_per_cls.to(device=logits.device, dtype=logits.dtype)
    samples_per_cls = samples_per_cls.clamp_min(eps)
    priors = samples_per_cls.log().view(1, -1)
    priors = priors.expand_as(logits)
    return F.cross_entropy(logits + priors, labels)


class Classifier(transformers.PreTrainedModel):
    def __init__(self, base_config, num_labels):
        super().__init__(base_config)
        self.base_config = base_config
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(base_config.hidden_size, base_config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(base_config.hidden_size, base_config.hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(base_config.hidden_size // 2, num_labels),
        )

    def forward(self, batch):
        return self.cls(batch)


class Model(torch.nn.Module):
    def __init__(
        self,
        path=None,
        tokenizer=None,
        **kwargs: dict,
    ):
        super().__init__()

        self.config = DictConfig(kwargs)
        self.tokenizer = tokenizer

        backbone_cls = transformers.BeitModel
        self.model = backbone_cls.from_pretrained(path)
        self.model.train()

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()

        if self.config.cls_path == "None":
            self.classifier = Classifier(self.model.config, num_labels=self.config.num_labels)
        else:
            self.classifier = Classifier.from_pretrained(
                self.config.cls_path,
                config=self.model.config,
                num_labels=self.config.num_labels,
            )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.classifier = self.classifier.to("cuda")
        self.device = self.model.device

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.use_adv_bsl = bool(self.config.get("use_adv_bsl", False))

        clip_min_cfg = self.config.get("pgd_clip_min", None)
        clip_max_cfg = self.config.get("pgd_clip_max", None)
        default_clip_min = [-2.1179, -2.0357, -1.8044]
        default_clip_max = [2.2489, 2.4285, 2.6400]
        clip_min = clip_min_cfg if clip_min_cfg is not None else default_clip_min
        clip_max = clip_max_cfg if clip_max_cfg is not None else default_clip_max

        self.adv_config = {
            "step_size": float(self.config.get("pgd_step_size", 2.0 / 255.0)),
            "epsilon": float(self.config.get("pgd_epsilon", 8.0 / 255.0)),
            "perturb_steps": int(self.config.get("pgd_perturb_steps", 10)),
            "distance": self.config.get("pgd_distance", "l_inf"),
            "clip_min": clip_min,
            "clip_max": clip_max,
            "random_start": bool(self.config.get("pgd_random_start", True)),
        }

    @classmethod
    def set_tokenizer(cls, config):
        tokenizer = hydra.utils.get_class(config._target_)
        tokenizer = tokenizer(**config)
        return tokenizer

    def set_loss_func(self, weight=None):
        return torch.nn.CrossEntropyLoss(weight=weight)

    @torch.no_grad()
    def set_acc_func(self):
        def acc_func(preds, labels):
            preds = torch.argmax(preds, dim=-1)
            acc = preds == labels
            acc = torch.sum(acc) / acc.size(0)
            return acc

        return acc_func

    def forward(self, input_ids=None, pixel_values=None, labels=None, **kwargs):
        pixel_values = self._extract_pixel_values(pixel_values if pixel_values is not None else input_ids)
        logits = self._forward_logits(pixel_values)
        loss = None if labels is None else self.loss_func(logits, labels)
        return {"logits": logits, "loss": loss}

    def training_step(
        self,
        item: Any,
        batch_idx: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        samples_per_cls: Optional[torch.Tensor] = None,
        use_adv: Optional[bool] = None,
    ):
        pixel_values, labels = self._prepare_batch(item)
        if labels is None:
            raise ValueError("Labels must be provided for training.")
        use_adv = self.use_adv_bsl if use_adv is None else use_adv

        logits_for_loss = None
        if use_adv and optimizer is not None and samples_per_cls is not None:
            loss = self._pgd_bsl_loss(
                pixel_values,
                labels,
                samples_per_cls,
                optimizer,
            )
        else:
            logits_for_loss = self._forward_logits(pixel_values)
            loss = self.loss_func(logits_for_loss, labels)

        with torch.no_grad():
            logits_eval = (
                logits_for_loss.detach()
                if logits_for_loss is not None
                else self._forward_logits(pixel_values)
            )
            acc = self.acc_func(logits_eval, labels)

        return {"loss": loss, "logits": logits_eval, "acc": acc}

    def validation_step(self, item: Any, batch_idx: int = None):
        pixel_values, labels = self._prepare_batch(item)
        if labels is None:
            raise ValueError("Labels must be provided for validation.")

        with torch.no_grad():
            logits = self._forward_logits(pixel_values)
            loss = self.loss_func(logits, labels)
            acc = self.acc_func(logits, labels)

        return {"loss": loss, "logits": logits, "acc": acc}

    def test_step(self, batch: Any, batch_idx: int = None):
        raise NotImplementedError("test_step is not used.")

    def predict_step(self, item: Any, batch_idx: int = None):
        pixel_values, labels = self._prepare_batch(item)

        with torch.no_grad():
            logits = self._forward_logits(pixel_values)
            predict = torch.argmax(logits, dim=-1)

        labels_cpu = labels.cpu().tolist() if labels is not None else [None for _ in range(predict.size(0))]
        predict_cpu = predict.cpu().tolist()

        columns = ["labels", "preds"]
        result = zip(labels_cpu, predict_cpu)

        raw = item.get("data", [None for _ in range(len(labels_cpu))])
        result = [{"output": dict(zip(columns, x)), "data": y} for x, y in zip(result, raw)]
        for idx, entry in enumerate(result):
            if entry["data"] is not None and "source" in entry["data"]:
                entry["data"]["source"] = entry["data"]["source"].tolist()
        return result

    def configure_optimizers(self, lr=1e-3, **kwargs):
        params = list(self.classifier.parameters()) + list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        return optimizer

    def get_model(self):
        return self.model

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("Need model save code for multi-GPU.")
        else:
            self.model.save_pretrained(path)
            self.classifier.save_pretrained(path + "cls")
        logging.info(f"SAVE {path}")

    def _forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = self._extract_pixel_values(pixel_values)
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            if isinstance(outputs, dict):
                last_hidden = outputs["last_hidden_state"]
            else:
                last_hidden = outputs[0]
            pooled = last_hidden[:, 0]
        logits = self.classifier(pooled)
        return logits

    def _prepare_batch(self, batch: Dict[str, Any]):
        pixel_values = self._extract_pixel_values(batch["inputs"]).to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)
        return pixel_values, labels

    @staticmethod
    def _extract_pixel_values(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, Mapping) and "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
        elif hasattr(inputs, "pixel_values"):
            pixel_values = inputs.pixel_values
        else:
            pixel_values = inputs

        if isinstance(pixel_values, torch.Tensor):
            return pixel_values
        if isinstance(pixel_values, Sequence) and not isinstance(pixel_values, (str, bytes)):
            return torch.stack([
                pv if isinstance(pv, torch.Tensor) else torch.as_tensor(pv)
                for pv in pixel_values
            ])
        return torch.as_tensor(pixel_values)

    @staticmethod
    def _prepare_bound(value: Optional[Any], reference: torch.Tensor) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            bound = value.to(device=reference.device, dtype=reference.dtype)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            bound = torch.tensor(list(value), device=reference.device, dtype=reference.dtype)
        else:
            return float(value)

        if bound.dim() == 0:
            return bound.item()
        if bound.shape == reference.shape:
            return bound

        ref_dim = reference.dim()
        if bound.dim() == 1 and ref_dim >= 2:
            view_shape = [1, bound.shape[0]] + [1] * max(ref_dim - 2, 0)
            return bound.view(*view_shape)

        while bound.dim() < ref_dim:
            bound = bound.unsqueeze(-1)
        return bound

    def _clamp(self, tensor: torch.Tensor, clip_min: Optional[Any], clip_max: Optional[Any]) -> torch.Tensor:
        min_bound = self._prepare_bound(clip_min, tensor)
        max_bound = self._prepare_bound(clip_max, tensor)

        if min_bound is not None and max_bound is not None:
            return torch.clamp(tensor, min=min_bound, max=max_bound)
        if min_bound is not None:
            return torch.clamp(tensor, min=min_bound)
        if max_bound is not None:
            return torch.clamp(tensor, max=max_bound)
        return tensor

    def _pgd_bsl_loss(
        self,
        x_natural: torch.Tensor,
        labels: torch.Tensor,
        samples_per_cls: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer],
        **override,
    ) -> torch.Tensor:
        config = {**self.adv_config}
        config.update({k: v for k, v in override.items() if v is not None})
        step_size = config["step_size"]
        epsilon = config["epsilon"]
        perturb_steps = config["perturb_steps"]
        distance = config["distance"]
        clip_min = config["clip_min"]
        clip_max = config["clip_max"]
        random_start = config["random_start"]

        if distance != "l_inf":
            raise NotImplementedError(f"Distance metric '{distance}' is not supported.")

        was_training = self.training
        self.eval()

        if random_start:
            noise = torch.empty_like(x_natural).normal_(0, 1)
            x_adv = x_natural.detach() + 0.001 * noise
        else:
            x_adv = x_natural.detach().clone()
        x_adv = self._clamp(x_adv, clip_min, clip_max)

        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = self._forward_logits(x_adv)
                loss_ce = F.cross_entropy(logits_adv, labels)
            grad = torch.autograd.grad(loss_ce, x_adv, retain_graph=False, create_graph=False)[0]
            step = step_size * torch.sign(grad.detach())
            x_adv = x_adv.detach() + step
            x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)
            x_adv = self._clamp(x_adv, clip_min, clip_max)

        if was_training:
            self.train()
        else:
            self.eval()

        x_adv = self._clamp(x_adv, clip_min, clip_max).detach()
        if optimizer is not None:
            optimizer.zero_grad()
        logits_adv = self._forward_logits(x_adv)
        loss = compute_bsl_loss(logits_adv, labels, samples_per_cls)
        return loss
