# Copyright 2025 Joshua Southerland
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple, Union
from pathlib import Path

import json
from safetensors.torch import save_file, load_file
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class HlmBaseModel(nn.Module):
    def __init__(self, char_model: AutoModel, word_model: AutoModel):
        super().__init__()
        self.char_model = char_model
        self.word_model = word_model

    @staticmethod
    def from_pretrained(path: str) -> "HlmBaseModel":
        """ Helper for loading a local pre-trained HLM model.

        Args:
            path: the model directory, containing char and word model subdirectories

        Returns:
            HlmBaseModel

        Raises:
            FileNotFoundError: if the model directory does not exist

        """
        _path = Path(path)
        if not _path.is_dir():
            raise FileNotFoundError(f"{_path} is not a directory")
        char_model = AutoModel.from_pretrained(_path / "char_model")
        word_model = AutoModel.from_pretrained(_path / "word_model")
        return HlmBaseModel(char_model, word_model)

    def save_pretrained(self, path: str) -> None:
        _path = Path(path)
        self.char_model.save_pretrained(_path / "char_model")
        self.word_model.save_pretrained(_path / "word_model")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Any]:
        # Note: I didn't use head_mask during training
        orig_shape = input_ids.shape # B, W, C
        unrolled_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1], input_ids.shape[2])
        _attn_mask = (unrolled_ids != 0).long()

        char_output = self.char_model(
            input_ids= unrolled_ids,
            attention_mask=_attn_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        char_output_embeds = char_output[0]
        char_cls_embeds = char_output_embeds[:, 0, :] # -> B*S, E
        dim = char_cls_embeds.shape[-1]
        word_model_inputs = char_cls_embeds.reshape(orig_shape[0],
                                                    orig_shape[1],
                                                    dim) # -> B, S, E
        word_model_attn_mask = attention_mask[:, :, 0]
        batch_size = word_model_inputs.shape[0]
        num_words = word_model_inputs.shape[1]
        num_chars = orig_shape[2]

        dlbrt_output = self.word_model(
            input_ids=None,
            attention_mask=word_model_attn_mask,
            inputs_embeds=word_model_inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        dlbert_output_embeds = dlbrt_output[0]  # B, S, E

        word_embeds = dlbert_output_embeds.reshape((batch_size, num_words, 1, dim))
        initial_embeds = char_output_embeds.reshape((batch_size, num_words, num_chars, dim))

        # TODO: remember a residual connection should help
        char_embeds = torch.concatenate([word_embeds, initial_embeds[:, :, 1:, :]], dim=2)
        return initial_embeds, char_embeds, word_embeds, _attn_mask


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, num_labels: int, seq_classif_dropout: float, problem_type: str):
        super().__init__()
        self.dim = dim
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.seq_classif_dropout = seq_classif_dropout

        # TODO: LayerNorm?
        self.pre_classifier = nn.Linear(2*dim, 2*dim) # 2* due to embedding concatenation
        self.dropout = nn.Dropout(self.seq_classif_dropout)
        self.classifier = nn.Linear(2*dim, self.num_labels)

        self._init_weights(self.pre_classifier)
        self._init_weights(self.classifier)
        self._init_weights(self.dropout)

    def _init_weights(self, module: nn.Module, init_range: float = 0.02):
        # Initialize the weights.
        # Modified From:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py

        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, pooled_output):
        pooled_output = self.pre_classifier(pooled_output)  # B, D
        pooled_output = nn.GELU()(pooled_output)  # B, D
        pooled_output = self.dropout(pooled_output)  # B, D
        logits = self.classifier(pooled_output)  # B, num_labels
        return logits


class HlmForSequenceClassification(nn.Module):
    def __init__(self,
                 base_model_path: str,
                 num_labels: int = 2,
                 seq_classif_dropout: float = 0.1,
                 problem_type: str = "single_label_classification"):
        super().__init__()
        self.base_model = HlmBaseModel.from_pretrained(base_model_path)
        self.dim = self.base_model.word_model.config.hidden_size
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.seq_classif_dropout = seq_classif_dropout
        self.head = ClassifierHead(self.dim, num_labels, seq_classif_dropout, problem_type)

        # Take a hint from the word model
        init_range = self.base_model.word_model.config.initializer_range
        self.head._init_weights(self.head, init_range)


    @staticmethod
    def from_pretrained(path: str, device: Union[int,str]="cpu") -> "HlmForSequenceClassification":
        _path = Path(path)
        if not _path.is_dir():
            raise FileNotFoundError(f"{_path} is not a directory")

        base_model_path = path
        config_fpath = _path / "hlm_ft_config.json"
        with open(config_fpath, "r") as fin:
            config = json.load(fin)

        _model = HlmForSequenceClassification(
            base_model_path,
            num_labels=config["num_labels"],
            seq_classif_dropout=config["seq_classif_dropout"],
            problem_type=config["problem_type"],
        )

        # base model is loaded, need to load the head
        head_fpath = str(_path / "head.safetensors")
        _model.head.load_state_dict(load_file(head_fpath, device=device), strict=True)
        return _model

    def save_pretrained(self, path: str) -> None:
        _path = Path(path)
        _path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(path)
        save_file(self.head.state_dict(), str(_path / "head.safetensors"))
        config_fpath = _path / "hlm_ft_config.json"
        config = {
            "num_labels": self.num_labels,
            "seq_classif_dropout": self.seq_classif_dropout,
            "problem_type": self.problem_type,
        }
        with open(config_fpath, "w") as fout:
            json.dump(config, fout, indent=4)

    def calc_loss(self, logits, labels):
        # Modified From:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return loss

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None, # TOOO: not implemented
                output_hidden_states: Optional[bool] = None,  # TODO: not implemented
                return_dict: Optional[bool] = None,  # TODO: partially implemented
        ) -> Union[Tuple[torch.Tensor, Any], SequenceClassifierOutput]:

            initial_embeds, char_embeds, word_embeds, _attn_mask = self.base_model(input_ids=input_ids,
                                                                                   attention_mask=attention_mask,
                                                                                   head_mask=head_mask,
                                                                                   inputs_embeds=inputs_embeds,
                                                                                   output_attentions=output_attentions,
                                                                                   output_hidden_states=output_hidden_states,
                                                                                   return_dict=return_dict)

            _initial_word_cls_embeds = initial_embeds[:,:,0,:]
            _word_embeds = word_embeds[:,:,0,:] # B, S, D
            _embeds = torch.concatenate([_initial_word_cls_embeds, _word_embeds], dim=2)

            _mask = attention_mask[:,:,:1]  # B, S, 1
            _mask = _mask.expand(_embeds.shape).float()
            maked_embeds = _embeds * _mask
            sum_embeds = torch.sum(maked_embeds, dim=1)
            sum_mask = torch.clamp(_mask.sum(1), min=1e-9)
            pooled_output = sum_embeds / sum_mask

            logits = self.head(pooled_output)
            loss = self.calc_loss(logits, labels)

            # Simplified from
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
            if not return_dict:
                return loss, logits

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )
