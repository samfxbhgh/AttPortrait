# ------------------------------------------------------------------------------
# modified from https://github.com/foivospar/Arc2Face
# Last Modified: 2025-06-26
# Author: Zhang Yuhao
# ------------------------------------------------------------------------------

import torch
from transformers import CLIPTextModel
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision as tv

from datetime import datetime
import json
import os
import shutil


class CLIPTextModelWrapper(CLIPTextModel):
    # Adapted from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/clip/modeling_clip.py#L812
    # Modified to accept precomputed token embeddings "input_token_embs" as input or calculate them from input_ids and return them.
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_token_embs: Optional[torch.Tensor] = None,
        return_token_embs: Optional[bool] = False,
    ) -> Union[Tuple, torch.Tensor, BaseModelOutputWithPooling]:

        if return_token_embs:
            return self.text_model.embeddings.token_embedding(input_ids)
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        output_attentions = output_attentions if output_attentions is not None else self.text_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.text_model.config.use_return_dict
    
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
    
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    
        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=input_token_embs)
    
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
    
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
    
        if self.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
    
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
   
class AUEncoder(torch.nn.Module):
    def __init__(self, number_of_aus: int = 12, hidden_dim: int = 64, clip_dim: int = 1024, pad_zeros: bool = True):
        super().__init__()
        self.au_encoder = nn.Linear(number_of_aus, hidden_dim)
        self.n_aus = number_of_aus
        self.hidden_dim = hidden_dim
        self.clip_dim = clip_dim
        self.pad_zeros = pad_zeros
        
    def forward(self, x):
        x = x.clone()
        x = torch.cat([x, self.au_encoder(x)], dim=1)
        if self.pad_zeros:
            x = torch.cat([
                x,
                torch.zeros(x.shape[0], self.clip_dim - self.n_aus - self.hidden_dim).float().to(x.device)
            ], dim=1)
        x = x.reshape(-1, 1, self.clip_dim)
        return x
    
class AttributeClassifierResnet50(nn.Module):
    def __init__(self, n_attributes=40):
        super().__init__()
        self.n_attributes = n_attributes
        self.model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, self.n_attributes)

    def forward(self, x):
        return self.model(x)


class AttributeClassifier(nn.Module):
    def __init__(self, n_attributes=40, pretrained=True):
        super().__init__()
        self.n_attributes = n_attributes
        self.model = timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=pretrained, num_classes=n_attributes)
        # self.model = timm.create_model('caformer_m36.sail_in22k_ft_in1k', pretrained=pretrained, img_size=224, num_classes=n_attributes)

    def forward(self, x):
        return self.model(x)


class Checkpointer:

    def __init__(self, ckpt_dir, module_and_optimizer_dict, track_best_mode='min', max_to_keep=None):
        assert track_best_mode in ['min', 'max']

        self.ckpt_dir = ckpt_dir
        self.module_and_optimizer_dict = module_and_optimizer_dict
        self.track_best_mode = track_best_mode
        self.track_metrics = []
        self.max_to_keep = max_to_keep

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

    def load_if_exists(self, ckpt_dir_or_file=None, map_location=None, load_best=False, strict=True):
        if ckpt_dir_or_file is None:
            ckpt_dir_or_file = self.ckpt_dir

        if os.path.isdir(ckpt_dir_or_file):
            if load_best:  # best
                ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
            else:  # newest
                list_path = os.path.join(ckpt_dir_or_file, 'checkpoint.json')
                if os.path.exists(list_path):
                    with open(list_path) as f:
                        ckpt_list = json.load(f).get('ckpt_list', [])
                        ckpt_path = os.path.join(ckpt_dir_or_file, ckpt_list[0]) if len(ckpt_list) > 0 else None
                else:
                    ckpt_path = None
        else:
            ckpt_path = ckpt_dir_or_file

        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=map_location)
            for name, module_or_optimizer in self.module_and_optimizer_dict.items():
                if isinstance(module_or_optimizer, torch.nn.Module):
                    module_or_optimizer.load_state_dict(ckpt[name], strict=strict)
                else:
                    module_or_optimizer.load_state_dict(ckpt[name])
            print('Restore checkpoint from %s!' % ckpt_path)
            return True
        else:
            return False

    def save(self, ckpt_name=None, metric=None, is_best=False):
        # save checkpoint
        if ckpt_name is None:
            ckpt_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.ckpt")
        if not ckpt_name.endswith('.ckpt'):
            ckpt_name += '.ckpt'
        save_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save({name: module_or_optimizer.state_dict() for name, module_or_optimizer in self.module_and_optimizer_dict.items()}, save_path)

        # record
        list_path = os.path.join(self.ckpt_dir, 'checkpoint.json')
        if os.path.exists(list_path):
            with open(list_path) as f:
                info = json.load(f)
                ckpt_list = info.get('ckpt_list', [])
                ckpt_list = [ckpt_name] + ckpt_list
        else:
            info = {}
            ckpt_list = [ckpt_name]

        if self.max_to_keep is not None:
            for ckpt_name in ckpt_list[self.max_to_keep:]:
                ckpt_path = os.path.join(self.ckpt_dir, ckpt_name.rstrip())
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
            ckpt_list[self.max_to_keep:] = []

        info['ckpt_list'] = ckpt_list

        metric = float(metric) if metric else None
        if metric and not is_best:
            if len(self.track_metrics) > 0:
                if self.track_best_mode == 'min':
                    is_best = metric < min(self.track_metrics)
                else:
                    is_best = metric > max(self.track_metrics)
            else:
                is_best = True
            self.track_metrics.append(metric)

        # copy best
        if is_best:
            print(f'Save the best: {info["ckpt_list"][0]}, metric: {metric}!')
            shutil.copyfile(save_path, os.path.join(self.ckpt_dir, 'best_model.ckpt'))
            info['best_model'] = info['ckpt_list'][0]
            info['best_metric'] = metric

        with open(list_path, 'w') as f:
            json.dump(info, f, indent=4, separators=(',', ': '))



def clip_embedding_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
) -> torch.Tensor:
    seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

    if position_ids is None:
        position_ids = self.position_ids[:, :seq_length].clone()

    if inputs_embeds is None:
        inputs_embeds = self.token_embedding(input_ids)

    position_embeddings = self.position_embedding(position_ids)
    embeddings = inputs_embeds + position_embeddings

    return embeddings



    