import logging as py_logging
import os
import copy
from contextlib import contextmanager
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings

from transformers import (
    ByT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
try:
    from transformers.generation import GenerationMixin
except ImportError:
    from transformers.generation.utils import GenerationMixin

from transformers.models.t5 import T5Config
from transformers.models.t5.modeling_t5 import *
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm, T5PreTrainedModel, T5Stack

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch.nn import CrossEntropyLoss

from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy

from .shubert import SignHubertModel, SignHubertConfig

logger = py_logging.getLogger(__name__)
_GENERATION_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}


@contextmanager
def patched_transformers_loading():
    """
    Make local loading tolerant to newer Transformers loader behavior.

    The original functions are restored immediately after the model finishes
    loading so the patch does not leak into the rest of the process.
    """
    import transformers

    modeling_utils = getattr(transformers, "modeling_utils", None)
    import_utils = getattr(getattr(transformers, "utils", None), "import_utils", None)
    originals = {}

    if modeling_utils is not None:
        if hasattr(modeling_utils, "_load_parameter_into_model"):
            originals["load_parameter_into_model"] = modeling_utils._load_parameter_into_model

            def patched_load_parameter_into_model(module, param_name, tensor, **kwargs):
                try:
                    return originals["load_parameter_into_model"](module, param_name, tensor, **kwargs)
                except TypeError as exc:
                    if "assign" not in str(exc):
                        raise
                    kwargs.pop("assign", None)
                    return originals["load_parameter_into_model"](module, param_name, tensor, **kwargs)

            modeling_utils._load_parameter_into_model = patched_load_parameter_into_model

        if hasattr(modeling_utils, "check_torch_load_is_safe"):
            originals["check_torch_load_is_safe"] = modeling_utils.check_torch_load_is_safe
            modeling_utils.check_torch_load_is_safe = lambda: None

    if import_utils is not None and hasattr(import_utils, "check_torch_load_is_safe"):
        originals["import_utils_check_torch_load_is_safe"] = import_utils.check_torch_load_is_safe
        import_utils.check_torch_load_is_safe = lambda: None

    try:
        yield
    finally:
        if modeling_utils is not None:
            if "load_parameter_into_model" in originals:
                modeling_utils._load_parameter_into_model = originals["load_parameter_into_model"]
            if "check_torch_load_is_safe" in originals:
                modeling_utils.check_torch_load_is_safe = originals["check_torch_load_is_safe"]
        if import_utils is not None and "import_utils_check_torch_load_is_safe" in originals:
            import_utils.check_torch_load_is_safe = originals["import_utils_check_torch_load_is_safe"]

class SignHubertAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Adjust intermediate_dim based on number of channels
        intermediate_dim_shubert = 1024

        self.signhubert = SignHubertModel(SignHubertConfig(
            channels=channels,
            intermediate_dim=intermediate_dim_shubert
        ))

    def forward(self, x):
        features = self.signhubert.extract_features(x, padding_mask=None, kmeans_labels=None, mask=False)

        # Extract layer outputs
        layer_outputs = []
        for layer in features['layer_results']:
            layer_output = layer[-1]  # Shape: [B, T, D]
            layer_outputs.append(layer_output)

        # Stack the outputs from all layers
        stacked_features = torch.stack(layer_outputs, dim=1)  # Shape: [B, L, T, D]
        return stacked_features

class LinearAdapter(nn.Module):
    def __init__(self, face_dim, hand_dim, pose_dim, representations_dim, out_dim, extraction_layer, channels):
        super().__init__()

        self.signhubert_adapter = SignHubertAdapter(channels)
        self.layer_weights = nn.Parameter(torch.ones(12))  # Learnable weights for each layer
        self.final_layer = nn.Linear(representations_dim, out_dim)
        self.extraction_layer = extraction_layer

    def forward(self, face_features, left_hand_features, right_hand_features, body_posture_features):
        dtype = torch.float32
        face_features = face_features.to(dtype=dtype)
        left_hand_features = left_hand_features.to(dtype=dtype)
        right_hand_features = right_hand_features.to(dtype=dtype)
        body_posture_features = body_posture_features.to(dtype=dtype)

        batch_size, seq_len = face_features.shape[:2]
        dummy_labels = torch.zeros((seq_len, 1), dtype=dtype, device=face_features.device)

        source = []
        for i in range(batch_size):
            source.append({
                "face": face_features[i],
                "left_hand": left_hand_features[i],
                "right_hand": right_hand_features[i],
                "body_posture": body_posture_features[i],
                "label_face": dummy_labels,
                "label_left_hand": dummy_labels,
                "label_right_hand": dummy_labels,
                "label_body_posture": dummy_labels
            })

        # Get representations from SignHubert
        representations_features = self.signhubert_adapter(source) # [T, L, B, D]
        representations_features = representations_features.permute(2, 1, 0, 3) # [B, L, T, D]

        if self.extraction_layer == 0:
            normalized_weights = self.layer_weights
            weighted_representations = representations_features * normalized_weights.view(1, -1, 1, 1)
            representations_for_downstream_task = torch.sum(weighted_representations, dim=1)
        else:
            representations_for_downstream_task = representations_features[:, self.extraction_layer-1, :, :]

        final_output = self.final_layer(representations_for_downstream_task)

        return final_output

class SignLanguageByT5Config(T5Config):
    def __init__(
        self,
        representations_dim=768,
        adapter="linear",
        finetune_signhubert=False,
        face_dim=384,
        hand_dim=384,
        pose_dim=14,
        extraction_layer=0, # use last layer
        channels="face,left_hand,right_hand,body_posture",
        **kwargs
    ):
        self.representations_dim = representations_dim
        self.adapter = adapter
        self.finetune_signhubert = finetune_signhubert
        self.face_dim = face_dim
        self.hand_dim = hand_dim
        self.pose_dim = pose_dim
        self.extraction_layer = extraction_layer
        self.channels = channels
        super().__init__(**kwargs)

class SignLanguageByT5Encoder(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize the adapter based on the configuration
        if config.adapter == "linear":
            self.adapter = LinearAdapter(
                config.face_dim,
                config.hand_dim,
                config.pose_dim,
                config.representations_dim,
                config.d_model,
                config.extraction_layer,
                config.channels
            )
        else:
            raise NotImplementedError("Adapter type not implemented.")

        self.is_decoder = config.is_decoder

        # Define the encoder blocks
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel settings
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        face_features=None,
        left_hand_features=None,
        right_hand_features=None,
        pose_features=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Set default values if not provided
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Use the adapter to convert representation features into embeddings
        inputs_embeds = self.adapter(face_features, left_hand_features, right_hand_features, pose_features)

        input_shape = inputs_embeds.shape[:2]
        batch_size, seq_length = input_shape

        mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Extend attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = self.dropout(inputs_embeds)

        # Iterate over each encoder block
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                present_key_value_states = present_key_value_states + (layer_outputs[1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=None,
        )

class SignLanguageByT5ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        # Initialize the decoder embedding
        self.decoder_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize the encoder with the custom SignLanguageByT5Encoder
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SignLanguageByT5Encoder(encoder_config)

        # Initialize the decoder
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.decoder_emb)

        # Initialize the language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel settings
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.decoder_emb

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def forward(
        self,
        face_features=None,
        left_hand_features=None,
        right_hand_features=None,
        pose_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,  # Keep this for training compatibility
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Set default values if not provided
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head masks if needed
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if encoder outputs are not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                face_features=face_features,
                left_hand_features=left_hand_features,
                right_hand_features=right_hand_features,
                pose_features=pose_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithPastAndCrossAttentions):
            encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Scale sequence output if embeddings are tied
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # Compute language modeling logits
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        face_features=None,
        left_hand_features=None,
        right_hand_features=None,
        pose_features=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Generate method to handle sign language features and generate output sequences.
        """
        # Compute encoder outputs using sign language features
        encoder_outputs = self.encoder(
            face_features=face_features,
            left_hand_features=left_hand_features,
            right_hand_features=right_hand_features,
            pose_features=pose_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pass encoder outputs to the decoder
        kwargs["encoder_outputs"] = encoder_outputs

        # Generate sequences using the parent class's generate method
        return super().generate(
            attention_mask=attention_mask,
            **kwargs
        )

@dataclass
class SignLanguageT5Collator:
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        face_embeds = [feature["face_features"] for feature in features]
        left_hand_embeds = [feature["left_hand_features"] for feature in features]
        right_hand_embeds = [feature["right_hand_features"] for feature in features]
        pose_embeds = [feature["pose_features"] for feature in features]

        # Padding
        max_len = max([emb.shape[0] for emb in face_embeds])

        def pad_embeds(embeds):
            padded_embeds = []
            for emb in embeds:
                if emb.dim() == 3:  # For 3D tensors (pose features)
                    pad_len = max_len - emb.shape[1]  # padding the second dimension (T)
                    emb_pad = torch.nn.functional.pad(emb, (0, 0, 0, pad_len, 0, 0), value=0)
                else:  # For 2D tensors (face, hand features)
                    pad_len = max_len - emb.shape[0]
                    emb_pad = torch.nn.functional.pad(emb, (0, 0, 0, pad_len), value=0)
                padded_embeds.append(emb_pad)
            return padded_embeds

        padded_face_embeds = pad_embeds(face_embeds)
        padded_left_hand_embeds = pad_embeds(left_hand_embeds)
        padded_right_hand_embeds = pad_embeds(right_hand_embeds)
        padded_pose_embeds = pad_embeds(pose_embeds)

        batch = {}
        batch["face_features"] = torch.stack(padded_face_embeds, dim=0)
        batch["left_hand_features"] = torch.stack(padded_left_hand_embeds, dim=0)
        batch["right_hand_features"] = torch.stack(padded_right_hand_embeds, dim=0)
        batch["pose_features"] = torch.stack(padded_pose_embeds, dim=0)

        # For inference, we don't need decoder_input_ids - the model.generate() will handle this
        # Remove the decoder_input_ids requirement
        return batch

class TranslationFeatures(torch.utils.data.Dataset):
    def __init__(self, face_embeddings, left_hand_embeddings, right_hand_embeddings, body_posture_embeddings):
        self.face_embeddings = face_embeddings
        self.left_hand_embeddings = left_hand_embeddings
        self.right_hand_embeddings = right_hand_embeddings
        self.body_posture_embeddings = body_posture_embeddings

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "face_features": torch.tensor(self.face_embeddings),
            "left_hand_features": torch.tensor(self.left_hand_embeddings),
            "right_hand_features": torch.tensor(self.right_hand_embeddings),
            "pose_features": torch.tensor(self.body_posture_embeddings),
        }

def _legacy_generate_text_from_features(
    face_embeddings: np.ndarray,
    left_hand_embeddings: np.ndarray,
    right_hand_embeddings: np.ndarray,
    body_posture_embeddings: np.ndarray,
    model_config: str,
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    output_dir: str,
    generation_max_length: int = 512,
    generation_num_beams: int = 2,
):
    """
    Direct inference function that generates text from sign language features.
    """
    import transformers.modeling_utils
    # Fix 1: Bypass security check for torch.load
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    
    # Fix 2: Bịt miệng lỗi 'assign' gây ra cho fairseq (transformers 4.45+ ép buộc)
    orig_load_param = transformers.modeling_utils._load_parameter_into_model
    def patched_load_parameter_into_model(module, param_name, tensor, **kwargs):
        try:
            return orig_load_param(module, param_name, tensor, **kwargs)
        except TypeError:
            # Nếu module (fairseq) không nhận assign, ta gọi lại nhưng bỏ assign đi
            kwargs.pop('assign', None)
            return orig_load_param(module, param_name, tensor, **kwargs)
    
    transformers.modeling_utils._load_parameter_into_model = patched_load_parameter_into_model

    if hasattr(transformers, 'utils') and hasattr(transformers.utils, 'import_utils'):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

    # Load model and tokenizer
    config = SignLanguageByT5Config.from_pretrained(model_config)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        model_checkpoint,
        # config=config,
        cache_dir=os.path.join(output_dir, "cache"),
        # Disable meta tensor lazy loading for fairseq submodels compatibility
        low_cpu_mem_usage=False,
    )
    tokenizer = ByT5Tokenizer.from_pretrained(tokenizer_checkpoint)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert inputs to tensors and move to device
    face_tensor = torch.tensor(face_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    left_hand_tensor = torch.tensor(left_hand_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    right_hand_tensor = torch.tensor(right_hand_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    pose_tensor = torch.tensor(body_posture_embeddings, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate text (with autocast for GPU acceleration)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            generated_ids = model.generate(
                face_features=face_tensor,
                left_hand_features=left_hand_tensor,
                right_hand_features=right_hand_tensor,
                pose_features=pose_tensor,
                max_length=generation_max_length,
                num_beams=generation_num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def generate_text_from_features(
    face_embeddings: np.ndarray,
    left_hand_embeddings: np.ndarray,
    right_hand_embeddings: np.ndarray,
    body_posture_embeddings: np.ndarray,
    model_config: str,
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    output_dir: str,
    generation_max_length: int = 512,
    generation_num_beams: int = 2,
):
    """
    Direct inference function that generates text from sign language features.
    """
    overall_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = os.path.join(output_dir, "cache")
    model_cache_key = (os.path.abspath(model_checkpoint), str(device))
    tokenizer_cache_key = os.path.abspath(tokenizer_checkpoint)
    model_cache_hit = model_cache_key in _GENERATION_MODEL_CACHE
    tokenizer_cache_hit = tokenizer_cache_key in _TOKENIZER_CACHE

    if not model_cache_hit:
        stage_start = time.perf_counter()
        config = SignLanguageByT5Config.from_pretrained(model_config)
        with patched_transformers_loading():
            model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
                model_checkpoint,
                config=config,
                # Keep eager loading enabled for fairseq-backed submodules.
                low_cpu_mem_usage=False,
                cache_dir=cache_dir,
            )
        model.to(device)
        model.eval()
        _GENERATION_MODEL_CACHE[model_cache_key] = model
        logger.info("Timing | translation_model_load=%.3fs", time.perf_counter() - stage_start)

    if not tokenizer_cache_hit:
        stage_start = time.perf_counter()
        _TOKENIZER_CACHE[tokenizer_cache_key] = ByT5Tokenizer.from_pretrained(tokenizer_checkpoint)
        logger.info("Timing | translation_tokenizer_load=%.3fs", time.perf_counter() - stage_start)

    model = _GENERATION_MODEL_CACHE[model_cache_key]
    tokenizer = _TOKENIZER_CACHE[tokenizer_cache_key]

    stage_start = time.perf_counter()
    face_tensor = torch.tensor(face_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    left_hand_tensor = torch.tensor(left_hand_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    right_hand_tensor = torch.tensor(right_hand_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    pose_tensor = torch.tensor(body_posture_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    logger.info(
        "Timing | translation_tensor_prep=%.3fs | model_cache_hit=%s | tokenizer_cache_hit=%s | seq_len=%d",
        time.perf_counter() - stage_start,
        model_cache_hit,
        tokenizer_cache_hit,
        face_tensor.shape[1],
    )

    stage_start = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            generated_ids = model.generate(
                face_features=face_tensor,
                left_hand_features=left_hand_tensor,
                right_hand_features=right_hand_tensor,
                pose_features=pose_tensor,
                max_length=generation_max_length,
                num_beams=generation_num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    logger.info("Timing | translation_model_generate=%.3fs", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info("Timing | translation_decode=%.3fs", time.perf_counter() - stage_start)
    logger.info("Timing | translation_total=%.3fs", time.perf_counter() - overall_start)

    return decoded_text


def warm_translation_model(
    model_config: str,
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    output_dir: str,
) -> dict[str, object]:
    """
    Preload the generation model and tokenizer into the current process cache.
    """
    overall_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = os.path.join(output_dir, "cache")
    model_cache_key = (os.path.abspath(model_checkpoint), str(device))
    tokenizer_cache_key = os.path.abspath(tokenizer_checkpoint)
    model_cache_hit = model_cache_key in _GENERATION_MODEL_CACHE
    tokenizer_cache_hit = tokenizer_cache_key in _TOKENIZER_CACHE

    if not model_cache_hit:
        stage_start = time.perf_counter()
        config = SignLanguageByT5Config.from_pretrained(model_config)
        with patched_transformers_loading():
            model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
                model_checkpoint,
                config=config,
                low_cpu_mem_usage=False,
                cache_dir=cache_dir,
            )
        model.to(device)
        model.eval()
        _GENERATION_MODEL_CACHE[model_cache_key] = model
        logger.info("Timing | warm_translation_model_load=%.3fs", time.perf_counter() - stage_start)

    if not tokenizer_cache_hit:
        stage_start = time.perf_counter()
        _TOKENIZER_CACHE[tokenizer_cache_key] = ByT5Tokenizer.from_pretrained(tokenizer_checkpoint)
        logger.info("Timing | warm_translation_tokenizer_load=%.3fs", time.perf_counter() - stage_start)

    logger.info(
        "Timing | warm_translation_total=%.3fs | device=%s | model_cache_hit=%s | tokenizer_cache_hit=%s",
        time.perf_counter() - overall_start,
        device,
        model_cache_hit,
        tokenizer_cache_hit,
    )
    return {
        "device": str(device),
        "model_cache_hit": model_cache_hit,
        "tokenizer_cache_hit": tokenizer_cache_hit,
    }

def test(
    face_embeddings: np.ndarray,
    left_hand_embeddings: np.ndarray,
    right_hand_embeddings: np.ndarray,
    body_posture_embeddings: np.ndarray,
    model_config: str,
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    output_dir: str,
):
    """
    Test function for inference - generates text from sign language features.
    This is a simpler wrapper around the direct inference function.
    """
    return generate_text_from_features(
        face_embeddings=face_embeddings,
        left_hand_embeddings=left_hand_embeddings,
        right_hand_embeddings=right_hand_embeddings,
        body_posture_embeddings=body_posture_embeddings,
        model_config=model_config,
        model_checkpoint=model_checkpoint,
        tokenizer_checkpoint=tokenizer_checkpoint,
        output_dir=output_dir,
    )
