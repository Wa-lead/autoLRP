from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from .LRPTensor import LRPTensor
from .LRPnn import LRPLayer

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class LNargsDetach(object):

    def __init__(self):
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True
        self.elementwise_affine = True

class BERTXAIConfig(BertConfig):
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.layer_norm_eps = 1e-12
        self.n_classes = 5
        self.num_hidden_layers = 12

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.detach_layernorm = True  # Detaches the attention-block-output LayerNorm
        self.detach_kq = True  # Detaches the kq-softmax branch
        self.detach_mean = False
        self.device = "mps"
        self.train_mode = False

        ########## Custom Bert encoder params ##########
        super().__init__(
            output_attentions=False,
            # default BERT encoder value: False
            output_hidden_states=False,
            # default BERT encoder value: False
            attention_probs_dropout_prob=0.1,
            # default BERT encoder value: 0.1
            hidden_dropout_prob=0.1,
            # default BERT encoder value: 0.1
            intermediate_size=3072,
            # default BERT encoder value: 3072
            hidden_act="gelu",
            # default BERT encoder value: 'gelu'
            is_decoder=False,  # default BERT encoder value: False
            is_encoder_decoder=False,
            # default BERT encoder value: False
            chunk_size_feed_forward=0,
            # default BERT encoder value: 0
            add_cross_attention=False,
        )  # default BERT encoder value: False)

class LRPLayerNorm(LRPLayer):
    
    def __init__(self, layer):
        super().__init__(layer)
        self.eps = layer.eps
        self.elementwise_affine = layer.elementwise_affine
        if self.elementwise_affine:
            self.weight = layer.weight
            self.bias = layer.bias

    def forward(self, x):
        if not isinstance(x, LRPTensor):
            x = LRPTensor(x)
        self.input = x
        self.input.retain_grad()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.std = (var + self.eps).sqrt().detach()
        y = (x - mean) / self.std
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return LRPTensor(y)

class LRPGELU(LRPLayer):

    def lrp_backpass(self, output):
        if not isinstance(output, LRPTensor):
            output = LRPTensor(output)
        output.sum().backward(retain_graph=True)
        grad = self.input.grad
        relevance = (
            grad * self.input * (F.gelu(self.input) / (self.input + 1e-9)).detach()
        )
        return relevance

class LRPAttention(LRPLayer):
    def __init__(self, layer):
        super().__init__(layer)
        self.query = LRPLayer(layer.query) if hasattr(layer, "query") else None
        self.key = LRPLayer(layer.key) if hasattr(layer, "key") else None
        self.value = LRPLayer(layer.value) if hasattr(layer, "value") else None
        self.dropout = layer.dropout if hasattr(layer, "dropout") else None
        self.num_attention_heads = (
            layer.num_attention_heads if hasattr(layer, "num_attention_heads") else None
        )
        self.attention_head_size = (
            layer.attention_head_size if hasattr(layer, "attention_head_size") else None
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, **kwargs):
        if not isinstance(hidden_states, LRPTensor):
            hidden_states = LRPTensor(hidden_states)
        self.input = hidden_states
        self.input.retain_grad()

        if self.query is not None and self.key is not None and self.value is not None:
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / np.sqrt(self.attention_head_size)

            self.attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout is not None:
                self.attention_probs = self.dropout(self.attention_probs)

            context_layer = torch.matmul(self.attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.num_attention_heads * self.attention_head_size,
            )
            context_layer = context_layer.view(*new_context_layer_shape)

            return LRPTensor(context_layer)
        else:
            return LRPTensor(self.layer(hidden_states, **kwargs))

    def lrp_backpass(self, output):
        if not isinstance(output, LRPTensor):
            output = LRPTensor(output)
        output.sum().backward(retain_graph=True)

        if self.query is not None and self.key is not None and self.value is not None:
            query_relevance = self.query.lrp_backpass(self.query.input.grad)
            key_relevance = self.key.lrp_backpass(self.key.input.grad)
            value_relevance = self.value.lrp_backpass(self.value.input.grad)
            return query_relevance + key_relevance + value_relevance
        else:
            return self.input.relevance_scores

class LRPBertSelfAttention(LRPLayer):
    def __init__(self, config, base_self_attention):
        super().__init__(base_self_attention)
        self.config = config
        self.query = LRPLayer(base_self_attention.query)
        self.key = LRPLayer(base_self_attention.key)
        self.value = LRPLayer(base_self_attention.value)

class LRPBertEncoder(LRPLayer):
    def __init__(self, config, base_encoder):
        super().__init__(base_encoder)
        self.config = config
        self.layer = nn.ModuleList(
            [LRPLayer(layer) for layer in base_encoder.layer]
        )
        
        self.gradient_checkpointing = base_encoder.gradient_checkpointing
        # self._override_default_bert_layers() # causes error

    def _override_default_bert_layers(self):
        for i in range(len(self.layer)):
            attention = self.layer[i].attention
            # xai_impl Replace BertSelfAttention in layer.attention with my custom BertSelfAttentionXAI
            attention.self = LRPBertSelfAttention(config=self.config)

            # xai_impl Replace BertLayerNorm in layer.attention.output, which is the BertSelfOutput component
            attention.output.LayerNorm = LRPLayerNorm(
                (self.config.hidden_size,), eps=1e-12, args=LNargsDetach()
            )

            self.layer[i].attention = attention

            # xai_impl Replace BertLayerNorm in layer.attention.output, which is the BertOutput component
            self.layer[i].output.LayerNorm = LRPLayerNorm(
                (self.config.hidden_size,), eps=1e-12, args=LNargsDetach()
            )

            # xai_impl Replace nonlinear activation function
            self.layer[i].intermediate.intermediate_act_fn = LRPGELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: torch.Tuple[torch.Tuple[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
    ) -> torch.Tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class LRPBertModel(LRPLayer):
    def __init__(self, config, base_model):
        # Call the parent class's __init__ method first
        super().__init__(base_model)
        self.config = config
        
        # Wrap existing components
        self.embeddings = LRPLayer(base_model.embeddings)
        self.encoder = LRPBertEncoder(config=config, base_encoder=base_model.encoder)
        if self.pooler is not None:
            self.pooler = LRPLayer(base_model.pooler)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        print("inputs_embeds:", inputs_embeds)
        if inputs_embeds is not None:
            inputs_embeds = self.embeddings.word_embeddings(inputs_embeds)
        self.input = inputs_embeds
        return super().forward(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )