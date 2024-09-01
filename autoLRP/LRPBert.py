from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from .LRPBase import LRPTensor, LRPLayer
from .LRPutil import LRPLayerNorm, LRPGELU, LNargsDetach

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


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
        
        def _create_lrp_layer_norm():
            return LRPLayerNorm(
                normalized_shape=(self.config.hidden_size,),
                eps=1e-12,
                elementwise_affine=True,
                args=LNargsDetach()
            )
            
        for bert_layer in base_encoder.layer:
            wrapped_layer = LRPLayer(bert_layer)
            
            # Wrap attention
            wrapped_layer.attention.self = LRPBertSelfAttention(
                config=self.config, 
                base_self_attention=wrapped_layer.attention.self
            )
            # wrapped_layer.attention.output.LayerNorm = _create_lrp_layer_norm()
            
            # # Wrap output LayerNorm
            # wrapped_layer.output.LayerNorm = _create_lrp_layer_norm()
            
            # # Wrap intermediate activation function
            wrapped_layer.intermediate.intermediate_act_fn = LRPGELU(
                wrapped_layer.intermediate.intermediate_act_fn
            )
            
        
        
# class LRPBertEncoder(LRPLayer):
#     def __init__(self, config, base_encoder):
#         super().__init__(base_encoder)
#         self.config = config
#         self.layer = nn.ModuleList(
#             [LRPLayer(layer) for layer in base_encoder.layer]
#         )
#         self.gradient_checkpointing = base_encoder.gradient_checkpointing
        
#     def _override_default_bert_layers(self):
#         for i in range(len(self.layer)):
#             attention = self.layer[i].attention
#             # xai_impl Replace BertSelfAttention in layer.attention with my custom BertSelfAttentionXAI
#             attention.self = LRPBertSelfAttention(config=self.config)

#             # xai_impl Replace BertLayerNorm in layer.attention.output, which is the BertSelfOutput component
#             attention.output.LayerNorm = LRPLayerNorm(
#                 (self.config.hidden_size,), eps=1e-12, args=LNargsDetach()
#             )

#             self.layer[i].attention = attention

#             # xai_impl Replace BertLayerNorm in layer.attention.output, which is the BertOutput component
#             self.layer[i].output.LayerNorm = LRPLayerNorm(
#                 (self.config.hidden_size,), eps=1e-12, args=LNargsDetach()
#             )

#             # xai_impl Replace nonlinear activation function
#             self.layer[i].intermediate.intermediate_act_fn = LRPGELU()

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
        
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    from autoLRP.LRPBase import LRPTensor
    from autoLRP.LRPModel import LRPModel
    from autoLRP.plotutils import plot_bilrp_sentences  

    # Setup device
    device = "mps"

    # Load pre-trained model and tokenizer
    model_name = "sentence-transformers/stsb-bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)