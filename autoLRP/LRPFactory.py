from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertSelfAttention

from .LRPBert import LRPBertModel, LRPBertEncoder, LRPBertSelfAttention, BERTXAIConfig
from .LRPutil import LRPLayerNorm, LRPGELU, LNargsDetach
from .LRPBase import LRPLayer  

from collections import defaultdict

# Create a default LNargsDetach instance
default_ln_args = LNargsDetach()

lrp_implementations = defaultdict(
    lambda: LRPLayer,
    {
        BertEncoder: lambda m: LRPBertEncoder(config=BERTXAIConfig(), base_encoder=m),
        BertModel: lambda m: LRPBertModel(config=BERTXAIConfig(), base_model=m),
        BertSelfAttention: lambda m: LRPBertSelfAttention(config=BERTXAIConfig(), base_self_attention=m),        
        nn.LayerNorm: lambda m: LRPLayerNorm(m.normalized_shape, eps=m.eps, elementwise_affine=m.elementwise_affine, args=default_ln_args),
        nn.GELU: lambda m: LRPGELU(m),
    }
)


class LRPFactory:
    @staticmethod
    def create_lrp_layer(module):        
        return lrp_implementations[type(module)](module)

    @classmethod
    def wrap(cls, module):
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for i, layer in enumerate(module):
                module[i] = cls.wrap(layer)
        elif isinstance(module, nn.Module):
            module = cls.create_lrp_layer(module)
            # Uncomment the following lines if you want to recursively wrap child modules
            # for name, child in module.named_children():
            #     setattr(module, name, cls.wrap(child))
        return module

