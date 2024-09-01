# lrp_base.py

import torch
from torch import nn

class LRPTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, LRPTensor):
            return data
        instance = torch.Tensor._make_subclass(cls, data)
        instance.requires_grad_(True)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.relevance_scores = torch.zeros_like(self.data)
        self.register_hook(self._grad_hook)

    def _grad_hook(self, grad):
        self.relevance_scores = self.relevance_scores + grad * self.detach()
        return grad

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, torch.Tensor) and not isinstance(result, LRPTensor):
            result = LRPTensor(result)
        return result

class LRPLayer(nn.Module):
    def __init__(self, layer):
        if isinstance(layer, LRPLayer):
            super().__init__()
            self.__dict__ = layer.__dict__.copy()
        else:
            super().__init__()
            self.__dict__ = layer.__dict__.copy()
            self.__class__ = type(
                f"LRP{layer.__class__.__name__}", (LRPLayer, layer.__class__), {}
            )

    def forward(self, *args, **kwargs):
        self.input = args[0] if args else kwargs.get('hidden_states', kwargs.get('inputs_embeds'))
        if isinstance(self.input, LRPTensor):
            self.input.retain_grad()
        output = super().forward(*args, **kwargs)
        if isinstance(output, tuple):
            return tuple(LRPTensor(o) if isinstance(o, torch.Tensor) else o for o in output)
        return LRPTensor(output) if isinstance(output, torch.Tensor) else output

    def lrp_backpass(self, output):
        output.sum().backward(retain_graph=True)
        y = (self.input.grad * self.input).detach()
        y.requires_grad_(True)
        return y
