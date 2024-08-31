import torch

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