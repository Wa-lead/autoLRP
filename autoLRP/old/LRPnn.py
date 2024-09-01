import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

from .LRPTensor import LRPTensor
from ..LRPFactory import LRPFactory


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

    @classmethod
    def wrap(cls, module):
        return LRPFactory.wrap(module)


class LRPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = LRPLayer.wrap(model)
        self.layers = [m for m in self.model.modules() if isinstance(m, LRPLayer)]

        # Disable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def lrp_backpass(self, output):
        R = output
        for dim in tqdm(range(output.shape[-1])):
            mask = torch.zeros_like(output)
            mask[:, dim] = 1
            R = output * mask
            for layer in reversed(self.layers):
                R = layer.lrp_backpass(R)
            self.zero_grad()

    def zero_grad(self):
        super().zero_grad()

        def zero_lrp_tensor_grads(module):
            for name, param in module.named_parameters():
                if isinstance(param, LRPTensor) and param.grad is not None:
                    param.grad = torch.zeros_like(param.grad)
            for name, buffer in module.named_buffers():
                if isinstance(buffer, LRPTensor) and buffer.grad is not None:
                    buffer.grad = torch.zeros_like(buffer.grad)
            for name, submodule in module.named_children():
                if isinstance(submodule, LRPLayer):
                    if (
                        hasattr(submodule, "input")
                        and isinstance(submodule.input, LRPTensor)
                        and submodule.input.grad is not None
                    ):
                        submodule.input.grad = torch.zeros_like(submodule.input.grad)
                zero_lrp_tensor_grads(submodule)

        zero_lrp_tensor_grads(self)

if __name__ == '__main__':
    torch.manual_seed(2342342342)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2), nn.Softmax(dim=-1))
    lrp_model_tests = LRPModel(model)
    input_tensor = LRPTensor(torch.rand(1, 10))
    output = model(input_tensor)
    relevance_scores = lrp_model_tests.lrp_backpass(output)
    print("Input relevance scores:", input_tensor.relevance_scores)