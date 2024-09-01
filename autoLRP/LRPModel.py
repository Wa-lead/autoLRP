# LRPnn.py

import torch
from torch import nn
from tqdm import tqdm

from .LRPBase import LRPLayer, LRPTensor
from .LRPFactory import LRPFactory


class LRPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = LRPFactory.wrap(model)
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