from torch.nn import functional as F
from LRPTensor import LRPTensor
from LRPnn import LRPLayer

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
