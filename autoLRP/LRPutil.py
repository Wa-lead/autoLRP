from torch.nn import functional as F
from .LRPBase import LRPTensor, LRPLayer
import torch
from torch import nn


class LNargsDetach(object):

    def __init__(self):
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True
        self.elementwise_affine = True
        

class LRPLayerNorm(LRPLayer):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, args=None):
        # Create a dummy nn.LayerNorm to pass to the parent constructor
        dummy_layer = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        super().__init__(dummy_layer)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if args is None:
            from .LRPutil import LNargsDetach
            args = LNargsDetach()

        self.sigma = args.sigma
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if not isinstance(input, LRPTensor):
            input = LRPTensor(input)
        self.input = input
        self.input.retain_grad()

        mean = input.mean(dim=-1, keepdim=True)
        if self.mean_detach:
            mean = mean.detach()

        # Calculate std directly as in LayerNormXAI
        std = torch.sqrt(((input - mean) ** 2).sum(dim=-1, keepdims=True) / input.shape[-1])

        # Add a debug log to check std calculation (optional)
        std_torch = torch.std(input, dim=-1, keepdim=True, unbiased=False)
   
        if self.std_detach:
            std = std.detach()

        input_norm = (input - mean) / (std + self.eps)

        if self.elementwise_affine:
            input_norm = input_norm * self.weight + self.bias

        return LRPTensor(input_norm)



class LRPGELU(LRPLayer):
    def __init__(self, inplace: bool = False):
        super(LRPGELU, self).__init__(nn.GELU())
        self.gelu = nn.GELU()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, LRPTensor):
            input = LRPTensor(input)
        self.input = input
        self.input.retain_grad()
        output = nn.Identity()(input) * (F.gelu(input) / (nn.Identity()(input) + 1e-9)).detach()
        return LRPTensor(output)

    # def lrp_backpass(self, output):
    #     if not isinstance(output, LRPTensor):
    #         output = LRPTensor(output)
    #     output.sum().backward(retain_graph=True)
    #     grad = self.input.grad
    #     relevance = (
    #         grad * self.input * (F.gelu(self.input) / (self.input + 1e-9)).detach()
    #     )
    #     return relevance
