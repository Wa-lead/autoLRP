r"""Every node kind the library addresses, and what each kind saves.
Names are autograd node names without the version digit
(``'MulBackward'`` for ``MulBackward0``); our own wrappers carry no
digit. ``SAVED_TENSORS`` says where each kind keeps its operands and
:func:`saved_tensors` reads them off a node. Nothing here imports the
package; every other module reads these.
"""
from typing import NamedTuple, Optional

# Two-operand nodes.
PRODUCT_NODES = ('AddmmBackward', 'MmBackward', 'BmmBackward', 'ConvolutionBackward')
MUL_NODES = ('MulBackward', 'DivBackward')
ADD_NODES = ('AddBackward', 'SubBackward')

# One-operand nodes.
SOFTMAX_NODES = ('LogSoftmaxBackward', 'SoftmaxBackward')
LAYERNORM_NODES = ('NativeLayerNormBackward', 'LayerNormBackward')
REDUCTION_NODES = ('MeanBackward', 'SumBackward', 'LinalgVectorNormBackward', 'NormBackward')
CUMSUM_NODES = ('CumsumBackward',)

# Elementwise nonlinearities, by the torch function name the intercept
# catches. Each runs through a wrapper that saves its input and output;
# the wrapper's node is the CamelCase name plus ``Backward``.
ELEMENTWISE = (
    'relu', 'leaky_relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'hardtanh',
    'hardswish', 'hardsigmoid', 'elu', 'selu', 'celu', 'softplus', 'softsign',
    'log_sigmoid', 'mish',
    'exp', 'log', 'sqrt', 'rsqrt', 'pow', 'abs', 'sin', 'cos', 'tan', 'clamp',
)


def canonical(name: str) -> str:
    """``'MulBackward0'`` -> ``'MulBackward'``: the node name without its
    version digits, the form every key in this file has."""
    return name.rstrip('0123456789')


def node_of(fname: str) -> str:
    """``'leaky_relu'`` -> ``'LeakyReluBackward'``."""
    return ''.join(part.capitalize() for part in fname.split('_')) + 'Backward'


ELEMENTWISE_NODES = tuple(node_of(fname) for fname in ELEMENTWISE)


# Fused attention kernels, present when decomposition is off: one node
# standing for two products and a softmax.
SDPA_NODES = ('ScaledDotProductEfficientAttention', 'ScaledDotProductFlashAttention',
              'ScaledDotProductCudnnAttention')

# Transparent nodes: relevance passes to the input operand unchanged. A
# sign change or a subtraction from a constant; the normalizations that
# are affine per channel at eval time; dropout, inert at eval time; a
# fused RMSNorm (manual decompositions hit the elementwise entries).
PASSTHROUGH_NODES = (
    'NegBackward', 'RsubBackward',
    'NativeBatchNormBackward', 'CudnnBatchNormBackward', 'BatchNormBackward',
    'NativeGroupNormBackward', 'GroupNormBackward', 'InstanceNormBackward',
    'NativeDropoutBackward', 'DropoutBackward',
    'RmsNormBackward', 'NativeRmsNormBackward',
)

# Routing nodes: the native gradient already sends each output's
# relevance to the input positions it came from. Selections, index ops,
# pooling (average pooling spreads, max pooling routes to the winner),
# the GQA head expansion, whose gradient sums over the repeated heads.
ROUTING_NODES = (
    'MaxBackward', 'MinBackward', 'AmaxBackward', 'AminBackward',
    'WhereBackward', 'MaskedFillBackward', 'IndexSelectBackward', 'GatherBackward',
    'SortBackward', 'TopkBackward', 'RepeatBackward', 'RepeatInterleaveBackward',
    'FlipBackward', 'RollBackward', 'ConstantPadNdBackward',
    'UpsampleNearest', 'UpsampleBilinear',
    'AdaptiveAvgPool', 'AvgPool', 'AdaptiveMaxPool', 'MaxPool',
)

# Shape ops: routing nodes too, and the layer-capture prehooks skip them
# because a view is not a place where relevance means anything new.
SHAPE_NODES = (
    'ReshapeAliasBackward', 'ViewBackward', 'ReshapeBackward',
    'TransposeBackward', 'PermuteBackward', 'SqueezeBackward',
    'UnsqueezeBackward', 'ExpandBackward', 'CatBackward', 'StackBackward',
    'SplitBackward', 'SplitWithSizesBackward', 'NarrowBackward',
    'SliceBackward', 'IndexBackward', 'SelectBackward', 'AliasBackward',
    'CloneBackward', 'ToCopyBackward', 'ContiguousBackward',
    'UnbindBackward', 'TBackward', 'AsStridedBackward',
)


# What each kind saves that the library reads, as (attribute, position)
# pairs: the attribute the tensor sits under on the node, and its position
# among the node's inputs (where relevance is delivered), None for a
# saved tensor that is not an input (a softmax's output, a layer norm's
# mean and rstd). A native node's attributes are ``_saved_*``; one of our
# wrappers saves through ``saved_tensors[i]``. A kind that exists in both
# forms (Add, Sub, Mean, Sum, Softmax) lists the wrapper's, since the
# native node keeps nothing; the reader returns None for what is absent.
_X_Y = (('saved_tensors[0]', 0), ('saved_tensors[1]', None))     # our unary wrappers: input, output
_X = (('saved_tensors[0]', 0),)

SAVED_TENSORS = {
    'AddmmBackward':       (('_saved_mat1', 1), ('_saved_mat2', 2)),        # position 0 holds the bias
    'MmBackward':          (('_saved_self', 0), ('_saved_mat2', 1)),
    'BmmBackward':         (('_saved_self', 0), ('_saved_mat2', 1)),
    'ConvolutionBackward': (('_saved_input', 0), ('_saved_weight', 1)),
    'MulBackward':         (('_saved_self', 0), ('_saved_other', 1)),
    'DivBackward':         (('_saved_self', 0), ('_saved_other', 1)),
    'SubBackward':         (('saved_tensors[0]', 0), ('saved_tensors[1]', 1)),   # our Sub; a native sub saves nothing
    **{n: (('_saved_query', 0), ('_saved_key', 1), ('_saved_value', 2)) for n in SDPA_NODES},
    'LinalgVectorNormBackward': (('_saved_self', 0),),
    'NormBackward':             (('_saved_self', 0),),
    **{n: (('_saved_input', 0), ('_saved_weight', 1), ('_saved_bias', 2),
           ('_saved_result1', None), ('_saved_result2', None)) for n in LAYERNORM_NODES},   # mean, rstd
    'AddBackward':         (('saved_tensors[0]', 0), ('saved_tensors[1]', 1)),
    'MeanBackward':        _X,
    'SumBackward':         _X,
    'CumsumBackward':      _X,
    'SoftmaxBackward':     _X_Y,
    **{n: _X_Y for n in ELEMENTWISE_NODES},
}


class SavedTensor(NamedTuple):
    r"""One saved tensor of a node. ``name``: the attribute it sits under
    (``_saved_mat1``, or ``saved_tensors[i]`` for one of our wrappers).
    ``position``: its index among the node's inputs, the handle autograd
    gives an operand for delivery (``next_functions[i]``,
    ``grad_inputs[i]``); ``None`` for a saved tensor that is not an input.
    ``tensor``: the value, ``None`` if the node did not save it."""
    name: str
    position: Optional[int]
    tensor: Optional[object]


def saved_tensors(node):
    r"""What ``node`` saved that the library reads, as :class:`SavedTensor`
    records in the order :data:`SAVED_TENSORS` lists for its kind; ``()``
    for a kind the library does not attribute through."""
    canon = canonical(node.name())
    spec = SAVED_TENSORS.get(canon)
    if spec is None:                                     # a kernel-suffixed name: ScaledDotProduct...ForCpuBackward
        hits = [k for k in SAVED_TENSORS if k in canon]
        spec = SAVED_TENSORS[max(hits, key=len)] if hits else ()
    saved = getattr(node, 'saved_tensors', None)          # one of our wrappers, else absent
    out = []
    for name, position in spec:
        if name.startswith('saved_tensors['):
            i = int(name[len('saved_tensors['):-1])
            tensor = saved[i] if saved is not None and i < len(saved) else None
        else:
            tensor = getattr(node, name, None)
        out.append(SavedTensor(name, position, tensor))
    return tuple(out)
