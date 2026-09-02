"""Config helper for the relevance tests: the old one-word ``rule='zplus'``
meant "this rule on every linear family". Spell that over BASE."""
from autolrp import BASE


def on_linear(spec):
    return {**BASE, 'AddmmBackward': spec, 'MmBackward': spec,
            'ConvolutionBackward': spec}
