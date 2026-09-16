<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/autolrp-lockup-light.png" alt="autoLRP" width="360">
</p>

# autoLRP

[![PyPI](https://img.shields.io/pypi/v/autolrp)](https://pypi.org/project/autoLRP/)
[![Python](https://img.shields.io/pypi/pyversions/autolrp)](https://pypi.org/project/autoLRP/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Layer-wise Relevance Propagation propagates a logit back to the input space
under one constraint: the logit equals the sum of the relevance that reaches
the input.

This is a model-agnostic PyTorch implementation. It works at the operation
level, so it needs no module rewriting and no module names. It acts on the
torch operations the forward pass ran, following the philosophy of autograd,
hence *autoLRP*.

```python
import autolrp


x = autolrp.tensor(model_input)    # wrap the model input
out = model(x)                     # run the model
out[0, pred].lrp()                 # pick the class logit, call .lrp()
relevance_map = x.relevance        # same shape as model_input
```

```bash
pip install autolrp
```

Links: [PyPI](https://pypi.org/project/autoLRP/) ·
[Source](https://github.com/Wa-lead/autoLRP) ·
[Issues](https://github.com/Wa-lead/autoLRP/issues)

## Examples

A few of the notebooks in [`examples/showcase`](examples/showcase). Each wraps
the input, runs the model untouched, and reads `.relevance` back.

### Image classification

Pixels behind the predicted class, across a CNN and a transformer.
Notebooks: [VGG-16](examples/showcase/vision/01_vgg16.ipynb),
[ViT-B/16](examples/showcase/vision/03_vit_b_16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/image_classification.png?v=3" width="90%">
</p>

### Next-token prediction

[GPT-2](examples/showcase/language/01_gpt2.ipynb): context tokens behind the
next word.

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/gpt2_rose.png" width="75%">
</p>

### Sentiment

[BERT](examples/showcase/language/07_bert_sentiment.ipynb): words behind the
decision.

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/bert_positive.png" width="90%">
</p>

### Image similarity (BiLRP)

Attributing a *similarity* score to the patch pairs that make two images alike.
BiLRP, [Eberle et al. 2020](https://arxiv.org/abs/2003.05431).
Notebook: [BiLRP](examples/showcase/paper_impl/01_bilrp_vgg16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/bilrp_cats.png" width="80%">
</p>

### Contrastive attribution (CLRP)

Isolating what is distinctive to each of two classes in one image.
CLRP, [Gu et al. 2018](https://arxiv.org/abs/1812.02100).
Notebook: [CLRP](examples/showcase/paper_impl/03_clrp_vgg16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/clrp.png?v=2" width="95%">
</p>

### Attention rule variant (CP-LRP)

One preset fragment switches the attention rule ([Ali et al. 2022](https://arxiv.org/abs/2202.07304)).
Notebook: [attention presets](examples/showcase/extras/03_attention_fused_vs_decomposed.ipynb)

```python
out[0, pred].lrp(config=LRPConfig(rule={**BASE, **CPLRP}))
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/cplrp_bert.png" width="90%">
</p>

## How it works

`.lrp()` never rewrites your model. It works on the autograd graph the forward
pass already built:

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/pipeline.png?v=4" width="100%">
</p>

1. **wrap.** `autolrp.tensor(x)` marks the input. A handful of ops are replaced
   by versions that save the activations the rules need, and fused attention is
   written out as ordinary ops. Gradients stay native, so the graph is
   otherwise unchanged.
2. **walk.** After the forward pass, the autograd graph is traversed into an
   ordered plan of nodes.
3. **analyze.** Analyzers tag nodes with *facts*, for example which operand of
   an attention product is the softmax weights.
4. **resolve.** Each node gets one rule, chosen by the config from a fact on the
   node when it has one, otherwise from the node name.
5. **backward.** One pass back to the wrapped input runs those rules as hooks
   that turn the incoming gradient into relevance. What reaches the input is
   `x.relevance`; the model's own `.grad` is never touched.

## Configuration

A node is addressed by its autograd name without the version digit, or by a
fact an analyzer attached to it. `BASE` is the starting table; override entries
on it, or use a preset:

```python
from autolrp import LRPConfig, BASE, CPLRP, ATTNLRP, UNIFORM, on, ELEMENTWISE_NODES

LRPConfig(rule={**BASE, 'AddmmBackward': 'zplus'})
LRPConfig(rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.25})})
LRPConfig(rule={**BASE, **on(ELEMENTWISE_NODES, 'yx')})
LRPConfig.composite()                 # z+ on conv, epsilon elsewhere
LRPConfig(rule={**BASE, **ATTNLRP})
LRPConfig(rule={**BASE, **CPLRP})
LRPConfig(rule={**BASE, **UNIFORM})
```

`print(BASE)` lists every key and the rule it runs. The config says exactly
that: a key that is not a node name or a registered fact, a rule the key cannot
run, and a node that no entry addresses are all errors.

```
LRPConfig(rule={**BASE, 'linear': 'zplus'})
  ValueError: unknown rule key 'linear': not a node name [...]
LRPConfig(rule={**BASE, 'MulBackward': 'zbox'})
  ValueError: rule entry 'MulBackward'='zbox': 'zbox' is not a choice here. Choices: ['proportional']
```

## Citing

If you use autoLRP in your research, please cite it:

```bibtex
@software{alasad2026autolrp,
  author  = {Alasad, Waleed},
  title   = {autoLRP: Layer-wise Relevance Propagation on the PyTorch autograd graph},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/Wa-lead/autoLRP}
}
```

## License

autoLRP is released under the MIT License. See [LICENSE](LICENSE).