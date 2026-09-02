<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/autolrp-lockup-light.png" alt="autoLRP" width="360">
</p>

# autoLRP

[![PyPI](https://img.shields.io/pypi/v/autolrp)](https://pypi.org/project/autoLRP/)
[![Python](https://img.shields.io/pypi/pyversions/autolrp)](https://pypi.org/project/autoLRP/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A model agnostic PyTorch implementation of Layer-wise Relevance Propagation.
It works at the operation level, so it needs no module rewriting and no module
names: wrap your input in `autoLRP.tensor()`, run the model as it is, pick an
output scalar, and call `.lrp()`. It follows the philosophy of autograd, hence
*autoLRP*.

```python
import autoLRP
from autoLRP import LRPConfig, BASE

x = autoLRP.tensor(image)          # the input you want relevance for
out = model(x)                     # run the model unchanged
out[0, pred].lrp()                 # relevance of class `pred`
heatmap = x.relevance              # same shape as `image`
```

```bash
pip install autolrp
```

Links: [PyPI](https://pypi.org/project/autoLRP/) ·
[Source](https://github.com/Wa-lead/autoLRP) ·
[Issues](https://github.com/Wa-lead/autoLRP/issues)

## Examples

Every figure below is produced by a notebook in
[`examples/showcase`](examples/showcase). The input is wrapped, the model runs
untouched, and `.lrp()` fills `.relevance`.

### Image classification (VGG-16, ViT-B/16)

The same image through a CNN and a vision transformer: relevance on the pixels
that drive the *tiger shark* class, from the same three lines of code.

Notebooks: [VGG-16](examples/showcase/vision/01_vgg16.ipynb),
[ViT-B/16](examples/showcase/vision/03_vit_b_16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/vgg16_shark.png" width="49%">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/vit_shark.png" width="49%">
</p>

### Next-token prediction (GPT-2)

Which context tokens drive the next word.

Notebook: [GPT-2](examples/showcase/language/01_gpt2.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/gpt2_france.png" width="70%">
</p>

### Sentiment (BERT)

Which words carry the sentiment decision.

Notebook: [BERT sentiment](examples/showcase/language/07_bert_sentiment.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/bert_positive.png" width="90%">
</p>

### Image similarity (BiLRP)

Beyond single predictions: decompose the dot-product similarity of two VGG-16
embeddings into the patch *pairs* that make the images look alike. Red pairs
support the similarity, blue pairs oppose it.

Notebook: [BiLRP](examples/showcase/paper_impl/01_bilrp_vgg16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/bilrp_cats.png" width="80%">
</p>

### Contrastive attribution (CLRP)

Separating two classes present in one image. Plain LRP for *zebra* and
*elephant* highlights both animals; CLRP subtracts the shared evidence so each
target keeps only what is distinctive to it.

Notebook: [CLRP](examples/showcase/paper_impl/03_clrp_vgg16.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/clrp_zebra_elephant.png?v=1" width="90%">
</p>

### Attention rule variant (CP-LRP)

One keyword changes how attention is propagated. `attn='cplrp'` treats the
attention weights as constants (Ali et al. 2022) instead of propagating through
them.

```python
out[0, pred].lrp(config=LRPConfig(attn='cplrp'))
```

Notebook: [attention presets](examples/showcase/extras/03_attention_fused_vs_decomposed.ipynb)

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/showcase/cplrp_bert.png" width="90%">
</p>

## How it works

`.lrp()` never rewrites your model. It works on the autograd graph the forward
pass already built:

<p align="center">
  <img src="https://raw.githubusercontent.com/Wa-lead/autoLRP/main/assets/pipeline.png?v=3" width="100%">
</p>

1. **wrap.** `autoLRP.tensor(x)` marks the input. A handful of ops (`add`,
   `sum`, `softmax`, fused attention, and a few more) are replaced by versions
   that save the activations the rules need. Gradients stay native, so the graph
   is otherwise unchanged.
2. **walk.** After the forward pass, the autograd graph is traversed into an
   ordered plan of nodes.
3. **analyze.** Analyzers tag nodes with *facts*, for example which operand of
   an attention product is the softmax weights.
4. **resolve.** Each node gets one rule, chosen by the config from a fact on the
   node when it has one, otherwise from the node name.
5. **backward.** One `backward` pass runs those rules as hooks that turn the
   incoming gradient into relevance. Whatever reaches the wrapped input is
   `x.relevance`.

## Configuration

Every rule-bearing node is addressed by its autograd name without the version
digit, or by a fact an analyzer attached to it. `BASE` is the starting table:

```python
>>> print(BASE)
{'AddmmBackward': 'epsilon', 'MmBackward': 'epsilon', 'ConvolutionBackward': 'epsilon',
 'BmmBackward': 'epsilon', 'MulBackward': 'proportional', 'DivBackward': 'proportional',
 'AddBackward': 'proportional', 'SubBackward': 'proportional',
 'statistic_operand': ('detach', {'by': 'statistic_operand'})}
```

Override entries on it, or use a preset:

```python
LRPConfig(rule={**BASE, 'AddmmBackward': 'zplus'})
LRPConfig(rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.25})})
LRPConfig.composite()               # z+ on conv, epsilon elsewhere
LRPConfig(attn='attnlrp')           # epsilon products, Jacobian softmax
LRPConfig(attn='cplrp')             # attention weights treated as constants
LRPConfig(attn='uniform')
```

The config says exactly what runs. A key that is not a node name or a
registered fact, a rule the key's family cannot run, and a node that no entry
addresses are all errors:

```
LRPConfig(rule={**BASE, 'linear': 'zplus'})
  ValueError: unknown rule key 'linear': not a node name [...]
LRPConfig(rule={**BASE, 'MulBackward': 'zbox'})
  ValueError: rule entry 'MulBackward'='zbox': 'zbox' is not a choice here. Choices: [...]
```

Rule tables, by family:

| family   | node names                                           | rules                                                               |
| -------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| linear   | `AddmmBackward`, `MmBackward`, `ConvolutionBackward` | `epsilon`, `zplus`, `gamma`, `gamma_montavon`, `alpha_beta`, `zbox` |
| bilinear | `BmmBackward`                                        | `epsilon`, `uniform`, `detach_lhs`, `detach_rhs`                    |
| product  | `MulBackward`, `DivBackward`                         | `proportional`, `detach_lhs`, `detach_rhs`                          |
| sum      | `AddBackward`, `SubBackward`                         | `proportional`, `equal`, `fixed`, `detach_lhs`, `detach_rhs`        |

## Citing

If you use autoLRP in your research, please cite it:

```bibtex
@software{alasad2026autolrp,
  author  = {Alasad, Waleed},
  title   = {autoLRP: Layer-wise Relevance Propagation on the PyTorch autograd graph},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Wa-lead/autoLRP}
}
```

## License

autoLRP is released under the MIT License. See [LICENSE](LICENSE).
