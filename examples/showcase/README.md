# autoLRP showcases

Each notebook wraps one model's input, runs the model as it is, and
reads the relevance back. The four in `extras/` run without downloads
and were executed with their outputs saved; the others need the
pretrained weights they name (torchvision, HuggingFace) and the images
under `../../data`.

Setup, from the repository root:

```
pip install -e .            # autoLRP
pip install torchvision transformers matplotlib scipy pillow
```

Every notebook starts with the same two lines, `sys.path.insert(0,
'..')` for the shared plotting helpers in `_common.py` and
`sys.path.insert(0, '../../..')` for the repository root if the
package is not installed.

| notebook | what it shows |
| --- | --- |
| `extras/01_config_vocabulary` | `BASE`, overriding entries, the errors, `explain`, frozen equals trainable, the three conventions |
| `extras/02_custom_analyzer` | a depth schedule of gammas, and a side fact that detaches a squeeze-and-excitation gate |
| `extras/03_attention_fused_vs_decomposed` | the two attention paths agree; what each preset gives q, k, v; a constant memory |
| `extras/04_custom_installer` | a rule function as a config value; your own hook for a node |
| `vision/01_vgg16` | composite recipe on VGG-16; epsilon, composite and z+ side by side |
| `vision/02_resnet50` | residual connections |
| `vision/03_vit_b_16` | gamma on the linear families with attention weights as constants |
| `vision/04_efficientnetv2` | squeeze-and-excitation gates as products of two live operands |
| `language/01_gpt2` | next-token attribution, and `explain` on the whole model |
| `language/02_roberta` | masked-token prediction |
| `language/03_llama_3_2_1b` | RMSNorm, RoPE, grouped-query attention, SwiGLU under `BASE`, with `explain` |
| `language/04_gemma_4`, `05_qwen3` | the same family under `composite()` and `attnlrp()` |
| `language/07_bert_sentiment` | sentence classification under `BASE` |
| `state_space/01_mamba` | the selective scan as ordinary ops |
| `paper_impl/01_bilrp_vgg16` | BiLRP with the paper's depth-dependent gammas written as analyzers |
| `paper_impl/02_bilrp_bert` | BiLRP on sentence pairs |
| `paper_impl/03_clrp_vgg16` | contrastive LRP, the paper's figure |
| `paper_impl/04_evaluation_metrics` | the four faithfulness metrics in `autolrp.eval` |

`test_viz.py` exercises the plotting helpers with random data.
