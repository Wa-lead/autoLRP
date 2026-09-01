r"""Centralized per-model wiring — the parts that historically broke.

One place for: model id, embedding-layer path, dtype, attention impl, the
autoLRP requires-grad trap, T5 decoder shifting, and the EQA span-prediction
forward. The runner imports from here so no wiring is duplicated or guessed
per-script.

STATUS: UNTESTED. Model ids and embedding paths are from source/HF, not run.
"""
from __future__ import annotations
import torch

# (model_id, embedding attribute path, kind, dtype)
MODELS = {
    "imdb":    ("yash3056/Llama-3.2-1B-imdb",          "model.embed_tokens",            "seqcls",  torch.bfloat16),
    "wiki":    ("unsloth/Llama-3.2-1B",                "model.embed_tokens",            "causal",  torch.bfloat16),  # ungated mirror of gated meta-llama/Llama-3.2-1B (identical weights)
    "bert":    ("MrKite/bert-large-squadv2",           "bert.embeddings.word_embeddings",     "qa", torch.bfloat16),
    "roberta": ("deepset/roberta-large-squad2",        "roberta.embeddings.word_embeddings",  "qa", torch.bfloat16),
    "t5":      ("sjrhuschlee/flan-t5-large-squad2",    "shared",                              "qa_enc_dec", torch.bfloat16),
    "vit":     ("nateraw/vit-base-patch16-224-cifar10", None,                                 "vision", torch.float32),
    "vgg":     ("vgg16",                                None,                                 "vision", torch.float32),
}


def get_embed_layer(model, path):
    m = model
    for p in path.split("."):
        m = getattr(m, p)
    return m


def ensure_param_grads(model):
    r"""autoLRP needs params to require grad so AddmmBackward saves the input
    activation (_saved_mat1); otherwise its epsilon rule sees None -> NaN."""
    for p in model.parameters():
        p.requires_grad_(True)


def load_model(task, *, for_autolrp=False, for_attnlrp=False, dtype_override=None):
    r"""Load model + tokenizer + embedding layer for a task. Returns
    (model, tokenizer_or_None, embed_layer_or_None, kind).

    Gated models (e.g. meta-llama/Llama-3.2-1B for the 'wiki' task) require a
    Hugging Face token. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the
    environment, or run `huggingface-cli login`, with access granted on the
    model page. We keep the exact model the paper uses rather than swapping to
    an ungated one, so the benchmark stays faithful; if you cannot get access,
    pass a different --... model at your own change-of-scope.

    for_attnlrp: apply LXT monkey_patch to the right modeling module BEFORE
        instantiation (global patch).
    for_autolrp: set sdpa + call ensure_param_grads.
    """
    import os
    model_id, emb_path, kind, dtype = MODELS[task]
    if dtype_override is not None:  # e.g. float32 where bf16 underflows a stabilizer
        dtype = dtype_override
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth = {"token": hf_token} if hf_token else {}

    if kind in ("seqcls", "causal"):
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  AutoModelForCausalLM)
        tok = AutoTokenizer.from_pretrained(model_id, **auth)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if for_attnlrp:
            from transformers.models.llama import modeling_llama
            from lxt.efficient import monkey_patch
            monkey_patch(modeling_llama)
            cls = (modeling_llama.LlamaForSequenceClassification if kind == "seqcls"
                   else modeling_llama.LlamaForCausalLM)
            model = cls.from_pretrained(model_id, torch_dtype=dtype, **auth,
                                        **({"num_labels": 2} if kind == "seqcls" else {}))
        else:
            cls = AutoModelForSequenceClassification if kind == "seqcls" else AutoModelForCausalLM
            model = cls.from_pretrained(model_id, torch_dtype=dtype, **auth,
                                        **({"num_labels": 2} if kind == "seqcls" else {}))
        # EAGER for AttnLRP/autoLRP: explicit attention ops so the hooks/rules
        # see softmax and the QK/AV matmuls in the graph.
        model.model.config._attn_implementation = "eager"
        if kind == "seqcls":
            model.config.pad_token_id = model.config.eos_token_id
        model.to(device).eval()
        if for_autolrp:
            ensure_param_grads(model)
        return model, tok, get_embed_layer(model, emb_path), kind

    if kind in ("qa", "qa_enc_dec"):
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        tok = AutoTokenizer.from_pretrained(model_id, **auth)
        if for_attnlrp:
            # Bridge lxt-AttnLRP onto the QA models. bert is native (lxt DEFAULT_MAP);
            # roberta/t5 use attnlrp_bridge's per-model patched forwards (lxt does not
            # support their old-style attention). Patch BEFORE from_pretrained.
            import attnlrp_bridge
            attnlrp_bridge.apply_attnlrp(task)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager",
            **auth).to(device).eval()
        if for_autolrp:
            ensure_param_grads(model)
        return model, tok, get_embed_layer(model, emb_path), kind

    if kind == "vision":
        if task == "vit":
            from transformers import ViTForImageClassification
            # ViT-AttnLRP is NOT computed in-process. The paper itself loads a
            # precomputed attribution tensor (ViT.ipynb cells 42-46:
            # torch.load("attnlrp_cifar10_attrs.pt")) generated by lxt's OWN repo
            # (LRP-eXplains-Transformers), because lxt does not support HF
            # ViTForImageClassification directly. Per the decision to "use the
            # libraries as intended", we likewise generate that tensor by running
            # lxt's own ViT example and load it in the runner via
            # --attnlrp-attrs PATH (see run_morf_lerf.py). So there is NO in-process
            # ViT patch_map here — that was a hand-built reconstruction, which is
            # exactly the "extra stuff" we are removing. for_attnlrp loads a plain
            # eager model (the runner supplies the attributions, not the model).
            model = ViTForImageClassification.from_pretrained(
                model_id, attn_implementation=("eager" if for_attnlrp else "sdpa"),
                **auth).to(device).eval()
        else:
            from torchvision.models import vgg16, VGG16_Weights
            model = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()
        if for_autolrp:
            ensure_param_grads(model)
        return model, None, None, kind

    raise ValueError(task)


def t5_decoder_ids(model, input_ids):
    r"""decoder_input_ids for T5 forward = right-shifted input ids."""
    return model._shift_right(input_ids)
