r"""Local-architecture pipeline test — NO GPU, NO gated weights, NO training.

Instantiates real HF model classes (Llama, Bert) with TINY random-weight configs
and runs the actual adapters + scorers against them. This executes the real code
paths — LRPTensor propagation through genuine RMSNorm/attention, captum on
embeddings, span scoring, the MoRF/LeRF perturbation loop — so we verify the
PIPELINE RUNS and produces sane shapes/values, without the paper's models.

It does NOT verify the numbers match the paper (untrained tiny models). It
verifies the machinery executes end to end and the outputs are well-formed.

Run:  python test_local_arch.py    (autoLRP repo must be importable)
"""
from __future__ import annotations
import sys, torch
sys.path.insert(0, '.')
from scorers import morf_lerf, tgs_tps, score_target_prob, score_correct
import adapters as A


def tiny_llama_seqcls():
    from transformers import LlamaConfig, LlamaForSequenceClassification
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, num_labels=2)
    m = LlamaForSequenceClassification(cfg).eval()
    m.config.pad_token_id = 0
    m.model.config._attn_implementation = 'sdpa'
    for p in m.parameters(): p.requires_grad_(True)
    return m, m.model.embed_tokens


def tiny_bert_qa():
    from transformers import BertConfig, BertForQuestionAnswering
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)
    m = BertForQuestionAnswering(cfg).eval()
    for p in m.parameters(): p.requires_grad_(True)
    return m, m.bert.embeddings.word_embeddings


def autolrp_cfg(gamma):
    from autoLRP import LRPConfig
    from _recipe import make_rule
    return LRPConfig(rule=make_rule(gamma_linear=gamma),
                     softmax='passthrough', layernorm='passthrough',
                     activation='passthrough')


def ck(rel, n, name):
    ok = rel is not None and rel.ndim==1 and rel.shape[0]==n and torch.isfinite(rel).all() and float(rel.abs().sum())>0
    print(f"   {'PASS' if ok else 'FAIL'}  {name}: shape={None if rel is None else tuple(rel.shape)} "
          f"finite={None if rel is None else bool(torch.isfinite(rel).all())} "
          f"nonzero={None if rel is None else float(rel.abs().sum())>0}")
    return ok


def main():
    results = {}

    # ---- TEXT: tiny LLaMA seq-cls, all text methods + MoRF/LeRF ----
    print("\n[TEXT] tiny LLaMA seq-cls")
    m, emb = tiny_llama_seqcls()
    ids = torch.randint(1,128,(1,10))
    tgt = int(m(ids).logits[0].argmax())

    rel = A.attribute_autolrp(m, ids, tgt, embed_layer=emb, config=autolrp_cfg(1.0), output_kind='logits')
    results['autolrp_text'] = ck(rel, 10, 'autolrp attribute')

    rel_ig = A.attribute_captum(m, ids, tgt, method='ig', embed_layer=emb,
                                attention_mask=torch.ones_like(ids), output_kind='logits', n_steps=8)
    results['ig_text'] = ck(rel_ig, 10, 'ig attribute')

    rel_gs = A.attribute_captum(m, ids, tgt, method='gradshap', embed_layer=emb,
                                attention_mask=torch.ones_like(ids), output_kind='logits', n_samples=6)
    results['gradshap_text'] = ck(rel_gs, 10, 'gradshap attribute')

    # MoRF/LeRF on the autoLRP relevance
    morf, lerf = morf_lerf(m, ids, rel, tgt, score_fn=score_target_prob,
                           baseline_value=0, n_steps=5, rank='abs')
    mlen_ok = len(morf)==6 and len(lerf)==6 and all(torch.isfinite(torch.tensor(morf+lerf)))
    print(f"   {'PASS' if mlen_ok else 'FAIL'}  morf/lerf curves: len(morf)={len(morf)} len(lerf)={len(lerf)}")
    print(f"        morf={[round(x,3) for x in morf]}")
    print(f"        lerf={[round(x,3) for x in lerf]}")
    results['morf_lerf'] = mlen_ok

    # ---- EQA: tiny BERT QA, span attribution + TGS/TPS ----
    print("\n[EQA] tiny BERT QA")
    mq, embq = tiny_bert_qa()
    ids2 = torch.randint(1,128,(1,12)); am2 = torch.ones_like(ids2)
    out = mq(ids2)
    s, e = int(out.start_logits[0].argmax()), int(out.end_logits[0].argmax())
    if e < s: e = s

    relq = A.attribute_autolrp(mq, ids2, (s,e), embed_layer=embq, config=autolrp_cfg(0.001), output_kind='span')
    results['autolrp_eqa'] = ck(relq, 12, 'autolrp span attribute')

    relq_ig = A.attribute_captum(mq, ids2, (s,e), method='ig', embed_layer=embq,
                                 attention_mask=am2, output_kind='span', n_steps=8)
    results['ig_eqa'] = ck(relq_ig, 12, 'ig span attribute')

    # TGS/TPS with fabricated offsets/spans
    offsets = torch.stack([torch.arange(12), torch.arange(12)+1], dim=1)
    tgs, tps = tgs_tps(relq, gold_char_spans=[(s, e+1)], predicted_token_span=(s,e),
                       token_offsets=offsets, membership='index')
    print(f"   PASS  tgs_tps ran: TGS={tgs} TPS={tps} (booleans)")
    results['tgs_tps'] = isinstance(tgs,bool) and isinstance(tps,bool)

    # ---- VISION: tiny ViT, autoLRP + pixel-perturbation MoRF/LeRF ----
    print("\n[VISION] tiny ViT")
    from transformers import ViTConfig, ViTForImageClassification
    torch.manual_seed(0)
    vcfg_m = ViTConfig(hidden_size=48, num_hidden_layers=2, num_attention_heads=4,
                       intermediate_size=96, image_size=32, patch_size=8, num_channels=3, num_labels=10)
    mv = ViTForImageClassification(vcfg_m).eval()
    for p in mv.parameters(): p.requires_grad_(True)
    img = torch.randn(1,3,32,32)
    from autoLRP import LRPConfig
    from _recipe import make_rule
    vcfg = LRPConfig(rule=make_rule(gamma_linear=0.001, conv_gamma=125.0),
                     softmax='passthrough', layernorm='passthrough', activation='passthrough')
    import autoLRP as autolrp
    xv = autolrp.tensor(img); ov = mv(xv).logits; pv=int(ov.argmax(-1)); ov[0,pv].lrp(config=vcfg)
    relmap = xv.relevance.detach()
    results['autolrp_vision'] = (relmap is not None and tuple(relmap.shape)==(1,3,32,32)
                                 and bool(torch.isfinite(relmap).all()) and float(relmap.abs().sum())>0)
    print(f"   {'PASS' if results['autolrp_vision'] else 'FAIL'}  autolrp vision map: {tuple(relmap.shape)}")
    PATCH=8; IMG=32; npx=IMG//PATCH
    pooled = torch.nn.functional.max_pool2d(relmap.sum(1), kernel_size=PATCH, stride=PATCH)
    patch_rel = pooled.flatten()
    def _vperturb(image, pidx, base):
        o=image.clone()
        for pid in pidx.tolist():
            r,c=divmod(pid,npx); o[0,:,r*PATCH:(r+1)*PATCH,c*PATCH:(c+1)*PATCH]=base
        return o
    def _vsc(o,t):
        lg=o.logits if hasattr(o,'logits') else o; return float(torch.softmax(lg[0],-1)[t])
    vm, vl = morf_lerf(mv, img, patch_rel, pv, score_fn=_vsc, baseline_value=0.0,
                       n_steps=npx*npx, rank='abs', apply_perturbation=_vperturb, input_kind='ids')
    results['vision_morf_lerf'] = len(vm)==(npx*npx+1) and len(vl)==(npx*npx+1)
    print(f"   {'PASS' if results['vision_morf_lerf'] else 'FAIL'}  vision morf/lerf: len={len(vm)}")

    print("\n" + "="*50)
    n_pass = sum(results.values()); n = len(results)
    for k,v in results.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("="*50)
    print(f"{n_pass}/{n} pipeline components execute correctly on local architectures")
    sys.exit(0 if n_pass==n else 1)


if __name__ == '__main__':
    main()