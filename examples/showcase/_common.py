"""Shared helpers for the per-model showcase notebooks.

Publication-quality visualization functions for vision, language, audio,
and multimodal attribution.  Pure matplotlib — works headless on servers.

Each notebook prepends `../..` to sys.path so it can `from _common import ...`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
# Use Agg only when no display is available (headless servers).
# In Jupyter / IPython the inline backend is already set — don't override it.
try:
    _ipy = get_ipython()          # noqa: F821 — only exists in IPython/Jupyter
except NameError:
    matplotlib.use('Agg')         # headless-safe fallback
matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'figure.figsize':    (6, 4),
    'figure.dpi':        110,
    'savefig.dpi':       150,
    'figure.facecolor':  'white',
    'savefig.facecolor': 'white',
})
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as _cm
from scipy.ndimage import gaussian_filter

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

SPECIAL_TOKENS = frozenset({
    '[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>',
    '<|endoftext|>', '<|begin_of_text|>', '<bos>', '<eos>',
})

# Showcase images — diverse objects for multi-example notebooks.
SHOWCASE_IMAGES = [
    ('cats/cat_1',         'cat'),
    ('shark',              'shark'),
    ('showcase/church',    'church'),
    ('showcase/parachute', 'parachute'),
    ('showcase/french_horn', 'french horn'),
    ('showcase/golf_ball', 'golf ball'),
]

# Language prompts for multi-example notebooks.
SHOWCASE_PROMPTS = [
    'The capital of France is',
    'Water freezes at zero degrees',
    'The largest planet in our solar system is',
    'Albert Einstein was born in',
]

# Custom diverging colormap used everywhere.
RELEVANCE_CMAP = LinearSegmentedColormap.from_list(
    'relevance',
    ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b'],
)


# ═══════════════════════════════════════════════════════════════════════════
# Tiny helpers
# ═══════════════════════════════════════════════════════════════════════════

def text_color_for_bg(rgba):
    """Return '#1a1a1a' or 'white' based on background luminance."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    L = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if L < 0.45 else '#1a1a1a'


def _sym_norm(arr: np.ndarray, percentile: float = 99):
    """Symmetric normalisation using the given percentile of |arr|."""
    vmax = max(np.percentile(np.abs(arr), percentile), 1e-12)
    return Normalize(vmin=-vmax, vmax=vmax), vmax


def _save(fig, save_name: str | None):
    if save_name:
        path = FIGURES_DIR / save_name
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15,
                    facecolor='white')
        print(f'  saved {path}')
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════

def imagenet_transform(resize=256, crop=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(crop),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path, transform=None):
    if transform is None:
        transform = imagenet_transform()
    return transform(Image.open(path).convert('RGB')).unsqueeze(0)


def load_showcase_images(transform=None, device='cpu'):
    """Load the standard set of diverse showcase images.

    Returns list of (img_tensor, label_str).
    """
    data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
    results = []
    for stem, label in SHOWCASE_IMAGES:
        for ext in ('.jpg', '.JPEG', '.png'):
            p = data_dir / f'{stem}{ext}'
            if p.exists():
                img = load_image(str(p), transform=transform).to(device)
                results.append((img, label))
                break
    return results


def imagenet_classes():
    here = Path(__file__).resolve().parent
    with open(here.parent.parent / 'data' / 'imagenet_classes.txt') as f:
        return [line.strip() for line in f]


def _denorm(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """(1,C,H,W) or (C,H,W) → (H,W,3) numpy [0,1]."""
    x = t[0] if t.ndim == 4 else t
    x = x.detach().cpu().float().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        x[c] = x[c] * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def _to_spatial(rel, hw=None):
    """Relevance tensor → (H,W) numpy, optionally resized."""
    r = rel.detach().cpu().float() if isinstance(rel, torch.Tensor) else torch.tensor(rel).float()
    if r.ndim == 4: r = r[0]
    if r.ndim == 3: r = r.sum(0)
    arr = r.numpy()
    if hw is not None and arr.shape != tuple(hw):
        arr = F.interpolate(
            torch.tensor(arr)[None, None], size=hw,
            mode='bilinear', align_corners=False,
        )[0, 0].numpy()
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# 1.  VISION ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Vision panels — shared building blocks
# ---------------------------------------------------------------------------

_PANEL_FRAME = dict(linewidth=0.8, edgecolor='#cccccc', facecolor='none')


def _frame_panel(ax):
    """Hide ticks but keep a thin gray rectangle around the image."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def _draw_input_panel(ax, img, caption='Input'):
    ax.imshow(img)
    _frame_panel(ax)
    ax.set_xlabel(caption, fontsize=10, color='#444444', labelpad=6)


def _draw_overlay_panel(ax, img, hmap, vmax, caption='Attribution overlay'):
    ax.imshow(img, alpha=0.4)                    # faint base so relevance dominates
    im = ax.imshow(hmap, cmap=RELEVANCE_CMAP, alpha=0.8,
                   vmin=-vmax, vmax=vmax)
    _frame_panel(ax)
    ax.set_xlabel(caption, fontsize=10, color='#444444', labelpad=6)
    return im


def _draw_focus_panel(ax, img, hmap, vmax, caption='Top-10% focus'):
    """Mask out everything but the top-10% magnitude pixels, render the
    signed relevance through the same diverging cmap on a desaturated
    grayscale base — keeps signed information instead of collapsing to
    a flat red blob."""
    thresh = np.percentile(np.abs(hmap), 90)
    mask = gaussian_filter((np.abs(hmap) >= thresh).astype(np.float32), 1.0)
    mask = np.clip(mask / max(mask.max(), 1e-8), 0, 1)
    # Desaturated gray base (so colored relevance pops without the
    # original RGB fighting for attention).
    gray = (img.mean(axis=-1, keepdims=True).repeat(3, axis=-1)
            * 0.45 + 0.55)
    ax.imshow(gray)
    ax.imshow(hmap * mask, cmap=RELEVANCE_CMAP, alpha=0.85,
              vmin=-vmax, vmax=vmax)
    _frame_panel(ax)
    ax.set_xlabel(caption, fontsize=10, color='#444444', labelpad=6)


def _add_signed_colorbar(fig, im, *, anchor=(0.5, 0.02), width=0.18, height=0.014):
    """Compact horizontal colorbar centered at the bottom of the figure."""
    cax = fig.add_axes([anchor[0] - width / 2, anchor[1], width, height])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor('#bbbbbb')
    cb.ax.tick_params(length=0, labelsize=8, colors='#666666')
    cb.set_ticks([cb.vmin, 0, cb.vmax])
    cb.set_ticklabels(['−', '0', '+'])


# ---------------------------------------------------------------------------
# Public vision helpers
# ---------------------------------------------------------------------------

def show_vision_attribution(img_tensor, relevance, pred_label: str,
                            rule_name: str = '', save_name: str | None = None,
                            figsize=(11, 4.4)):
    """3-panel vision attribution figure: input | overlay | top-10% focus."""
    img = _denorm(img_tensor)
    hmap = _to_spatial(relevance, hw=img.shape[:2])
    _, vmax = _sym_norm(hmap, percentile=99)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.06, top=0.86, bottom=0.14)

    _draw_input_panel(axes[0], img)
    im = _draw_overlay_panel(axes[1], img, hmap, vmax)
    _draw_focus_panel(axes[2], img, hmap, vmax)

    title = f'$\\bf{{{pred_label}}}$'
    if rule_name:
        title = f'{title}   ·   {rule_name}'
    fig.suptitle(title, fontsize=13, y=0.97)
    _add_signed_colorbar(fig, im)
    _save(fig, save_name)


def show_vision_comparison(img_tensor, rule_results: list, pred_label: str,
                           save_name: str | None = None):
    """N-rule comparison: N rows × 3 cols (input | overlay | focus).

    Parameters
    ----------
    rule_results : list of (rule_name, relevance_tensor)
    """
    img = _denorm(img_tensor)
    nr = len(rule_results)
    fig, axes = plt.subplots(nr, 3, figsize=(11, 3.6 * nr + 0.4))
    fig.subplots_adjust(wspace=0.06, hspace=0.30, top=0.92,
                         bottom=0.08, left=0.10)
    if nr == 1:
        axes = axes[np.newaxis, :]

    last_im = None
    for row, (rname, rel) in enumerate(rule_results):
        hmap = _to_spatial(rel, hw=img.shape[:2])
        _, vmax = _sym_norm(hmap, percentile=99)

        _draw_input_panel(axes[row, 0], img,
                          caption='Input' if row == 0 else '')
        last_im = _draw_overlay_panel(
            axes[row, 1], img, hmap, vmax,
            caption='Attribution overlay' if row == 0 else '')
        _draw_focus_panel(axes[row, 2], img, hmap, vmax,
                          caption='Top-10% focus' if row == 0 else '')

        # Row label on the left, vertically centered between the three
        # panels of this row.
        bbox = axes[row, 0].get_position()
        fig.text(0.04, (bbox.y0 + bbox.y1) / 2, rname, ha='right',
                 va='center', fontsize=11, color='#1a1a1a',
                 fontweight='bold')

    fig.suptitle(f'$\\bf{{{pred_label}}}$   ·   rule comparison',
                 fontsize=13, y=0.985)
    if last_im is not None:
        _add_signed_colorbar(fig, last_im, anchor=(0.5, 0.025))
    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TEXT ATTRIBUTION  (bar-stem chart)
# ═══════════════════════════════════════════════════════════════════════════

_POS_COLOR = '#b2182b'   # dark red
_NEG_COLOR = '#2166ac'   # dark blue
_PRED_COLOR = '#2ecc71'  # green


def _style_bar_ax(ax):
    """Remove top/right spines, style left/bottom."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc'); ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#cccccc'); ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(colors='#888888', length=3, width=0.5)
    ax.set_ylabel('relevance', fontsize=10, color='#888888')


def _bar_colors_alpha(rel, amax, is_pred=None, mask_idx=None):
    """Per-bar face color with alpha encoding magnitude."""
    colors = []
    for i, r in enumerate(rel):
        if is_pred is not None and i == is_pred:
            a = 0.5 + 0.3 * abs(r) / max(amax, 1e-12)
            colors.append((*matplotlib.colors.to_rgb(_PRED_COLOR), a))
        elif mask_idx is not None and i == mask_idx:
            a = 0.5 + 0.3 * abs(r) / max(amax, 1e-12)
            colors.append((*matplotlib.colors.to_rgb(_PRED_COLOR), a))
        elif r >= 0:
            a = 0.25 + 0.55 * abs(r) / max(amax, 1e-12)
            colors.append((*matplotlib.colors.to_rgb(_POS_COLOR), a))
        else:
            a = 0.25 + 0.55 * abs(r) / max(amax, 1e-12)
            colors.append((*matplotlib.colors.to_rgb(_NEG_COLOR), a))
    return colors


def _bar_edge_colors(rel, is_pred=None, mask_idx=None):
    colors = []
    for i, r in enumerate(rel):
        if (is_pred is not None and i == is_pred) or (mask_idx is not None and i == mask_idx):
            colors.append(_PRED_COLOR)
        elif r >= 0:
            colors.append(_POS_COLOR)
        else:
            colors.append(_NEG_COLOR)
    return colors


def _format_token(t):
    return t.replace('Ġ', ' ').replace('Ċ', '\\n').replace('▁', ' ')


def show_text_attribution(tokens, relevance, predicted_token=None,
                          title: str = '', save_name: str | None = None,
                          skip_special: bool = True,
                          mask_pos: int | None = None,
                          mask_fill: str | None = None,
                          prompt_text: str | None = None):
    """Bar-stem chart: vertical bars from zero baseline, one per token."""
    if prompt_text and not title:
        title = prompt_text
    tokens = list(tokens)
    rel = np.asarray(relevance, dtype=float)

    # Filter specials
    removed_special = False
    if skip_special:
        keep = [i for i, t in enumerate(tokens) if t not in SPECIAL_TOKENS]
        if len(keep) < len(tokens):
            removed_special = True
        if mask_pos is not None:
            new_pos = None
            for j, k in enumerate(keep):
                if k == mask_pos:
                    new_pos = j; break
            mask_pos = new_pos
        tokens = [tokens[i] for i in keep]
        rel = rel[keep]

    # Append predicted token as last bar
    pred_idx = None
    if predicted_token is not None:
        tokens = tokens + [_format_token(predicted_token)]
        rel = np.append(rel, rel.max() * 0.5)  # moderate height
        pred_idx = len(tokens) - 1

    n = len(tokens)
    amax = np.abs(rel).max()
    fig_w = max(6, n * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    xs = np.arange(n)
    fc = _bar_colors_alpha(rel, amax, is_pred=pred_idx, mask_idx=mask_pos)
    ec = _bar_edge_colors(rel, is_pred=pred_idx, mask_idx=mask_pos)
    edge_lw = [1.5 if (pred_idx is not None and i == pred_idx) or
               (mask_pos is not None and i == mask_pos) else 0.8
               for i in range(n)]

    bars = ax.bar(xs, rel, width=0.55, color=fc, edgecolor=ec,
                  linewidth=edge_lw)
    ax.axhline(0, color='#cccccc', linewidth=0.5)
    _style_bar_ax(ax)

    # X-axis labels
    labels = [_format_token(t) for t in tokens]
    label_colors = ['#666666'] * n
    if pred_idx is not None:
        label_colors[pred_idx] = _PRED_COLOR
    if mask_pos is not None and mask_fill:
        labels[mask_pos] = mask_fill
        label_colors[mask_pos] = _PRED_COLOR
    rotation = 35 if n > 10 else 0
    ha = 'right' if rotation else 'center'
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12, fontfamily='sans-serif',
                       rotation=rotation, ha=ha)
    for lbl, col in zip(ax.get_xticklabels(), label_colors):
        lbl.set_color(col)
        if col == _PRED_COLOR:
            lbl.set_fontweight('semibold')

    # Mask fill annotation
    if mask_pos is not None and mask_fill:
        ax.text(mask_pos, -amax * 0.15, f'→ {mask_fill}', fontsize=9,
                color=_PRED_COLOR, ha='center', fontweight=600)

    if removed_special:
        ax.text(0.5, -0.12, '(special tokens hidden)', fontsize=7,
                color='#999999', ha='center', transform=ax.transAxes)

    if title:
        fig.text(0.02, 0.98, title, fontsize=11, color='#555555',
                 va='top', ha='left', fontweight='normal')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, save_name)


def show_enc_dec_attribution(enc_tokens, enc_relevance, dec_token: str,
                             title: str = '', save_name: str | None = None,
                             skip_special: bool = True,
                             prompt_text: str | None = None):
    """Encoder-decoder bar chart: two stacked subplots with arrow."""
    if prompt_text and not title:
        title = prompt_text
    tokens = list(enc_tokens)
    rel = np.asarray(enc_relevance, dtype=float)

    if skip_special:
        keep = [i for i, t in enumerate(tokens) if t not in SPECIAL_TOKENS]
        tokens = [tokens[i] for i in keep]
        rel = rel[keep]

    n = len(tokens)
    amax = np.abs(rel).max()
    fig_w = max(6, n * 0.9 + 1.5)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, 4.5),
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.35},
    )

    # Top: encoder tokens
    xs = np.arange(n)
    fc = _bar_colors_alpha(rel, amax)
    ec = _bar_edge_colors(rel)
    ax1.bar(xs, rel, width=0.55, color=fc, edgecolor=ec, linewidth=0.8)
    ax1.axhline(0, color='#cccccc', linewidth=0.5)
    _style_bar_ax(ax1)
    labels = [_format_token(t) for t in tokens]
    rotation = 35 if n > 10 else 0
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=12, rotation=rotation,
                        ha='right' if rotation else 'center', color='#666666')
    ax1.set_title('encoder input', fontsize=9, color='#999999', loc='left')

    # Bottom: single decoded token bar
    dec_rel = amax * 0.7
    ax2.bar([0], [dec_rel], width=0.55,
            color=(*matplotlib.colors.to_rgb(_PRED_COLOR), 0.6),
            edgecolor=_PRED_COLOR, linewidth=1.5)
    ax2.axhline(0, color='#cccccc', linewidth=0.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels([_format_token(dec_token)], fontsize=12,
                        fontweight='semibold', color=_PRED_COLOR)
    ax2.set_title('decoded', fontsize=9, color='#999999', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cccccc'); ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_color('#cccccc'); ax2.spines['bottom'].set_linewidth(0.5)
    ax2.tick_params(colors='#888888', length=3, width=0.5)

    # Arrow from encoder to decoder
    con = matplotlib.patches.ConnectionPatch(
        xyA=(n / 2, 0), coordsA=ax1.transData,
        xyB=(0, dec_rel), coordsB=ax2.transData,
        arrowstyle='->', color='#bbbbbb', lw=1.2,
    )
    fig.add_artist(con)
    fig.text(0.52, 0.45, 'cross-attention', fontsize=7,
             color='#999999', va='center')

    if title:
        fig.text(0.02, 0.98, title, fontsize=11, color='#555555',
                 va='top', ha='left')
    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TEXT COMPARISON  (Mamba vs GPT-2 — stacked bar charts)
# ═══════════════════════════════════════════════════════════════════════════

def show_text_comparison(model_results: list, title: str = '',
                         save_name: str | None = None,
                         prompt_text: str | None = None):
    """Two bar charts stacked, shared x-axis and y-scale.

    model_results: [(model_name, tokens_list, rel_array, predicted_token), ...]
    """
    if prompt_text and not title:
        title = prompt_text

    # Filter specials, compute global y-scale
    processed = []
    global_amax = 0
    for mname, toks, rel, ptok in model_results:
        keep = [i for i, t in enumerate(toks) if t not in SPECIAL_TOKENS]
        ft = [toks[i] for i in keep]
        fr = np.asarray(rel, dtype=float)[keep]
        # append predicted token
        ft = ft + [_format_token(ptok)]
        fr = np.append(fr, fr.max() * 0.5)
        global_amax = max(global_amax, np.abs(fr).max())
        processed.append((mname, ft, fr, len(ft) - 1))  # last = pred_idx
    global_amax = max(global_amax, 1e-12)
    ylim = global_amax * 1.15

    # Use the longest token list for sizing
    max_n = max(len(ft) for _, ft, _, _ in processed)
    fig_w = max(6, max_n * 0.9 + 1.5)
    nm = len(processed)

    fig, axes = plt.subplots(nm, 1, figsize=(fig_w, 2.8 * nm),
                             sharex=True)
    if nm == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.15)

    for idx, (mname, ft, fr, pred_idx) in enumerate(processed):
        ax = axes[idx]
        n = len(ft)
        xs = np.arange(n)
        fc = _bar_colors_alpha(fr, global_amax, is_pred=pred_idx)
        ec = _bar_edge_colors(fr, is_pred=pred_idx)
        edge_lw = [1.5 if i == pred_idx else 0.8 for i in range(n)]

        ax.bar(xs, fr, width=0.55, color=fc, edgecolor=ec, linewidth=edge_lw)
        ax.axhline(0, color='#cccccc', linewidth=0.5)
        ax.set_ylim(-ylim, ylim)
        _style_bar_ax(ax)
        ax.set_ylabel(mname, fontsize=10, fontweight=600, color='#555555')

        if idx == nm - 1:
            labels = [_format_token(t) for t in ft]
            rotation = 35 if n > 10 else 0
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=12, rotation=rotation,
                               ha='right' if rotation else 'center',
                               color='#666666')
            for lbl_i, lbl in enumerate(ax.get_xticklabels()):
                if lbl_i == pred_idx:
                    lbl.set_color(_PRED_COLOR)
                    lbl.set_fontweight('semibold')

        # Divider between subplots
        if idx < nm - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False)

    if title:
        fig.text(0.02, 0.98, title, fontsize=11, color='#555555',
                 va='top', ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  AUDIO ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def show_audio_attribution(mel_features, relevance, decoded_token: str = '',
                           title: str = '', sr: int = 16000,
                           hop_length: int = 160,
                           save_name: str | None = None, figsize=(14, 5)):
    """Mel spectrogram + relevance overlay (2 rows, shared x-axis)."""
    spec = mel_features.detach().cpu().float().numpy() if isinstance(
        mel_features, torch.Tensor) else np.asarray(mel_features, dtype=float)
    if spec.ndim == 3: spec = spec[0]

    rel = relevance.detach().cpu().float().numpy() if isinstance(
        relevance, torch.Tensor) else np.asarray(relevance, dtype=float)
    if rel.ndim == 3: rel = rel[0]

    # If relevance is 1-D (per-sample), pool to mel resolution
    if rel.ndim == 1:
        r1 = F.adaptive_avg_pool1d(
            torch.tensor(rel).unsqueeze(0).unsqueeze(0).float(),
            spec.shape[1],
        )[0, 0].numpy()
        rel = np.tile(r1, (spec.shape[0], 1))

    norm, vmax = _sym_norm(rel, 99)
    n_frames = spec.shape[1]
    t_max = n_frames * hop_length / sr

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Row 1 — mel spectrogram
    axes[0].imshow(spec, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, t_max, 0, spec.shape[0]])
    axes[0].set_ylabel('Mel bin', fontsize=9)
    axes[0].set_title('Mel spectrogram', fontsize=9, color='#666666')

    # Row 2 — desaturated background + relevance overlay
    gray = np.mean(spec, axis=0, keepdims=True)
    gray = np.broadcast_to(gray, spec.shape)
    axes[1].imshow(gray, aspect='auto', origin='lower', cmap='gray',
                   alpha=0.3, extent=[0, t_max, 0, spec.shape[0]])
    im = axes[1].imshow(rel, aspect='auto', origin='lower',
                        cmap=RELEVANCE_CMAP, alpha=0.8,
                        vmin=-vmax, vmax=vmax,
                        extent=[0, t_max, 0, spec.shape[0]])
    axes[1].set_ylabel('Mel bin', fontsize=9)
    axes[1].set_xlabel('Time (seconds)', fontsize=9)
    axes[1].set_title('Relevance overlay', fontsize=9, color='#666666')

    # thin vertical colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.02)
    cbar.set_label('relevance', fontsize=8)

    if decoded_token:
        peak_frame = np.abs(rel).mean(axis=0).argmax()
        peak_t = peak_frame * hop_length / sr
        axes[1].annotate(
            f"Predicted: '{decoded_token}'",
            xy=(peak_t, spec.shape[0] * 0.85),
            xytext=(peak_t + t_max * 0.08, spec.shape[0] * 1.02),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'),
        )

    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha='left', y=1.0)
    _save(fig, save_name)


def show_waveform_attribution(waveform, relevance, predicted_text: str = '',
                              sr: int = 16000, save_name: str | None = None,
                              figsize=(14, 4)):
    """Waveform + per-sample relevance for raw audio (Wav2Vec2)."""
    wav = waveform.detach().cpu().float().numpy().flatten() if isinstance(
        waveform, torch.Tensor) else np.asarray(waveform).flatten()
    rel = relevance.detach().cpu().float().numpy().flatten() if isinstance(
        relevance, torch.Tensor) else np.asarray(relevance).flatten()

    t = np.arange(len(wav)) / sr

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.15)

    axes[0].plot(t, wav, color='#888888', linewidth=0.4)
    axes[0].set_ylabel('Amplitude', fontsize=9)
    axes[0].set_title('Input waveform', fontsize=9, color='#666666')

    axes[1].fill_between(t, rel, 0, where=rel >= 0,
                         color='#b2182b', alpha=0.7, linewidth=0)
    axes[1].fill_between(t, rel, 0, where=rel < 0,
                         color='#2166ac', alpha=0.7, linewidth=0)
    axes[1].set_ylabel('Relevance', fontsize=9)
    axes[1].set_xlabel('Time (seconds)', fontsize=9)
    axes[1].set_title('Per-sample relevance', fontsize=9, color='#666666')

    if predicted_text:
        peak_i = np.abs(rel).argmax()
        axes[1].annotate(
            f"Predicted: '{predicted_text}'",
            xy=(t[peak_i], rel[peak_i]),
            xytext=(t[peak_i] + 0.05, np.abs(rel).max() * 0.8),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'),
        )

    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MULTIMODAL  (DePlot)
# ═══════════════════════════════════════════════════════════════════════════

def show_multimodal_attribution(image, patch_relevance, decoded_token: str = '',
                                title: str = '', save_name: str | None = None,
                                figsize=(12, 5)):
    """Original chart + patch relevance overlay for DePlot / Pix2Struct."""
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB')).astype(float) / 255.0
    elif isinstance(image, torch.Tensor):
        img_np = _denorm(image)
    else:
        img_np = np.asarray(image, dtype=float)

    rel = np.asarray(patch_relevance, dtype=float).flatten()
    n_patches = len(rel)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.05)

    axes[0].imshow(img_np); axes[0].axis('off')
    axes[0].set_title('Original', fontsize=9, color='#666666')

    # Reconstruct a roughly-square grid, resize to image dims
    gs = int(math.ceil(math.sqrt(n_patches)))
    padded = np.zeros(gs * gs)
    padded[:n_patches] = rel
    grid = padded.reshape(gs, gs)
    h, w = img_np.shape[:2]
    grid_up = F.interpolate(
        torch.tensor(grid).float()[None, None], size=(h, w),
        mode='bilinear', align_corners=False,
    )[0, 0].numpy()

    norm, vmax = _sym_norm(grid_up, 99)
    axes[1].imshow(img_np)
    axes[1].imshow(grid_up, cmap=RELEVANCE_CMAP, alpha=0.6,
                   vmin=-vmax, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title('Patch relevance', fontsize=9, color='#666666')

    sup = title or (f"Which parts of the chart produce '{decoded_token}'?"
                    if decoded_token else 'Patch relevance')
    fig.suptitle(sup, fontsize=11, x=0.02, ha='left', y=1.0)
    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SigLIP COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def show_siglip_comparison(image_tensor, relevances_by_text: dict,
                           texts: list, save_name: str | None = None,
                           suptitle: str = '',
                           mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Same image, multiple text queries — shared colour scale."""
    img = _denorm(image_tensor, mean=mean, std=std)
    ncols = 1 + len(texts)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    fig.subplots_adjust(wspace=0.03)

    # global vmax
    hmaps = []
    gvmax = 0
    for t in texts:
        h = _to_spatial(relevances_by_text[t], hw=img.shape[:2])
        hmaps.append(h)
        gvmax = max(gvmax, np.percentile(np.abs(h), 99))
    gvmax = max(gvmax, 1e-12)

    axes[0].imshow(img); axes[0].axis('off')
    axes[0].set_title('Original', fontsize=9, color='#666666')

    for i, (t, h) in enumerate(zip(texts, hmaps)):
        axes[i + 1].imshow(img)
        axes[i + 1].imshow(h, cmap=RELEVANCE_CMAP, alpha=0.65,
                           vmin=-gvmax, vmax=gvmax)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(f'"{t}"', fontsize=9, fontstyle='italic',
                              color='#666666')

    sup = suptitle or 'SigLIP-2: the query changes what the model sees'
    fig.suptitle(sup, fontsize=11, x=0.02, ha='left', y=1.01)
    _save(fig, save_name)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy API (backward compat)
# ═══════════════════════════════════════════════════════════════════════════

def show_heatmap(img_tensor, relevance, title='', **kw):
    """Legacy wrapper → show_vision_attribution."""
    show_vision_attribution(img_tensor, relevance, pred_label=title)


def show_token_relevance(tokens, relevance, **kw):
    """Legacy wrapper → show_text_attribution."""
    show_text_attribution(tokens, relevance)


def show_contrastive_attribution(img_tensor, text_results, **kw):
    """Legacy wrapper → show_siglip_comparison."""
    texts = [t for t, _ in text_results]
    rels = {t: r for t, r in text_results}
    show_siglip_comparison(img_tensor, rels, texts, **kw)
