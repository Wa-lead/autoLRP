r"""LRP configuration.

``rule`` is a dict with one entry per rule-bearing node name (no
version digit: ``'MulBackward'``, not ``'MulBackward0'``), plus
optional entries keyed by a fact name (a label an analyzer attaches to
a node, such as ``'weights_operand'``). A fact entry wins over the
name entry on the nodes that carry the fact. There is no ``'default'``
and there are no aliases: what the dict says is what runs, and
``print(config.rule)`` is the whole answer.

Start from :data:`BASE` and override what you need::

    LRPConfig(rule={**BASE, 'BmmBackward': 'uniform'})

A key that is neither a node name nor a registered fact is an error at
construction, and so is a node-name entry whose value that name's
family cannot run. A fact entry's value is checked at construction
only against the union of every family's rule names (which family will
run it is not known until the fact lands on a node), so a fact entry
whose value the receiving node's family cannot run errors at install.
A node that no entry addresses is also an error at install.

The three unary fields (``softmax``, ``layernorm``, ``activation``)
take a handler name that applies to every node of that kind, or a dict
keyed the same way as ``rule`` by that kind's node names.
"""
from dataclasses import dataclass, field
from typing import Dict, Union

from .backward.rules import (FAMILIES, LINEAR_RULES, MUL_RULES, BMM_RULES,
                             ADD_RULES, VIRTUAL_RULES)
from .backward.analysis import ANALYZERS
from .backward.install import (
    SOFTMAX_HANDLERS, LAYERNORM_HANDLERS, ACTIVATION_HANDLERS,
)
from .backward.strategies import UNARY_NODES
from .backward.resolve import RuleSpec, canonical, _normalize


# One entry per rule-bearing node name (several names share a family:
# the three matmul names all draw from LINEAR_RULES), plus the entry
# that detaches a normalization statistic where the analyzer found one.
BASE: Dict[str, RuleSpec] = {
    'AddmmBackward':       'epsilon',
    'MmBackward':          'epsilon',
    'ConvolutionBackward': 'epsilon',
    'BmmBackward':         'epsilon',
    'MulBackward':         'proportional',
    'DivBackward':         'proportional',
    'AddBackward':         'proportional',
    'SubBackward':         'proportional',
    'statistic_operand':   ('detach', {'by': 'statistic_operand'}),
}

# Attention presets. ``attn=`` writes these entries into ``rule`` and
# sets ``softmax``; it is the only place that speaks method names.
# Each preset touches attention only. In particular ``attn='attnlrp'``
# sets epsilon ``BmmBackward`` and Jacobian softmax and nothing else;
# the full-model recipe (gamma conv, y/x activations on top of that)
# is the classmethod ``LRPConfig.attnlrp``.
_ATTN_PRESETS = {
    'cplrp':   ({'weights_operand': ('detach', {'by': 'weights_operand'}),
                 'BmmBackward': 'epsilon'},
                'passthrough'),                          # Ali et al. 2022
    'attnlrp': ({'BmmBackward': 'epsilon'}, 'jacobian'),  # Achtibat et al. 2024
    'uniform': ({'BmmBackward': 'uniform'}, 'passthrough'),
}

_ALL_RULE_NAMES = (set(LINEAR_RULES) | set(MUL_RULES) | set(BMM_RULES)
                   | set(ADD_RULES))


def _check_key(field_name: str, key, node_names) -> str:
    """Return ``'node'`` or ``'fact'``; raise for anything else."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"{field_name} keys must be non-empty strings; got {key!r}")
    if key in ANALYZERS:
        return 'fact'
    if key in node_names:
        return 'node'
    if key != canonical(key) and canonical(key) in node_names:
        raise ValueError(
            f"{field_name} key {key!r} carries a version digit; write "
            f"{canonical(key)!r}")
    hint = ''
    if key == 'default':
        hint = " There is no 'default': start from autolrp.BASE and override entries."
    raise ValueError(
        f"unknown {field_name} key {key!r}: not a node name "
        f"{sorted(node_names)} and not a fact {sorted(ANALYZERS)}.{hint}")


def _check_name(field_name: str, key, spec, valid_names) -> None:
    name, kw = _normalize(spec)
    if callable(name):
        return
    if not isinstance(name, str):
        raise TypeError(
            f"{field_name} entry {key!r}: name must be a string, got "
            f"{type(name).__name__}")
    if name in VIRTUAL_RULES and field_name == 'rule':
        if not {'detach_lhs', 'detach_rhs'} <= set(valid_names):
            raise ValueError(
                f"rule entry {key!r}={spec!r}: this family has no side to "
                f"detach. Choices: {sorted(valid_names)}")
        by = kw.get('by')
        if not isinstance(by, str) or not by:
            raise ValueError(
                f"rule entry {key!r}={spec!r}: 'detach' needs by=<fact name>")
        if by not in ANALYZERS:
            raise ValueError(
                f"rule entry {key!r}={spec!r}: by={by!r} is not a registered "
                f"fact {sorted(ANALYZERS)}")
        return
    if name not in valid_names:
        raise ValueError(
            f"{field_name} entry {key!r}={spec!r}: {name!r} is not a choice "
            f"here. Choices: {sorted(valid_names)}")


def _validate_rule(mapping) -> None:
    for key, spec in mapping.items():
        kind = _check_key('rule', key, FAMILIES)
        family = FAMILIES[key] if kind == 'node' else None
        _check_name('rule', key, spec,
                    set(family) if family is not None else _ALL_RULE_NAMES)


def _validate_unary(field_name: str, value, handlers) -> None:
    if isinstance(value, dict):
        for key, spec in value.items():
            _check_key(field_name, key, UNARY_NODES[field_name])
            _check_name(field_name, key, spec, set(handlers))
    else:
        _check_name(field_name, '<all>', value, set(handlers))


@dataclass(frozen=True)
class LRPConfig:
    r"""Immutable configuration for one LRP attribution.

    Args:
        rule: Dict keyed by node name (no version digit) or fact name.
            Default: :data:`BASE`.
        softmax: Handler for softmax nodes, or a dict keyed by softmax
            node name.
        layernorm: Handler for layernorm nodes, or a dict.
        activation: Handler for activation nodes, or a dict.
        attn: Attention preset, ``'cplrp'``, ``'attnlrp'`` or
            ``'uniform'``. Writes the preset's entries into ``rule`` and
            sets ``softmax``. Passing a conflicting ``softmax`` or a
            conflicting entry for the same key is an error.
        eps: Stabilizer added to denominators.
        relevance_filter: Top-fraction relevance pass-through in
            ``(0, 1]``.
        capture_layers: When ``True``, :meth:`LRPTensor.lrp` returns a
            dict of per-node relevances.
    """
    rule:       Dict[str, RuleSpec] = field(default_factory=lambda: dict(BASE))
    softmax:    Union[str, Dict[str, str], None] = None
    layernorm:  Union[str, Dict[str, str]] = 'identity'
    activation: Union[str, Dict[str, str]] = 'passthrough'
    attn:       Union[str, None] = None

    # Denominator stabilizer. The 1e-11 default is small enough that it
    # only guards against division by zero, keeping conservation
    # near-exact; the preset constructors that pass eps=1e-6 trade a
    # little absorbed relevance for damping of near-zero denominators.
    eps: float = 1e-11
    relevance_filter: float = 1.0
    capture_layers: bool = False

    def __post_init__(self):
        if not isinstance(self.rule, dict):
            raise TypeError(
                "rule must be a dict keyed by node name or fact name; start "
                "from autolrp.BASE, e.g. rule={**BASE, 'BmmBackward': 'uniform'}")
        rule = dict(self.rule)

        if self.attn is not None:
            if self.attn not in _ATTN_PRESETS:
                raise ValueError(
                    f"Unknown attn preset {self.attn!r}. "
                    f"Choices: {sorted(_ATTN_PRESETS)}")
            entries, sm = _ATTN_PRESETS[self.attn]
            if self.softmax is not None and self.softmax != sm:
                raise ValueError(
                    f"softmax={self.softmax!r} conflicts with "
                    f"attn={self.attn!r}, whose softmax is {sm!r}; drop "
                    f"one of the two")
            # A preset entry replaces the BASE entry for the same key. An
            # entry that is neither BASE's nor the preset's was chosen by
            # the user and conflicts.
            for k, v in entries.items():
                if k in rule and rule[k] != v and rule[k] != BASE.get(k):
                    raise ValueError(
                        f"rule entry {k!r}={rule[k]!r} conflicts with "
                        f"attn={self.attn!r}, which writes {v!r}; drop one "
                        f"of the two")
                rule[k] = v
            object.__setattr__(self, 'softmax', sm)
        if self.softmax is None:
            object.__setattr__(self, 'softmax', 'passthrough')
        object.__setattr__(self, 'rule', rule)

        _validate_rule(self.rule)
        _validate_unary('softmax',    self.softmax,    SOFTMAX_HANDLERS)
        _validate_unary('layernorm',  self.layernorm,  LAYERNORM_HANDLERS)
        _validate_unary('activation', self.activation, ACTIVATION_HANDLERS)
        if not (0.0 < self.relevance_filter <= 1.0):
            raise ValueError(
                f"relevance_filter must be in (0, 1], "
                f"got {self.relevance_filter}")

    # ------------------------------------------------------------------ #
    # Canonical preset constructors                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def composite(cls, eps: float = 1e-6) -> 'LRPConfig':
        r"""Composite recipe: :math:`z^+` on conv, :math:`\epsilon` elsewhere.

        Montavon, Binder, Lapuschkin, Samek, Müller, "Layer-Wise
        Relevance Propagation: An Overview", Springer 2019.

        ``eps=1e-6`` damps near-zero denominators at the cost of a
        little absorbed relevance (vs the exact-conservation 1e-11
        class default).
        """
        return cls(rule={**BASE, 'ConvolutionBackward': 'zplus'}, eps=eps)

    @classmethod
    def attnlrp(cls, gamma: float = 0.25) -> 'LRPConfig':
        r"""AttnLRP recipe for transformers.

        Achtibat, Hatefi, Dreyer, Jain, Wiegand, Lapuschkin, Samek,
        "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
        Transformers", ICML 2024. :math:`\gamma` on conv,
        :math:`\epsilon` elsewhere, Jacobian softmax, :math:`y/x`
        activations.

        This is the paper's full-model recipe; it is a superset of
        ``LRPConfig(attn='attnlrp')``, which sets only the attention
        entries (:math:`\epsilon` ``BmmBackward``, Jacobian softmax)
        and leaves conv and activations at their defaults.
        """
        return cls(
            rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': gamma})},
            softmax='jacobian',
            activation='yx',
        )

    @classmethod
    def bilrp(cls) -> 'LRPConfig':
        r"""BiLRP per-dimension config: :data:`BASE` as is.

        Eberle, Büttner, Kräutli, Müller, Valleriani, Montavon,
        "Building and Interpreting Deep Similarity Models", IEEE TPAMI
        2022.
        """
        return cls()

    @classmethod
    def cplrp(cls) -> 'LRPConfig':
        r"""CP-LRP recipe: equivalent to ``LRPConfig(attn='cplrp')``.

        Ali, Schnake, Eberle, Montavon, Müller, Wolf, "XAI for
        Transformers: Better Explanations through Conservative
        Propagation", ICML 2022.
        """
        return cls(attn='cplrp')

    @classmethod
    def epsilon_alpha2_beta1(cls, eps: float = 1e-6) -> 'LRPConfig':
        r"""Alpha-beta rule with :math:`\alpha = 2`, :math:`\beta = 1` on
        every linear family; Bach et al. 2015, §2.2.

        ``eps=1e-6`` damps near-zero denominators at the cost of a
        little absorbed relevance (vs the exact-conservation 1e-11
        class default).
        """
        ab = ('alpha_beta', {'alpha': 2.0, 'beta': 1.0})
        return cls(rule={**BASE, 'AddmmBackward': ab, 'MmBackward': ab,
                         'ConvolutionBackward': ab}, eps=eps)
