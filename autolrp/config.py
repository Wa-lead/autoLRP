r"""LRP configuration.

``rule`` is the one place distribution is decided: a dict with one entry
per rule-bearing node name (no version digit: ``'MulBackward'``, not
``'MulBackward0'``), plus optional entries keyed by a fact name (a label
an analyzer attaches to a node, such as ``'attention_weights'``). A fact
entry wins over the name entry on the nodes that carry the fact. There
is no ``'default'`` and there are no aliases: what the dict says is what
runs, and ``print(config.rule)`` is the whole answer.

Start from :data:`BASE` and override what you need. Presets are dict
fragments merged the same way; the last entry for a key wins::

    LRPConfig(rule={**BASE, 'bilinear': 'gradient_input'})
    LRPConfig(rule={**BASE, **CPLRP})
    LRPConfig(rule={**BASE, **on(ELEMENTWISE_NODES, 'yx'),
                            **on(LAYERNORM_NODES, 'detach_std')})

A key that is neither a node name nor a registered fact is an error at
construction, and so is a node-name entry whose value that node's table
cannot run. A fact entry's value is checked at construction only
against the union of every table's rule names (which table will run it
is not known until the fact lands on a node), so a fact entry whose
value the receiving node's table cannot run errors at install. A node
that no entry addresses is also an error at install.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable

from .backward.rules import RULES_FOR, VIRTUAL_RULES
from .nodes import ELEMENTWISE_NODES, SOFTMAX_NODES, canonical
from .backward.analysis import ANALYZERS
from .backward.resolve import RuleSpec, _normalize


def on(names: Iterable[str], spec: RuleSpec) -> Dict[str, RuleSpec]:
    r"""``{name: spec}`` for every name, a fragment to merge into ``rule``:
    ``{**BASE, **on(ELEMENTWISE_NODES, 'yx')}``."""
    return {name: spec for name in names}


# One entry per rule-bearing node name (each table's default), plus the
# entry that detaches a normalization statistic where the analyzer found
# one. A product attributes what comes from the input: one operand, or
# both with half the relevance each (the fact ``bilinear``); ``attribute``
# in an entry overrides that.
BASE: Dict[str, RuleSpec] = {
    **{name: table.default for name, table in RULES_FOR.items()},
    'statistic_operand': ('detach', {'by': 'statistic_operand'}),
}

# Attention presets: fragments keyed by the fact ``bilinear``, a product
# with both operands from the input (``BmmBackward`` by name is a bmm
# with one, cross-attention against a fixed memory).
CPLRP: Dict[str, RuleSpec] = {                        # Ali et al. 2022
    'bilinear': ('detach', {'by': 'attention_weights'}),
}
ATTNLRP: Dict[str, RuleSpec] = {                      # Achtibat et al. 2024
    **on(SOFTMAX_NODES, 'jacobian'),
}
UNIFORM: Dict[str, RuleSpec] = {'bilinear': 'gradient_input'}

_ALL_RULE_NAMES = set().union(*(set(table) for table in RULES_FOR.values()))


def _check_key(key) -> str:
    """Return ``'node'`` or ``'fact'``; raise for anything else."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"rule keys must be non-empty strings; got {key!r}")
    if key in ANALYZERS:
        return 'fact'
    if key in RULES_FOR:
        return 'node'
    if key != canonical(key) and canonical(key) in RULES_FOR:
        raise ValueError(
            f"rule key {key!r} carries a version digit; write "
            f"{canonical(key)!r}")
    hint = ''
    if key == 'default':
        hint = " There is no 'default': start from autolrp.BASE and override entries."
    raise ValueError(
        f"unknown rule key {key!r}: not a node name "
        f"{sorted(RULES_FOR)} and not a fact {sorted(ANALYZERS)}.{hint}")


def _check_name(key, spec, valid_names, table=None) -> None:
    name, kw = _normalize(spec)
    if callable(name):
        return
    if not isinstance(name, str):
        raise TypeError(
            f"rule entry {key!r}: name must be a string, got "
            f"{type(name).__name__}")
    if name in VIRTUAL_RULES:
        if table is not None and not table.two_operand:
            raise ValueError(
                f"rule entry {key!r}={spec!r}: this table has no side to "
                f"detach. Choices: {sorted(valid_names)}")
        if 'rule' in kw and kw['rule'] not in valid_names:
            raise ValueError(f"rule entry {key!r}={spec!r}: rule={kw['rule']!r} is not a choice here. Choices: {sorted(valid_names)}")
        by = kw.get('by')
        if not isinstance(by, str) or not by:
            raise ValueError(
                f"rule entry {key!r}={spec!r}: 'detach' needs by=<fact name>")
        if by not in ANALYZERS:
            raise ValueError(
                f"rule entry {key!r}={spec!r}: by={by!r} is not a registered "
                f"fact {sorted(ANALYZERS)}")
        return
    if kw.get('attribute', 'both') not in ('lhs', 'rhs', 'both'):
        raise ValueError(f"rule entry {key!r}={spec!r}: attribute must be 'lhs', 'rhs' or 'both'")
    if name not in valid_names:
        raise ValueError(
            f"rule entry {key!r}={spec!r}: {name!r} is not a choice "
            f"here. Choices: {sorted(valid_names)}")


def _validate_rule(mapping) -> None:
    for key, spec in mapping.items():
        kind = _check_key(key)
        table = RULES_FOR[key] if kind == 'node' else None
        valid = set(table) if table is not None else _ALL_RULE_NAMES
        _check_name(key, spec, valid, table)


@dataclass(frozen=True)
class LRPConfig:
    r"""Immutable configuration for one LRP attribution.

    Args:
        rule: Dict keyed by node name (no version digit) or fact name.
            Default: :data:`BASE`.
        eps: Stabilizer added to denominators.
        relevance_filter: Top-fraction relevance pass-through in
            ``(0, 1]``.
        capture_layers: When ``True``, :meth:`LRPTensor.lrp` returns a
            dict of per-node relevances.
    """
    rule: Dict[str, RuleSpec] = field(default_factory=lambda: dict(BASE))

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
                "from autolrp.BASE, e.g. rule={**BASE, 'ConvolutionBackward': 'zplus'}")
        object.__setattr__(self, 'rule', dict(self.rule))
        _validate_rule(self.rule)
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

        This is the paper's full-model recipe; :data:`ATTNLRP` alone
        sets only the attention entries (:math:`\epsilon`
        ``BmmBackward``, Jacobian softmax) and leaves conv and
        activations at their defaults.
        """
        return cls(rule={**BASE,
                         'ConvolutionBackward': ('gamma', {'gamma': gamma}),
                         **ATTNLRP,
                         **on(ELEMENTWISE_NODES, 'yx')})

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
        r"""CP-LRP recipe: ``LRPConfig(rule={**BASE, **CPLRP})``.

        Ali, Schnake, Eberle, Montavon, Müller, Wolf, "XAI for
        Transformers: Better Explanations through Conservative
        Propagation", ICML 2022.
        """
        return cls(rule={**BASE, **CPLRP})

    @classmethod
    def epsilon_alpha2_beta1(cls, eps: float = 1e-6) -> 'LRPConfig':
        r"""Alpha-beta rule with :math:`\alpha = 2`, :math:`\beta = 1` on
        every linear node; Bach et al. 2015, §2.2.

        ``eps=1e-6`` damps near-zero denominators at the cost of a
        little absorbed relevance (vs the exact-conservation 1e-11
        class default).
        """
        ab = ('alpha_beta', {'alpha': 2.0, 'beta': 1.0})
        return cls(rule={**BASE, 'AddmmBackward': ab, 'MmBackward': ab,
                         'ConvolutionBackward': ab}, eps=eps)
