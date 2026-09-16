r"""From the config's rule dict to the function that runs at one node.

One route for every rule-bearing node. The node's name without its
version digit (``'MulBackward'`` for ``MulBackward0``) selects its rule
table in :data:`~autolrp.backward.rules.RULES_FOR`. The dict entry that
addresses the node names a rule in that table: an entry keyed by a fact
the node carries (a label an analyzer attached, such as
``'attention_weights'``) wins over the entry keyed by the node's name;
nothing else matches, no aliases, no substrings, no ``'default'``. A
node that no entry addresses is an error, and so is a node that two
fact entries address. The one virtual name ``'detach'`` becomes the
table's default rule attributing the operand the fact its ``by=`` names
does not point at, or that default alone when the node lacks the fact.
``_TRACE`` records ``(key, rule name)`` while
:func:`~autolrp.backward.engine.explain` runs.
"""
from typing import Any, Dict, Tuple, Union

from ..nodes import canonical, saved_tensors
from .graph import node_facts
from .rules import RULES_FOR

RuleSpec = Union[str, Tuple[str, Dict[str, Any]]]


def _normalize(spec) -> Tuple[Any, Dict[str, Any]]:
    """``'epsilon'`` -> ``('epsilon', {})``; ``('gamma', {...})`` -> as is."""
    if isinstance(spec, tuple):
        return spec[0], dict(spec[1])
    return spec, {}


_TRACE = None                     # list of (key, what) while explain() runs


def _trace(key, what):
    if _TRACE is not None:
        _TRACE.append((key, what))


def resolve(mapping: dict, node) -> Tuple[Any, Dict[str, Any]]:
    """The rule function and its keyword arguments for ``node`` under the
    config's ``rule`` dict ``mapping``."""
    name = node.name()
    canon = canonical(name)
    table = RULES_FOR.get(canon)
    if table is None:
        raise ValueError(
            f"{name} runs no rule: it is not a key of RULES_FOR "
            f"{sorted(RULES_FOR)}")
    facts = node_facts(node)
    hits = [k for k in mapping if k in facts]
    if len(hits) > 1:
        raise ValueError(
            f"two entries address {name}: {hits}. The node carries both "
            f"facts; keep one of the two entries")
    if hits:
        key = hits[0]
    elif canon in mapping:
        key = canon
    else:
        raise ValueError(
            f"no entry for node {name!r}. Add {canon!r} (or a fact the node "
            f"carries) to the mapping. Keys present: {sorted(mapping)}")
    spec = mapping[key]
    rule, kw = _normalize(spec)
    if rule == 'detach':
        rule, kw = _detach_side(kw, facts, table, name)
    if callable(rule):
        fn = rule
    elif rule not in table:
        raise ValueError(
            f"entry {key!r}={spec!r} addresses {name}, whose table cannot "
            f"run {rule!r}. Valid here: {sorted(table)} and 'detach'")
    else:
        fn = table[rule]
    if table.two_operand and 'attribute' not in kw:
        attribute = _attribute(node, facts)
        if attribute is not None:
            kw = {**kw, 'attribute': attribute}
    what = getattr(fn, '__name__', repr(fn))
    _trace(key, f"{what} ({kw['attribute']})" if 'attribute' in kw else what)
    return fn, kw


def _attribute(node, facts):
    r"""What the node's facts say a two-operand rule attributes: both under
    ``bilinear``; the other operand under ``weight_operand`` (the weight's
    position); ``None`` when they say nothing, and the rule's own default
    stands."""
    if facts.get('bilinear'):
        return 'both'
    weight = facts.get('weight_operand')
    if weight is not None:
        first = next(t.position for t in saved_tensors(node) if t.position is not None)
        return 'rhs' if weight == first else 'lhs'
    return None


def _detach_side(kw, facts, table, name):
    r"""``('detach', {'by': <fact>, 'rule': <rule>})`` on a two-operand
    table: the table's default (or ``rule``) attributing the operand the
    fact does not name; on a node without the fact, the default with the
    graph's own attribute."""
    kw = dict(kw)                     # never mutate the config's entry
    by = kw.pop('by', None)
    rule = kw.pop('rule', table.default)
    if not isinstance(by, str) or not by:
        raise ValueError(
            "'detach' needs by=<fact name>, e.g. "
            "('detach', {'by': 'attention_weights'})")
    if not table.two_operand:
        raise ValueError(f"'detach' on {name}: its table has no side to detach")
    if by not in facts:
        return rule, kw
    position = facts[by]
    if isinstance(position, bool) or position not in (0, 1):
        raise ValueError(
            f"'detach' by={by!r}: on {name} that fact is {position!r}, "
            f"not a side (0 for the first operand, 1 for the second)")
    return rule, {**kw, 'attribute': 'rhs' if position == 0 else 'lhs'}
