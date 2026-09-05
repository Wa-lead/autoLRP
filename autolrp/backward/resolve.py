r"""From a config mapping to the rule that runs at one node.

Two steps. :func:`match` picks the key: a fact name (a label an
analyzer attached to the node, such as ``'weights_operand'``) or the
node's own name without its version digits (``'MulBackward'`` for
``MulBackward0``). Nothing else matches: no aliases, no substrings, no
``'default'``; a fact key wins over the name key; a node that no key
addresses is an error, and so is a node that two fact keys address.
:func:`resolve_rule` then turns the entry into a function from the
family's table, resolving the virtual ``'detach'`` to a side from the
fact its ``by=`` names. ``_TRACE`` records what was resolved while
:func:`~autolrp.backward.engine.explain` runs.
"""
from typing import Any, Dict, Tuple, Union

from .graph import node_facts

RuleSpec = Union[str, Tuple[str, Dict[str, Any]]]


def canonical(name: str) -> str:
    """``'MulBackward0'`` -> ``'MulBackward'``."""
    return name.rstrip('0123456789')


def _normalize(spec) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(spec, tuple):
        return spec[0], dict(spec[1])
    return spec, {}


def match(mapping: dict, node) -> str:
    """Return the one key of ``mapping`` that addresses ``node``."""
    name = node.name()
    canon = canonical(name)
    facts = node_facts(node)
    hits = [k for k in mapping if k in facts]
    if len(hits) > 1:
        raise ValueError(
            f"two entries address {name}: {hits}. The node carries both "
            f"facts; keep one of the two entries")
    if hits:
        return hits[0]
    if canon in mapping:
        return canon
    raise ValueError(
        f"no entry for node {name!r}. Add {canon!r} (or a fact the node "
        f"carries) to the mapping. Keys present: {sorted(mapping)}")


def resolve(spec_or_dict, node) -> Tuple[Any, Dict[str, Any]]:
    """A bare spec applies to every node; a dict is matched exactly."""
    if isinstance(spec_or_dict, dict):
        return _normalize(spec_or_dict[match(spec_or_dict, node)])
    return _normalize(spec_or_dict)


_TRACE = None                     # list of (key, what) while explain() runs


def _trace(key, what):
    if _TRACE is not None:
        _TRACE.append((key, what))


def resolve_rule(mapping, node, registry):
    r"""The rule function and keyword arguments for ``node`` from the
    config's ``rule`` dict, looked up in ``registry`` (the family's table).
    The entry is picked by :func:`~autolrp.backward.resolve.match`; a
    ``'detach'`` entry becomes ``detach_lhs``/``detach_rhs`` from the slot
    fact named by its ``by=``, or the family's fallback when the node has
    no such fact. A name the family cannot run is an error.
    """
    key = match(mapping, node)
    spec = mapping[key]
    name, kw = _normalize(spec)
    if name == 'detach':
        name, kw = _detach_side(kw, node, registry)
    if callable(name):
        fn = name
    elif name not in registry:
        raise ValueError(
            f"entry {key!r}={spec!r} addresses {node.name()}, whose family "
            f"cannot run {name!r}. Valid here: {sorted(registry)} and "
            f"'detach'")
    else:
        fn = registry[name]
    _trace(key, getattr(fn, '__name__', repr(fn)))
    return fn, kw


def _detach_side(kw, node, registry):
    kw = dict(kw)                     # never mutate the config's entry
    by = kw.pop('by', None)
    if not isinstance(by, str) or not by:
        raise ValueError(
            "'detach' needs by=<fact name>, e.g. "
            "('detach', {'by': 'weights_operand'}); 'detach_lhs' / "
            "'detach_rhs' name a side directly")
    facts = node_facts(node)
    if by not in facts:
        return registry.default, {}   # no referent: the family fallback
    slot = facts[by]
    if isinstance(slot, bool) or slot not in (0, 1):
        raise ValueError(
            f"'detach' by={by!r}: on {node.name()} that fact is {slot!r}, "
            f"not a side (0 for the left operand, 1 for the right)")
    return ('detach_lhs' if slot == 0 else 'detach_rhs'), kw
