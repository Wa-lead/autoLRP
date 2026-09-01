r"""Match a config mapping to one node, exactly.

A mapping key is one of two things: a fact name (a label an analyzer
attached to the node, such as ``'weights_operand'``) or the node's own
name with the trailing version digits removed (``'MulBackward'`` for
``MulBackward0``). Nothing else matches: no aliases, no substrings, no
``'default'``. A fact key wins over the name key. A node that no key
addresses is an error, and so is a node that two fact keys address.
"""
from typing import Any, Dict, Tuple, Union

from .analysis import node_facts

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
