"""
DWA-10 Context Packer.
Selects top-K anchors by utility density within a token budget,
then formats them as a compact system-prompt prefix for Claude.
"""

from __future__ import annotations
from typing import List, Tuple
from .anchor import Anchor
from .memory import MemoryStore

DEFAULT_BUDGET = 800         # max tokens for anchor context block
P0_RESERVED = 200            # reserved tokens for P0 anchors


def pack(store: MemoryStore, budget: int = DEFAULT_BUDGET) -> Tuple[str, List[Anchor]]:
    """
    Returns (context_block_str, selected_anchors).
    P0 anchors are always included first.
    Remaining budget filled by utility-density ranking.
    """
    store.decay_all()
    store.rebalance()
    store.prune_dead()

    all_anchors = store.all_active()

    p0 = [a for a in all_anchors if a.class_ == "P0"]
    rest = sorted(
        [a for a in all_anchors if a.class_ != "P0"],
        key=lambda a: a.utility(),
        reverse=True,
    )

    selected: List[Anchor] = []
    used_tokens = 0

    # Always include P0
    for a in p0:
        selected.append(a)
        used_tokens += a.token_estimate

    # Fill remaining budget with highest-utility anchors
    remaining = budget - used_tokens
    for a in rest:
        if used_tokens + a.token_estimate > budget:
            continue
        selected.append(a)
        used_tokens += a.token_estimate

    if not selected:
        return "", []

    lines = ["## DWA-10 Memory Context\n"]
    for a in selected:
        lines.append(a.context_line())
    lines.append("")  # trailing newline

    return "\n".join(lines), selected
