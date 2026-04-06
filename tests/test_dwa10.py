"""
DWA-10 Free Tier Tests.
Run: pytest tests/ -v
"""

import time
import pytest
from dwa10.core.anchor import Anchor
from dwa10.core.memory import MemoryStore
from dwa10.core.packer import pack
from dwa10.core.extractor import extract_anchors, manual_anchor
from dwa10.core.summarizer import should_summarize, generate_summary
from dwa10.core import export as _export
from dwa10.tiers import DWATierError, require_pro


# ── ANCHOR ───────────────────────────────────────────────────────────────────

def test_anchor_creation():
    a = Anchor(content="My name is John", class_="P1")
    assert a.ash_id
    assert a.token_estimate > 0
    assert a.priority == 0.5

def test_anchor_decay_p1():
    a = Anchor(content="test", class_="P1", priority=0.5)
    a.last_reinforced = time.time() - 1000
    a.decay()
    assert a.priority < 0.5

def test_anchor_p0_no_decay():
    a = Anchor(content="critical", class_="P0", priority=1.0)
    a.last_reinforced = time.time() - 100000
    a.decay()
    assert a.priority == 1.0

def test_anchor_reinforce():
    a = Anchor(content="test", class_="P1", priority=0.5)
    a.reinforce()
    assert a.priority > 0.5
    assert a.usage_count == 1

def test_anchor_utility():
    a = Anchor(content="short", class_="P1", priority=0.8)
    assert a.utility() > 0

def test_anchor_roundtrip():
    a = Anchor(content="test content", class_="P0", scope="Global")
    d = a.to_dict()
    b = Anchor.from_dict(d)
    assert b.content == a.content
    assert b.class_ == a.class_
    assert b.ash_id == a.ash_id


# ── MEMORY STORE ─────────────────────────────────────────────────────────────

def test_store_add_and_retrieve():
    store = MemoryStore()
    a = Anchor(content="test fact", class_="P1")
    store.add(a)
    assert len(store.all_active()) == 1

def test_store_dedup_by_ash():
    store = MemoryStore()
    a = Anchor(content="same content", class_="P1")
    b = Anchor(content="same content", class_="P1", version=2)
    store.add(a)
    store.add(b)
    assert store.stats()["total"] == 1

def test_store_p2_goes_to_archival():
    store = MemoryStore()
    a = Anchor(content="low priority", class_="P2")
    store.add(a)
    assert len(store.archival) == 1
    assert len(store.core) == 0

def test_store_prune_dead():
    store = MemoryStore()
    a = Anchor(content="dying anchor", class_="P1", priority=0.001)
    store.add(a)
    pruned = store.prune_dead()
    assert pruned == 1
    assert store.stats()["total"] == 0


# ── PACKER ───────────────────────────────────────────────────────────────────

def test_pack_respects_budget():
    store = MemoryStore()
    for i in range(20):
        store.add(Anchor(content=f"fact number {i} is important", class_="P1"))
    context, selected = pack(store, budget=200)
    total_tokens = sum(a.token_estimate for a in selected)
    assert total_tokens <= 200

def test_pack_always_includes_p0():
    store = MemoryStore()
    store.add(Anchor(content="critical system rule", class_="P0", priority=1.0))
    for i in range(50):
        store.add(Anchor(content=f"low priority {i}", class_="P2", priority=0.1))
    context, selected = pack(store, budget=100)
    classes = [a.class_ for a in selected]
    assert "P0" in classes


# ── EXTRACTOR ────────────────────────────────────────────────────────────────

def test_extract_anchors_finds_facts():
    text = "My name is John and I prefer Python over JavaScript. My budget is $5000."
    anchors = extract_anchors(text)
    assert len(anchors) > 0

def test_manual_anchor_is_exact():
    a = manual_anchor("User is a senior developer", class_="P0")
    assert a.anchor_accuracy == "exact"
    assert a.class_ == "P0"
    assert a.priority == 1.0


# ── SUMMARIZER ───────────────────────────────────────────────────────────────

def test_should_summarize_by_count():
    assert should_summarize(15, 0.1) is True
    assert should_summarize(5, 0.1) is False

def test_should_summarize_by_window():
    assert should_summarize(3, 0.75) is True

def test_generate_summary_compresses_p2():
    store = MemoryStore()
    for i in range(5):
        store.add(Anchor(content=f"low priority fact {i} worth remembering", class_="P2"))
    summary = generate_summary(store, message_count=15)
    assert summary is not None
    assert "summary" in summary.content.lower()


# ── EXPORT / IMPORT ──────────────────────────────────────────────────────────

def test_export_json_structure():
    store = MemoryStore()
    store.add(Anchor(content="exportable fact", class_="P1"))
    data = _export.export_json(store)
    assert "anchors" in data
    assert "meta" in data
    assert data["meta"]["export_format_version"].startswith("DWA10")

def test_export_markdown_contains_content():
    store = MemoryStore()
    store.add(Anchor(content="markdown test fact", class_="P1"))
    md = _export.export_markdown(store)
    assert "DWA-10" in md
    assert "markdown test fact" in md

def test_export_import_roundtrip(tmp_path):
    store = MemoryStore()
    store.add(Anchor(content="roundtrip test content", class_="P1"))
    path = str(tmp_path / "test_memory")
    _export.save(store, path)
    new_store = MemoryStore()
    count = _export.load(new_store, path)
    assert count == 1
    assert new_store.stats()["total"] == 1


# ── TIER ENFORCEMENT ─────────────────────────────────────────────────────────

def test_pro_feature_raises_tier_error():
    with pytest.raises(DWATierError) as exc_info:
        require_pro("multi_agent_memory")
    assert "zulfr.com" in str(exc_info.value)
    assert "multi_agent_memory" in str(exc_info.value)
