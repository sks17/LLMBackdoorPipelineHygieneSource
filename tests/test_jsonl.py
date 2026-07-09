"""Tests for JSONL read/write/append, including pydantic-model rows."""

from __future__ import annotations

from trigger_audit.io.jsonl import append_jsonl, read_jsonl, read_jsonl_as, write_jsonl
from trigger_audit.schemas.triggers import TriggerSpec, TriggerType


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"a": 1, "text": "héllo"}, {"a": 2, "text": "world"}]
    assert write_jsonl(path, rows) == 2
    assert read_jsonl(path) == rows


def test_append_creates_and_extends(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, [{"a": 2}, {"a": 3}])
    assert [r["a"] for r in read_jsonl(path)] == [1, 2, 3]


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_leading_utf8_bom_is_tolerated(tmp_path):
    # Windows tools (PowerShell `Set-Content -Encoding utf8`, Notepad) prepend a UTF-8 BOM that
    # json.loads rejects; the reader must strip it. CRLF line endings must read fine too.
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\r\n{"a": 2}\r\n', encoding="utf-8-sig")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_pydantic_models_roundtrip(tmp_path):
    path = tmp_path / "triggers.jsonl"
    triggers = [
        TriggerSpec(trigger_id="t1", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_A"),
        TriggerSpec(trigger_id="t2", trigger_type=TriggerType.NATURAL_PHRASE, text="silver river"),
    ]
    write_jsonl(path, triggers)
    loaded = read_jsonl_as(path, TriggerSpec)
    assert [t.trigger_id for t in loaded] == ["t1", "t2"]
    assert loaded[1].trigger_type is TriggerType.NATURAL_PHRASE
