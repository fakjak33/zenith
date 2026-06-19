import json

from zenith import store
from zenith.sources import SOURCES, enabled_sources
from zenith.classify import from_entry, _html_has_visual


def test_item_id_stable_and_url_normalized():
    a = store.item_id("https://Example.com/Post/?utm=1#frag", "T")
    b = store.item_id("https://example.com/Post", "T")
    assert a == b  # query/fragment/case ignored


def test_seen_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "seen.json"
    monkeypatch.setattr(store, "SEEN_JSON", p)
    store.save_seen({"a", "b"})
    assert store.load_seen() == {"a", "b"}


def test_archive_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ARCHIVE_DIR", tmp_path)
    items = [{"id": "x", "source": "S", "category": "insight", "visual": "text",
              "title": "Hi", "link": "https://e.com/a", "summary": "", "published": ""}]
    store.write_archive("2026-01-02", items)
    assert store.load_archive("2026-01-02") == items
    assert "2026-01-02" in store.archive_dates()


def test_dedupe_logic():
    seen = set()
    ids = [store.item_id(u) for u in ["https://e.com/1", "https://e.com/1", "https://e.com/2"]]
    new = [i for i in ids if not (i in seen or seen.add(i))]
    assert len(new) == 2  # duplicate dropped


def test_classify_visual_from_html():
    assert _html_has_visual("<p>hi</p><img src='x.png'>") is True
    assert _html_has_visual("<p>just text</p>") is False


def test_classify_entry_media():
    assert from_entry({"media_content": [{"url": "x.jpg"}]}) is True
    assert from_entry({"summary": "<p>text only</p>"}) is None


def test_registry_integrity():
    names = [s.name for s in SOURCES]
    assert len(names) == len(set(names))                  # unique names
    assert all(s.category in {"insight", "research", "news"} for s in SOURCES)
    assert all(s.kind == "rss" for s in SOURCES)
    assert all(s.url.startswith("http") for s in enabled_sources())  # enabled have real urls
    # decent breadth across categories
    cats = {s.category for s in SOURCES}
    assert {"insight", "research", "news"}.issubset(cats)
