import json

from zenith import store, extract, apify, firecrawl, config
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
    assert all(s.kind in {"rss", "html"} for s in SOURCES)
    assert all(s.url.startswith("http") for s in enabled_sources())  # enabled have real urls
    # html sources must declare how to find articles (pattern) or be a clean hub
    assert all(isinstance(s.link_pattern, str) for s in SOURCES)
    # decent breadth across categories
    cats = {s.category for s in SOURCES}
    assert {"insight", "research", "news"}.issubset(cats)


def test_extract_links_basic():
    html = """
    <html><body><main>
      <a href="/insights/article/alpha-and-beta-in-2026">Alpha and Beta in 2026 Outlook</a>
      <a href="/insights/article/the-case-for-trend-following-now">The Case for Trend Following Now</a>
      <a href="/about">About</a>                         <!-- junk: too short / nav -->
      <a href="https://twitter.com/x">Follow us on Twitter</a>  <!-- off-domain -->
      <a href="/insights/logo.png">image</a>             <!-- skipped ext -->
    </main></body></html>
    """
    out = extract.extract_links(html, "https://firm.com/insights", r"/insights/")
    links = {o["link"] for o in out}
    assert "https://firm.com/insights/article/alpha-and-beta-in-2026" in links
    assert len(out) == 2                                   # only the two real articles
    assert all(o["link"].startswith("https://firm.com/") for o in out)


def test_extract_links_respects_pattern_and_dedup():
    html = ('<a href="/research/x-paper-on-momentum-factors-2026">Momentum Factors Paper 2026</a>'
            '<a href="/research/x-paper-on-momentum-factors-2026/">Momentum Factors Paper 2026</a>'
            '<a href="/blog/unrelated-short">nope</a>')
    out = extract.extract_links(html, "https://firm.com/", r"/research/")
    assert len(out) == 1                                   # pattern filters + dedup trailing slash


def test_apify_disabled_without_token(monkeypatch):
    monkeypatch.setattr(config, "APIFY_ENABLED", False)
    monkeypatch.setattr(config, "APIFY_TOKEN", "")
    html, note = apify.fetch_html("https://example.com")
    assert html is None and note == "apify:disabled"


def test_apify_budget_gate(monkeypatch):
    # over budget -> never calls the network
    monkeypatch.setattr(config, "APIFY_ENABLED", True)
    monkeypatch.setattr(config, "APIFY_TOKEN", "x")
    monkeypatch.setattr(apify, "monthly_spend_usd", lambda *a, **k: 999.0)
    assert apify.budget_ok() is False
    html, note = apify.fetch_html("https://example.com")
    assert html is None and note == "apify:budget"


def test_apify_usage_summary_shape():
    u = apify.usage_summary()
    assert {"enabled", "actor", "crawler", "budget_usd", "calls_this_run"} <= set(u)


def test_firecrawl_disabled_without_key(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", "")
    html, note = firecrawl.fetch_html("https://example.com")
    assert html is None and note == "firecrawl:disabled"
    assert firecrawl.enabled() is False


def test_get_html_respects_robots(monkeypatch):
    from zenith import fetch
    monkeypatch.setattr(fetch, "allowed", lambda url: False)
    html, via = fetch.get_html("https://example.com/blocked")
    assert html is None and via == "robots"
