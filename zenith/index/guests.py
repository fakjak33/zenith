"""INDEX Phase 2 — guest extraction from harvested podcast episodes.

WHAT THIS DOES AND DOES NOT CLAIM. Every field produced here is derived from
text the publisher actually wrote in the episode title or description. Nothing
is inferred from outside knowledge, and nothing is invented. When a parse is
uncertain it is KEPT but labelled `low` confidence, and only parses at or above
``INDEX_GUEST_MIN_CONFIDENCE`` are promoted into Person entities in the
directory — a directory full of misparsed title fragments would be worse than a
smaller correct one, so the bar for becoming a permanent entry is higher than
the bar for being recorded against an episode.

THE PATTERNS ARE EMPIRICAL. They were written by reading the real archives, not
guessed: a first pass assumed markers like "feat." and "w/" that barely occur,
while the shapes that actually dominate are quite different per show —

  Alpha Exchange      "Ulrike Hoffmann-Burchardi, CIO Americas, UBS"   name, role, firm
  Flirting w/ Models  "Adam Butler - Questioning the Quant Orthodoxy"  name before a dash
  Capital Allocators  "David Lyon - Hybrid Capital Solutions (EP.471)" name before a dash
  Top Traders         "SI196: Where Next for Trend Following? ft. Rob Carver"
  The Long View       "Michael Mauboussin: Finding Easy Games"         name before a colon
  Meb Faber           "Victor Haghani on Predicting the Market | #588"
  Monetary Matters    "... | Alan Strauss of Crystal Capital Partners"
  The Derivative      "The principles of VIX trading with Alex Orus of Principalium"
  Odd Lots            "Samanth Subramanian on the Undersea Cables ..."

THE VALIDATOR IS THE IMPORTANT PART. Any of these patterns will happily capture
a title fragment — "Where Next for Trend Following", "In Case You Missed It",
"Nature of Intelligence" — so a candidate only survives if it looks like a
person's name: 2-4 tokens, capitalised, no digits, no stopwords, not an
organisation, not shouted. Firm-shaped strings ("Aristides Capital") are
rejected as people and captured as FIRMS instead, which is where they belong.
"""

from __future__ import annotations

import re

from ..config import INDEX_GUEST_MIN_CONFIDENCE

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

# --------------------------------------------------------------------------
# Name validation
# --------------------------------------------------------------------------

# Lowercase particles that legitimately appear inside a surname.
_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da", "dos",
              "du", "la", "le", "el", "al", "bin", "ibn", "ter", "ten", "y"}

# Titles stripped from the front of a candidate before validation.
_HONORIFICS = {"prof", "prof.", "professor", "dr", "dr.", "sir", "dame", "mr", "mr.",
               "mrs", "mrs.", "ms", "ms.", "lord", "rev", "rev.", "hon", "gen",
               "senator", "governor", "president", "nobel"}

# Any of these words means the string is a phrase or an organisation, not a
# person. Kept deliberately broad: a false negative loses one guest, a false
# positive puts a nonsense entry in a permanent directory.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "without", "from", "into",
    "of", "in", "on", "at", "to", "by", "as", "is", "are", "was", "were", "be",
    "how", "why", "what", "when", "where", "who", "which", "that", "this", "these",
    "we", "our", "us", "you", "your", "they", "their", "it", "its", "his", "her",
    "not", "no", "all", "more", "most", "less", "best", "worst", "new", "old",
    "part", "episode", "ep", "season", "series", "special", "replay", "bonus",
    "radio", "show", "podcast", "interview", "conversation", "live", "highlights",
    "case", "missed", "greatest", "hits", "inside", "understanding", "lessons",
    "markets", "market", "investing", "investment", "trading", "trade", "money",
    "crypto", "bitcoin", "stocks", "bonds", "economy", "inflation", "recession",
    "rates", "risk", "returns", "portfolio", "alpha", "beta", "volatility",
    "nature", "physics", "biology", "intelligence", "history", "theory", "future",
    "sixth", "bureau", "friend", "world", "america", "american", "china", "chinese",
    "europe", "european", "global", "year", "years", "day", "days", "week", "month",
}

# Organisation markers — a candidate containing one of these is a firm, not a
# person. Firms are still captured, just into the right field.
_ORG_WORDS = {
    "capital", "partners", "management", "advisors", "advisers", "investments",
    "investment", "asset", "assets", "fund", "funds", "group", "holdings", "llc",
    "lp", "ltd", "inc", "plc", "gmbh", "associates", "securities", "bank",
    "research", "analytics", "technologies", "technology", "solutions", "systems",
    "ventures", "trust", "foundation", "university", "institute", "college",
    "school", "company", "corporation", "corp", "global", "international",
    "financial", "finance", "wealth", "equity", "credit", "capital's",
}

_INITIAL_RE = re.compile(r"^[A-Z]\.?$")
_NAME_TOKEN_RE = re.compile(r"^[A-Z][\w'’\-]*$", re.UNICODE)
_HAS_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+")

# Characters that never appear inside a person's name but do appear when a title
# fragment has leaked into the candidate: "Michael Mauboussin | AI",
# ‘Banks’ "Considerable" Exposure to’, "Understanding the 401(k) Market –".
_FORBIDDEN_CHARS = set('|"“”()[]{}<>:;/\\!?*=+—–_@#$%^~`')

# A possessive at the start is an organisation attributing a person -
# "JPMorgan's Jay Barry". Real surnames do not end in 's.
_POSSESSIVE_RE = re.compile(r"['’]s$", re.I)

# Gerunds and nominalisations head phrases -- "Energizing Lives", "Understanding
# the Market" -- so a candidate STARTING with one is not a name.
#
# This is checked on the FIRST token only. Applying it to every token rejected
# "David Harding", "Alex Fleming" and every other perfectly ordinary surname
# ending in -ing, which then compounded: with Harding failing the person test he
# was captured as Michael Adam's EMPLOYER instead. The broader guard against
# phrase-shaped candidates is the corpus vocabulary filter, which is
# self-calibrating and does not need this rule to be aggressive.
_VERBISH_SUFFIXES = ("ing", "tion", "sion", "ment", "ness")

# Apostrophes are legitimate only in these name prefixes.
_APOSTROPHE_OK = ("o'", "o’", "d'", "d’", "l'", "l’")

# Strip characters commonly left clinging to a parsed fragment.
_EDGE_CHARS = " .,:;–—- ‘’“”'\""


def _strip_honorifics(tokens: list[str]) -> list[str]:
    out = list(tokens)
    bare = {h.strip(".") for h in _HONORIFICS}
    while out and out[0].lower().strip(".") in bare:
        out.pop(0)
    return out


def canonical_name(name: str) -> str:
    """The form stored in the directory: honorifics removed, whitespace tidied.

    Keeping "Prof. William Goetzmann" and "William Goetzmann" as two people
    would fragment the graph, so the honorific is dropped from the STORED name
    rather than merely ignored while validating.
    """
    raw = _WS_RE.sub(" ", str(name or "").strip(_EDGE_CHARS))
    return " ".join(_strip_honorifics(raw.split()))


def _token_is_verbish(tok: str) -> bool:
    low = tok.lower().strip(".,'’")
    return len(low) >= 6 and low.endswith(_VERBISH_SUFFIXES)


def looks_like_person(name: str) -> bool:
    """True if ``name`` plausibly names a human being.

    This is the gate that keeps title fragments out of the directory. It is
    deliberately conservative - losing a real guest costs one row, while
    admitting a fragment costs the credibility of every row. An audit of the
    real 6,515-episode corpus showed roughly a quarter of raw candidates were
    fragments before these rules were tightened.
    """
    raw = str(name or "").strip(_EDGE_CHARS)
    if not raw or _HAS_DIGIT_RE.search(raw):
        return False
    if len(raw) < 4 or len(raw) > 45:
        return False
    if _FORBIDDEN_CHARS & set(raw):
        return False
    tokens = _strip_honorifics(raw.split())
    if not 2 <= len(tokens) <= 4:
        return False
    # "KEEP THE DEFERRED SALES CHARGE" - shouted titles are never names.
    letters = [c for c in raw if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return False
    lowered = {t.lower().strip(".,'’") for t in tokens}
    if lowered & _STOPWORDS:
        return False
    if lowered & _ORG_WORDS:
        return False
    real = 0
    for tok in tokens:
        low = tok.lower().strip(".")
        if low in _PARTICLES:
            continue
        if _INITIAL_RE.match(tok):
            continue
        if not _NAME_TOKEN_RE.match(tok):
            return False
        if _POSSESSIVE_RE.search(tok):
            return False
        # An all-caps acronym is never part of a personal name, but is common in
        # title fragments: "Top IPO Scholar", "The ETF Story". Initials are
        # single letters and are handled above.
        if len(tok.strip(".")) > 1 and tok.strip(".").isupper():
            return False
        if real == 0 and _token_is_verbish(tok):
            return False
        if ("'" in tok or "’" in tok) and not tok.lower().startswith(_APOSTROPHE_OK):
            return False
        real += 1
    # At least two substantive name tokens - "J. Smith" alone is too thin, and a
    # lone first name ("ft. Jerry") must not become a directory entry.
    return real >= 2


# Words that only appear when a SENTENCE has been captured instead of a firm
# name -- "his episode, we speak with Dyn", "joins the show to discuss".
_PROSE_WORDS = {
    "we", "i", "he", "she", "they", "his", "her", "their", "our", "you", "this",
    "that", "these", "those", "speak", "speaks", "joins", "join", "joined",
    "discuss", "discusses", "talks", "talk", "explains", "explain", "returns",
    "welcome", "welcomes", "episode", "today", "here", "back", "about", "why",
    "how", "what", "who", "where", "when", "is", "are", "was", "were", "has",
    "have", "had", "will", "would", "can", "could", "also", "just", "very",
}

# Cut a captured firm at the first character that signals the phrase has run on
# past the name: a dash introduces a subtitle, a comma introduces another
# entity, and the rest are never inside a firm name.
_FIRM_CUT_RE = re.compile(r"\s+[–—-]\s+|\s*[|;:()\[\]\"“”]|,")


# Words that describe a ROLE rather than name an organisation. A string made
# entirely of these is a job title ("Managing Director", "Head of Equity
# Derivatives Strategy") and must not be recorded as an employer.
_ROLE_ONLY_WORDS = {
    "chief", "executive", "officer", "managing", "director", "head", "deputy",
    "senior", "vice", "associate", "principal", "portfolio", "manager",
    "strategist", "analyst", "economist", "president", "chairman", "chairwoman",
    "founder", "cofounder", "co-founder", "partner", "strategy", "equity",
    "derivatives", "sales", "trading", "operations", "research", "investing",
}

# "Partner at Ruffer Investment Management" -> the firm is what follows.
_ROLE_PREFIX_RE = re.compile(r"^(?P<role>[A-Za-z'’\- ]{3,60}?)\s+(?:at|of|for)\s+(?P<firm>[A-Z].*)$")


def clean_firm(text: str, positional: bool = False) -> str:
    """Normalise a captured organisation string, or return "" if it is not one.

    Firm capture is the noisiest part of extraction: a loose regex over show
    notes will happily return "his episode, we speak with Dyn", "the Acquirers
    Fund" or another guest's name. This trims the string to its plausible
    organisation prefix and then requires it to pass ``looks_like_org``.

    ``positional=True`` says the CALLER already knows this slot holds an
    organisation because of where it sat in the text -- after " of " in
    "Alex Orus of Principalium", or in the firm column of "Name - Firm (role)".
    In that case the shape tests are relaxed: single-word firms are allowed
    (Transtrend, Winton, Principalium are all real firms already in this
    catalog, and all were being silently dropped), and the "does this look like
    a person" veto is skipped, since position has already settled the question.
    Prose and role-only rejections still apply.
    """
    raw = _WS_RE.sub(" ", str(text or "").strip(_EDGE_CHARS))
    if not raw:
        return ""
    raw = _FIRM_CUT_RE.split(raw)[0].strip(_EDGE_CHARS)
    # A leading article is not part of the name ("the Acquirers Fund").
    if raw.lower().startswith("the "):
        raw = raw[4:].strip()
    # "Research at Dimensional Fund Advisors" -> "Dimensional Fund Advisors".
    for _ in range(3):
        m = _ROLE_PREFIX_RE.match(raw)
        if not m or not {w.lower() for w in m.group("role").split()} <= _ROLE_ONLY_WORDS:
            break
        raw = m.group("firm").strip(_EDGE_CHARS)
        if raw.lower().startswith("the "):
            raw = raw[4:].strip()
    # A long title of the form "<role phrase> at <Firm>" survives the loop above
    # when the role phrase contains words outside the role vocabulary ("Head of
    # Consilient Research at Counterpoint Global"). The employer is whatever
    # follows the LAST " at ", so prefer that when it stands up on its own.
    if " at " in raw:
        tail = raw.rsplit(" at ", 1)[1].strip(_EDGE_CHARS)
        if tail and tail[:1].isupper() and not looks_like_person(tail):
            raw = tail
    # A pure job title names no employer at all.
    if raw and {w.lower().strip(".,") for w in raw.split()} <= _ROLE_ONLY_WORDS:
        return ""
    if not raw or not raw[:1].isupper():
        return ""
    toks = raw.split()
    if not 1 <= len(toks) <= 6:
        return ""
    if {t.lower().strip(".,'’") for t in toks} & _PROSE_WORDS:
        return ""
    if positional:
        # Position already established this is an organisation; just require it
        # to be capitalised and not obviously a sentence.
        return raw if all(t[:1].isupper() or t.lower() in _PARTICLES
                          for t in toks) else ""
    if looks_like_person(raw):
        return ""
    return raw if looks_like_org(raw) else ""


def looks_like_org(text: str) -> bool:
    """True if ``text`` reads as an organisation name."""
    raw = str(text or "").strip(_EDGE_CHARS)
    if not raw or len(raw) < 3 or len(raw) > 60:
        return False
    if _HAS_DIGIT_RE.search(raw) and not any(
            w.lower() in _ORG_WORDS for w in raw.split()):
        return False
    words = {w.lower().strip(".,'’") for w in raw.split()}
    if words & _PROSE_WORDS:
        return False
    if words & _ORG_WORDS:
        return True
    # A capitalised multi-word string with no stopwords is plausibly a firm.
    toks = raw.split()
    return (2 <= len(toks) <= 6 and not (words & _STOPWORDS)
            and all(t[:1].isupper() or t.lower() in _PARTICLES for t in toks))


# --------------------------------------------------------------------------
# Splitting multi-guest strings
# --------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"\s*(?:,\s*and\s+|\s+and\s+|\s*&\s*|\s*\+\s*|\s*/\s*)\s*", re.I)


def split_people(chunk: str) -> list[str]:
    """Split 'Andrew Beer & Tom Wrobel' / 'Gene Munster and Doug Clinton'.

    Comma is NOT a general separator here: "Robert Keith, Beartooth Group" and
    "Anastasia Titarchuk, CIO" both use a comma to introduce a ROLE or FIRM, so
    treating it as a person separator would manufacture people out of job titles.
    """
    parts = [p.strip() for p in _SPLIT_RE.split(str(chunk or "")) if p.strip()]
    return parts or ([chunk.strip()] if str(chunk or "").strip() else [])


# --------------------------------------------------------------------------
# " ... of <Firm>" — a firm stated right next to the name
# --------------------------------------------------------------------------
_OF_RE = re.compile(r"^(?P<name>.+?)\s+of\s+(?P<firm>[A-Z][^,|]{2,60})$")


def _split_of_firm(chunk: str) -> tuple[str, str]:
    m = _OF_RE.match(str(chunk or "").strip())
    if not m:
        return str(chunk or "").strip(), ""
    firm = clean_firm(m.group("firm"), positional=True)
    return m.group("name").strip(), firm


# --------------------------------------------------------------------------
# Extraction strategies. Each returns [(name, role, firm)] candidates.
# --------------------------------------------------------------------------
_DASHES = "-‐‑‒–—"
_DASH_SPLIT_RE = re.compile(rf"\s+[{_DASHES}]\s+")
_TRAILING_TAG_RE = re.compile(r"\s*[\(\[][^)\]]*[\)\]]\s*$")
_LEADING_TAG_RE = re.compile(r"^\s*(?:\[[^\]]*\]|#\d+|\d+)\s*[-–:]?\s*")
_EP_MARK_RE = re.compile(r"\b(?:EP|Ep|ep)\.?\s*\d+|\bS\d+E\d+\b|\|\s*#\d+\s*$")


def _clean_segment(text: str) -> str:
    """Drop episode numbering and trailing bracketed tags."""
    s = _EP_MARK_RE.sub("", str(text or ""))
    prev = None
    while prev != s:
        prev = s
        s = _TRAILING_TAG_RE.sub("", s).strip()
    return _LEADING_TAG_RE.sub("", s).strip()


def _p_name_role_firm(title: str) -> list[tuple[str, str, str]]:
    """'Benn Eifert, Founder and CIO, QVR Advisors' — Alpha Exchange's shape."""
    parts = [p.strip() for p in _clean_segment(title).split(",")]
    if len(parts) < 2:
        return []
    name = parts[0]
    # This strategy MUST validate its own name. It bypassed the shared
    # validator in an earlier version, which is how "Trends" (from "Trends,
    # Tall Heads, and Transformations with ...") became a directory entry.
    if not looks_like_person(name):
        return []
    role = parts[1] if len(parts) > 1 else ""
    firm = ", ".join(parts[2:]).strip() if len(parts) > 2 else ""
    if not firm and clean_firm(role, positional=True):
        role, firm = "", role
    return [(canonical_name(name), role.strip(" .,")[:80],
             clean_firm(firm, positional=True))]


def _p_ft_suffix(title: str) -> list[tuple[str, str, str]]:
    """'SI196: Where Next for Trend Following? ft. Rob Carver'."""
    m = re.search(r"\b(?:ft\.?|feat\.?|featuring)\s+(?P<who>.+)$", title, re.I)
    if not m:
        return []
    return _people_from_chunk(_clean_segment(m.group("who")))


def _p_with_suffix(title: str) -> list[tuple[str, str, str]]:
    """'The principles of VIX trading with Alex Orus of Principalium'."""
    m = re.search(r"\s(?:with|w/)\s+(?P<who>.+)$", title, re.I)
    if not m:
        return []
    return _people_from_chunk(_clean_segment(m.group("who")))


def _p_dash_prefix(title: str) -> list[tuple[str, str, str]]:
    """'Adam Butler - Questioning the Quant Orthodoxy (S5E13)'."""
    body = _clean_segment(title)
    parts = _DASH_SPLIT_RE.split(body, maxsplit=1)
    if len(parts) < 2:
        return []
    return _people_from_chunk(parts[0])


def _p_colon_prefix(title: str) -> list[tuple[str, str, str]]:
    """'Michael Mauboussin: Finding Easy Games'. Also handles a prefixed series
    label — 'Understanding Crypto 5: Stephen Diehl: The Case Against Crypto' —
    by trying each colon-delimited segment from the left."""
    body = _clean_segment(title)
    segs = [s.strip() for s in body.split(":") if s.strip()]
    for seg in segs[:2]:
        found = _people_from_chunk(seg)
        if found:
            return found
    return []


def _p_on_infix(title: str) -> list[tuple[str, str, str]]:
    """'Victor Haghani on Predicting the Market' / 'Dani Bassett & Perry Zurn on ...'."""
    m = re.match(r"^(?P<who>[^|:]{4,70}?)\s+on\s+\S", _clean_segment(title))
    if not m:
        return []
    return _people_from_chunk(m.group("who"))


def _p_pipe_segments(title: str) -> list[tuple[str, str, str]]:
    """'... | Alan Strauss of Crystal Capital Partners' — pipe-delimited shows.

    Tries every segment, because the guest sits in different positions across
    Meb Faber, Monetary Matters and Other People's Money.
    """
    if "|" not in title:
        return []
    out: list[tuple[str, str, str]] = []
    for seg in title.split("|"):
        seg = _clean_segment(seg)
        if not seg:
            continue
        found = _people_from_chunk(seg) or _p_on_infix(seg)
        for cand in found:
            if cand not in out:
                out.append(cand)
    return out


def _p_paren_with(title: str) -> list[tuple[str, str, str]]:
    """'... (w/ David Booth)' — Rational Reminder's occasional shape."""
    out: list[tuple[str, str, str]] = []
    for m in re.finditer(r"[\(\[]\s*(?:w/|with)\s+(?P<who>[^)\]]{4,70})[\)\]]", title, re.I):
        out.extend(_people_from_chunk(m.group("who")))
    return out


def _people_from_chunk(chunk: str) -> list[tuple[str, str, str]]:
    """Turn a text chunk into validated (name, role, firm) candidates."""
    out: list[tuple[str, str, str]] = []
    for piece in split_people(chunk):
        # A comma normally introduces a ROLE or FIRM ("Anastasia Titarchuk,
        # CIO"), which is why split_people leaves it alone. But when EVERY
        # comma-separated part reads as a person, it is a guest list, not a
        # qualifier -- "Andy Constan, Ben Hunt, Brent Kochuba". Without this the
        # co-guests were swallowed into the first guest's job title.
        parts = [x.strip() for x in piece.split(",") if x.strip()]
        if len(parts) > 1 and all(looks_like_person(x) for x in parts):
            for person in parts:
                out.append((canonical_name(person), "", ""))
            continue
        name, firm = _split_of_firm(piece)
        role = ""
        # "Anastasia Titarchuk, CIO" / "Robert Keith, Beartooth Group"
        if "," in name:
            head, _, tail = name.partition(",")
            tail = tail.strip()
            if looks_like_person(head.strip()):
                cleaned = clean_firm(tail)
                if cleaned and not firm:
                    firm = cleaned
                elif tail and not looks_like_person(tail):
                    # A trailing comma list that still contains a person's name
                    # is a co-guest list this parser could not fully split (a
                    # quoted nickname, an unusual spelling). Asserting it as a
                    # job title would be worse than asserting nothing.
                    if not any(looks_like_person(x.strip())
                               for x in tail.split(",") if x.strip()):
                        role = tail
                name = head.strip()
        if looks_like_person(name):
            # `firm` has already been through clean_firm() by whichever branch
            # produced it -- positionally where position justified it. Cleaning
            # it a second time without that context threw away every one-word
            # firm ("Principalium", "Transtrend") that had just been accepted.
            out.append((canonical_name(name), role.strip(" .,")[:80], firm.strip()))
    return out


STRATEGIES = {
    "name_role_firm": (_p_name_role_firm, "high"),
    "ft_suffix": (_p_ft_suffix, "high"),
    "dash_prefix": (_p_dash_prefix, "high"),
    "colon_prefix": (_p_colon_prefix, "high"),
    "paren_with": (_p_paren_with, "high"),
    "with_suffix": (_p_with_suffix, "medium"),
    "pipe_segments": (_p_pipe_segments, "medium"),
    "on_infix": (_p_on_infix, "medium"),
}

# Generic fallback, tried after a show's own declared patterns.
#
# Deliberately EXCLUDES colon_prefix, dash_prefix and on_infix. Those are
# positional ("whatever sits before the colon") rather than keyed on an explicit
# marker, so they are reliable only for shows that genuinely use that shape and
# pure noise elsewhere: running colon_prefix generically turned Odd Lots'
# "Listen Now: The Big Take" into a person called "Listen Now", and The
# Derivative's "Family Offices: an inside look..." into "Family Offices".
# A show that uses those shapes declares them; nobody else pays for them.
_GENERIC_ORDER = ("name_role_firm", "ft_suffix", "paren_with", "with_suffix",
                  "pipe_segments")


# --------------------------------------------------------------------------
# Description mining — affiliation stated in the show notes
# --------------------------------------------------------------------------
_ROLE_WORDS = (
    r"(?:Founder|Co-Founder|CEO|CIO|CTO|COO|CFO|Chief [A-Z][\w ]{2,30}|President|"
    r"Partner|Managing Partner|Managing Director|Director|Head of [A-Z][\w ]{2,30}|"
    r"Portfolio Manager|Professor|Chief Economist|Chairman|Strategist|Analyst)"
)
_AFFIL_RES = (
    # "Jon Havice, Founder and CIO of DGV Solutions"
    re.compile(rf"{{name}},?\s+(?:the\s+)?(?P<role>{_ROLE_WORDS}[\w \-]*?)\s+(?:of|at)\s+"
               r"(?P<firm>[A-Z][\w&.,' \-]{2,50}?)(?=[.,;)]|\s+(?:and|to|for|on|who|which)\b|$)"),
    # "Jon Havice, Founder and CIO, DGV Solutions"
    re.compile(rf"{{name}},\s+(?P<role>{_ROLE_WORDS}[\w \-]*?),\s+"
               r"(?P<firm>[A-Z][\w&.,' \-]{2,50}?)(?=[.,;)]|$)"),
    # "DGV Solutions founder Jon Havice"
    re.compile(r"(?P<firm>[A-Z][\w&.,' \-]{2,50}?)\s+"
               rf"(?P<role>{_ROLE_WORDS})\s+{{name}}"),
)


def mine_affiliation(name: str, text: str) -> tuple[str, str]:
    """Find (role, firm) for ``name`` stated in an episode description.

    Only accepts a firm that reads as an organisation, so a sentence fragment
    cannot become an employer.
    """
    if not name or not text:
        return "", ""
    escaped = re.escape(name)
    for template in _AFFIL_RES:
        pattern = re.compile(template.pattern.replace("{name}", escaped), re.I)
        m = pattern.search(text)
        if not m:
            continue
        role = (m.groupdict().get("role") or "").strip(" ,.")
        firm = (m.groupdict().get("firm") or "").strip(" ,.")
        # NOT positional. A regex sweeping free prose gives no guarantee about
        # what sits in the capture group -- an early version relaxed this and
        # recorded "David Harding" as Michael Adam's employer, because both
        # names appeared in the same round-table description. Description
        # mining must keep the full "is this actually a person?" veto.
        firm = clean_firm(firm)
        if firm:
            return role[:80], firm
        if role:
            return role[:80], ""
    return "", ""


# --------------------------------------------------------------------------
# Episode -> guests
# --------------------------------------------------------------------------
def extract_from_episode(episode: dict, pod) -> list[dict]:
    """Extract guest records from one episode.

    Returns [{name, role, firm, confidence, strategy, episode_id, podcast,
    title, published, url}]. Hosts are excluded: they are the show's own voice,
    and they get a `hosts` edge in the graph instead of a guest appearance.
    """
    title = episode.get("title", "")
    if not title:
        return []
    hosts_lower = {h.lower() for h in getattr(pod, "hosts", ())}
    declared = tuple(getattr(pod, "patterns", ()) or ())
    order = [s for s in declared if s in STRATEGIES]
    order += [s for s in _GENERIC_ORDER if s not in order]

    found: dict[str, dict] = {}
    for strategy in order:
        fn, base_conf = STRATEGIES[strategy]
        try:
            candidates = fn(title)
        except Exception:
            continue
        for name, role, firm in candidates:
            key = name.lower()
            if key in hosts_lower or key in found:
                continue
            # A show's OWN declared patterns are its real shapes; the same
            # pattern firing on a show that does not use it is weaker evidence.
            conf = base_conf if strategy in declared else _demote(base_conf)
            found[key] = {
                "name": name, "role": role, "firm": firm,
                "confidence": conf, "strategy": strategy,
                "episode_id": episode.get("id", ""),
                "podcast": episode.get("podcast", ""),
                "episode_title": title,
                "published": episode.get("published", ""),
                "url": episode.get("url", ""),
            }

    # Fill missing role/firm from the description, and promote confidence when
    # the notes independently corroborate the name.
    summary = episode.get("summary", "")
    for rec in found.values():
        # Does the show's own description independently mention this name? That
        # is the single most useful corroboration available: a title parse is a
        # guess about structure, whereas the notes naming the same person is a
        # second, independent statement by the publisher.
        #
        # The match is CASE-SENSITIVE on purpose. A person's name is capitalised
        # wherever it appears, but a phrase that merely looks name-shaped in a
        # Title Case heading appears in lowercase in prose -- so a
        # case-insensitive match "corroborated" episode titles like "Black
        # Swans, Gray Rhinos..." straight into the directory as people.
        rec["corroborated"] = bool(summary) and rec["name"] in summary
        if summary and (not rec["firm"] or not rec["role"]):
            role, firm = mine_affiliation(rec["name"], summary)
            rec["role"] = rec["role"] or role
            rec["firm"] = rec["firm"] or firm
        if rec["corroborated"] and rec["confidence"] == "medium":
            rec["confidence"] = "high"
    return list(found.values())


def _demote(conf: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}[conf]


_VOCAB_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]+")
_VOCAB_MIN_COUNT = 8
_VOCAB_LOWER_RATIO = 0.85


def common_vocabulary(episodes: list[dict]) -> set[str]:
    """Words the corpus itself uses as ordinary lowercase vocabulary.

    A self-calibrating alternative to hand-maintaining a blocklist of every
    non-name phrase a title can produce. Episode TITLES are Title Case, so they
    carry almost no signal about which words are common nouns -- but the
    DESCRIPTIONS are prose, and a word that appears there overwhelmingly in
    lowercase ("dynamics", "premia", "centers") is ordinary vocabulary, never a
    surname. A candidate whose every token is such a word is a phrase.

    Calibrated against the real 6,515-episode corpus: at these thresholds the
    filter removed 14 candidates -- "Team Dynamics", "Mortgage Rate", "Data
    Centers", "Structured Products", "Term Premia", "Political Conflict",
    "Insurance Companies" and similar -- and zero genuine names.
    """
    lower: dict[str, int] = {}
    total: dict[str, int] = {}
    for ep in episodes:
        for tok in _VOCAB_TOKEN_RE.findall(ep.get("summary", "") or ""):
            low = tok.lower()
            total[low] = total.get(low, 0) + 1
            if tok[:1].islower():
                lower[low] = lower.get(low, 0) + 1
    return {w for w, c in total.items()
            if c >= _VOCAB_MIN_COUNT and lower.get(w, 0) / c > _VOCAB_LOWER_RATIO}


def is_common_phrase(name: str, vocabulary: set[str]) -> bool:
    """True if every token of ``name`` is ordinary lowercase vocabulary."""
    if not vocabulary:
        return False
    toks = [t.lower().strip(".,'’") for t in str(name or "").split()]
    return bool(toks) and all(t in vocabulary for t in toks)


def extract_all(episodes: list[dict], registry_by_name: dict,
                vocabulary: set[str] | None = None) -> list[dict]:
    """Run extraction across every harvested episode.

    The corpus vocabulary is derived from the episodes themselves unless one is
    supplied, so a caller working on a subset can pass the full-corpus set.
    """
    vocab = common_vocabulary(episodes) if vocabulary is None else vocabulary
    out: list[dict] = []
    for ep in episodes:
        pod = registry_by_name.get(ep.get("podcast"))
        if pod is None:
            continue
        for rec in extract_from_episode(ep, pod):
            if is_common_phrase(rec["name"], vocab):
                continue
            out.append(rec)
    return out


def meets_threshold(conf: str, minimum: str = INDEX_GUEST_MIN_CONFIDENCE) -> bool:
    return CONFIDENCE_ORDER.get(conf, 0) >= CONFIDENCE_ORDER.get(minimum, 1)


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Collapse per-episode records into one profile per guest.

    Keeps EVERY appearance (that is the point of a guest database), takes the
    best confidence seen, and treats the most recent stated firm as current
    while retaining earlier ones as history — the same "preserve the past,
    identify the present" rule model.merge() applies to entities.
    """
    people: dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: r.get("published", "")):
        key = rec["name"].lower()
        prof = people.setdefault(key, {
            "name": rec["name"], "appearances": [], "podcasts": [],
            "firms": [], "roles": [], "confidence": "low", "strategies": [],
            "corroborated": False,
        })
        prof["corroborated"] = prof["corroborated"] or bool(rec.get("corroborated"))
        prof["appearances"].append({
            "podcast": rec["podcast"], "episode_id": rec["episode_id"],
            "title": rec["episode_title"], "published": rec["published"],
            "url": rec["url"],
        })
        for field, value in (("podcasts", rec["podcast"]), ("firms", rec["firm"]),
                             ("roles", rec["role"]), ("strategies", rec["strategy"])):
            if value and value not in prof[field]:
                prof[field].append(value)
        if CONFIDENCE_ORDER[rec["confidence"]] > CONFIDENCE_ORDER[prof["confidence"]]:
            prof["confidence"] = rec["confidence"]

    for prof in people.values():
        prof["appearances"].sort(key=lambda a: a.get("published", ""), reverse=True)
        prof["n_appearances"] = len(prof["appearances"])
        prof["n_podcasts"] = len(prof["podcasts"])
        # Appearing repeatedly, and especially across several different shows,
        # is independent corroboration that the parse is a real person.
        if prof["n_podcasts"] >= 2 or prof["n_appearances"] >= 3:
            prof["confidence"] = "high"
        # ...and the converse. A name seen exactly ONCE, in one title, that the
        # show's own notes never mention, rests entirely on a single structural
        # guess. Those are demoted so they are recorded against the episode but
        # never promoted into a permanent directory entry -- this is where most
        # surviving mis-parses live.
        elif prof["n_appearances"] == 1 and not prof["corroborated"]:
            prof["confidence"] = "low"
        prof["current_firm"] = prof["firms"][-1] if prof["firms"] else ""
        prof["past_firms"] = prof["firms"][:-1] if len(prof["firms"]) > 1 else []
        prof["role"] = prof["roles"][-1] if prof["roles"] else ""
    return people
