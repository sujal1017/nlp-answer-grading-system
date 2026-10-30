"""Small, dependency-free text helpers used by the grading engine."""
import re

STOPWORDS = frozenset(
    """a an the of and or to in on at for by with from as is are was were be been being this that these those
    it its into than then so such can may might will would should could also which who whom what when where
    how why do does did done have has had not no nor but if else their there they them he she his her we our
    you your i me my us about over under between among each any all some more most other another one two""".split()
)
NEGATIONS = frozenset(
    "not no never none cannot can't cant isn't isnt aren't arent doesn't doesnt don't dont neither nor".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")
_SUFFIXES = ("ations", "ation", "ingly", "ments", "ment", "ness", "ities", "ity", "ings", "ing",
             "ies", "ied", "ives", "ive", "ers", "er", "ed", "es", "ly", "s")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def stem(word: str) -> str:
    w = word.lower()
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            # "scanning" -> "scann" -> "scan"; "stopped" -> "stopp" -> "stop"
            if suf in ("ing", "ings", "ed", "er", "ers") and len(w) > 3 and w[-1] == w[-2] and w[-1] not in "sl":
                w = w[:-1]
            break
    return w


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, line breaks and bullets. Keeps non-trivial fragments."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\s)[\-•*]\s+", text or "")
    return [p.strip() for p in parts if p and len(tokens(p)) >= 2]


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in tokens(text))


def evidence_in_text(evidence: str, text: str) -> bool:
    """True if the evidence quote appears in the text (whitespace/case-insensitive)."""
    e, t = normalize(evidence), normalize(text)
    return bool(e) and e in t
