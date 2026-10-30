"""Question-paper parsing: question numbers, sub-questions, instructions and marks.

Generalises the original SDMCET-specific parser. Nothing is invented: if marks are
absent or ambiguous the question is flagged for manual confirmation instead.
"""
import re
from typing import Optional

MAIN_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Que\.?\s*)(\d{1,2})\s*[\.\):\-\]]?\s*(.*)$", re.IGNORECASE
)
PLAIN_NUM_RE = re.compile(r"^\s*(\d{1,2})\s*[\.\)]\s+(\S.*)$")
SUB_RE = re.compile(r"^\s*\(?([a-hA-H])\s*[\)\.\]]\s*(\S.*)?$")
# [5M] [5M, CO:1] (5 marks) 5 Marks (5M) [5]
MARKS_RE = re.compile(
    r"[\[\(]\s*(\d+(?:\.\d+)?)\s*(?:M|Marks?|Mks)?\s*(?:[,;|]\s*(?:CO|BL|L)\s*[:\-]?\s*\d+\s*)*[\]\)]"
    r"|(?<![\w.])(\d+(?:\.\d+)?)\s*(?:Marks?|Mks)\b",
    re.IGNORECASE,
)
HEADER_SKIP = re.compile(
    r"\b(USN|Course Code|Max\.?\s*Marks|Time\s*:|Duration|Semester|Date\s*:|Roll No|Name\s*:)\b", re.IGNORECASE
)
INSTRUCTION_RE = re.compile(r"^\s*(Note|Notes|Instructions?|Answer any|Attempt any|Answer all|Q\.?\s*No\.?\s*\d+\s*is)", re.IGNORECASE)

QTYPE_RULES = [
    ("numerical", r"\b(calculate|compute|find the (?:value|number|time|total)|solve|numerical|evaluate the|derive)\b"),
    ("algorithm", r"\b(algorithm|pseudo-?code|write a program|flowchart|steps? (?:of|in) the)\b"),
    ("comparison", r"\b(compare|differentiate|distinguish|contrast|difference between|versus|\bvs\.?\b)\b"),
    ("definition", r"^(?:\W*)(?:define|what is|what are|state the meaning|name the)\b"),
    ("application", r"\b(apply|application of|illustrate with|with an example|case study|use of|how would you)\b"),
    ("explanation", r"\b(explain|describe|discuss|elaborate|justify|why|how)\b"),
]


def detect_qtype(text: str, max_marks: Optional[float]) -> str:
    t = text.lower()
    for qtype, pattern in QTYPE_RULES:
        if re.search(pattern, t):
            if qtype == "definition" and re.search(r"\b(explain|discuss|describe|elaborate)\b", t):
                qtype = "explanation"  # compound "What is X? Discuss Y" asks for more than a definition
            if qtype in ("explanation", "definition") and max_marks and max_marks >= 8:
                return "long_answer"
            return qtype
    if max_marks and max_marks >= 8:
        return "long_answer"
    return "unknown"


def _find_marks(text: str):
    found = []
    for m in MARKS_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        found.append((float(raw), m.span()))
    return found


def _finalise(leaf: dict) -> dict:
    text = re.sub(r"\s+", " ", leaf["text"]).strip()
    reasons = list(leaf.get("reasons", []))
    marks_found = _find_marks(text)
    distinct = sorted({v for v, _ in marks_found})
    max_marks = None
    if len(distinct) == 1:
        max_marks = distinct[0]
    elif len(distinct) > 1:
        reasons.append(f"Multiple different marks values found in the question text: {distinct}")
    elif leaf.get("parent_marks") is not None:
        reasons.append(
            f"Marks not stated for this part; the parent question mentions {leaf['parent_marks']:g}. "
            "Confirm how they are divided."
        )
    else:
        reasons.append("Marks could not be found; please enter the maximum marks.")
    clean = text
    for _, (s, e) in reversed(marks_found):
        clean = clean[:s] + clean[e:]
    clean = re.sub(r"\s{2,}", " ", clean).strip(" -:")
    if not clean:
        reasons.append("Question text is empty after extraction.")
    return {
        "label": leaf["label"],
        "text": clean,
        "context": leaf.get("context", ""),
        "max_marks": max_marks,
        "qtype": detect_qtype(clean, max_marks),
        "source_page": leaf.get("page"),
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }


def parse_question_paper(pages: list[tuple[int, str]]) -> dict:
    """pages: [(page_number, text)]. Returns {"questions": [...], "instructions": str}."""
    leaves: list[dict] = []
    instructions: list[str] = []
    cur_main: Optional[dict] = None   # {"num", "text", "page", "marks"}
    cur_leaf: Optional[dict] = None
    main_has_subs = False

    def close_leaf():
        nonlocal cur_leaf
        if cur_leaf is not None:
            leaves.append(cur_leaf)
        cur_leaf = None

    for page_no, text in pages:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if HEADER_SKIP.search(line) and cur_main is None:
                continue
            m_main = MAIN_RE.match(line) or PLAIN_NUM_RE.match(line)
            # A line like "a) ..." must be treated as a sub-question, not "A"-prefixed text.
            m_sub = SUB_RE.match(line) if cur_main is not None else None
            if m_main and not (m_sub and not MAIN_RE.match(line)):
                num, rest = m_main.group(1), (m_main.group(2) or "").strip()
                # "Q. No. 3 is COMPULSORY" style instruction lines
                if INSTRUCTION_RE.match(line) and not rest:
                    instructions.append(line)
                    continue
                close_leaf()
                parent_marks = None
                mk = _find_marks(rest)
                if len({v for v, _ in mk}) == 1:
                    parent_marks = mk[0][0]
                cur_main = {"num": num, "text": rest, "page": page_no, "marks": parent_marks}
                main_has_subs = False
                # Tentatively a leaf itself (until a sub-question appears)
                cur_leaf = {"label": f"Q{num}", "text": rest, "page": page_no, "context": "",
                            "parent_marks": None, "is_main": True}
            elif m_sub and cur_main is not None:
                sub_letter = m_sub.group(1).upper()
                sub_text = (m_sub.group(2) or "").strip()
                if cur_leaf is not None and cur_leaf.get("is_main"):
                    # The main question has sub-parts: its own text becomes context only.
                    cur_main["text"] = cur_leaf["text"]
                    cur_leaf = None
                else:
                    close_leaf()
                main_has_subs = True
                cur_leaf = {
                    "label": f"Q{cur_main['num']}{sub_letter}", "text": sub_text, "page": page_no,
                    "context": cur_main["text"], "parent_marks": cur_main["marks"],
                }
            elif cur_leaf is not None:
                cur_leaf["text"] += " " + line
            elif cur_main is None:
                if INSTRUCTION_RE.match(line):
                    instructions.append(line)
                # other pre-question header lines are ignored
            else:
                # text after a main question that had subs but before any new sub: context
                cur_main["text"] += " " + line
    close_leaf()

    questions = [_finalise(l) for l in leaves if l["text"].strip() or l["label"]]
    # Drop empty main-question leaves that carried no text and had no parts
    questions = [q for q in questions if q["text"]]
    seen = {}
    for q in questions:  # duplicate labels would violate the unique constraint; flag instead of hiding
        if q["label"] in seen:
            n = seen[q["label"]] = seen[q["label"]] + 1
            q["review_reasons"].append(f"Duplicate label '{q['label']}' detected; renamed.")
            q["label"] = f"{q['label']}_{n}"
            q["needs_review"] = True
        else:
            seen[q["label"]] = 1
    return {"questions": questions, "instructions": " ".join(instructions)}
