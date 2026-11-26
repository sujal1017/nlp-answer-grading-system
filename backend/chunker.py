import os
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import pymupdf as fitz
from rapidfuzz import fuzz

STOPWORDS = {
    "about", "after", "also", "and", "any", "are", "been", "briefly", "define",
    "describe", "discuss", "each", "explain", "following", "for", "from", "give",
    "have", "how", "into", "its", "list", "of", "or", "state", "the", "their",
    "them", "then", "this", "that", "with", "write", "what", "when", "which",
    "your", "using", "used", "than", "them", "they", "will", "were",
}

# Explicitly ignore Roman numerals from triggering question boundaries
ROMAN_NUMERAL_RE = re.compile(
    r"^(?:#+\s*)?[([]?\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*[)\]:.-]",
    re.IGNORECASE,
)

# Explicitly ignore indented internal lists (e.g. 2+ spaces or tab followed by 1., a), i))
INDENTED_LIST_RE = re.compile(
    r"^(?:[ \t]{2,}|\t)[([]?\s*(?:\d+|[a-zA-Z]|i{1,3}|iv|v|vi{1,3})\s*[)\]:.-]",
    re.IGNORECASE,
)


def convert_pdf_to_images(pdf_bytes: bytes, output_dir: str) -> List[str]:
    """Converts an uploaded PDF file into individual PNG images per page."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_path = str(Path(output_dir) / f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def clean_page_markers(markdown_text: str) -> str:
    cleaned = re.sub(r"###\s*Page\s*\d+\s*", "\n", markdown_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"---+\s*Page\s*\d+\s*---+", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n---+\n", "\n", cleaned)
    return cleaned.strip()


def parse_sdmcet_question_paper(qp_markdown: str) -> List[Dict]:
    """Parses teacher-pasted SDMCET question text into Q1A/Q1B/... records."""
    text = clean_page_markers(qp_markdown)
    marks_pattern = re.compile(r"[\[(]?\s*(\d{1,2})\s*(?:Marks?|M)\b", re.IGNORECASE)
    co_pattern = re.compile(r"CO\s*[:\-]?\s*(\d+)", re.IGNORECASE)
    inline_line = re.compile(
        r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\s*[(\[]\s*([a-dA-D])\s*[)\]]\s*(.+)$",
        re.IGNORECASE,
    )
    inline_line_b = re.compile(
        r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\s+([a-dA-D])\s*[).:\-]\s*(.+)$",
        re.IGNORECASE,
    )

    extracted_questions: List[Dict] = []
    current = None

    def flush():
        nonlocal current
        if not current:
            return
        body = re.sub(r"\s+", " ", current["question_text"]).strip(" .-[]()")
        body = re.sub(r"\bOR\s*$", "", body, flags=re.IGNORECASE).strip()
        m_match = marks_pattern.search(body)
        if m_match:
            current["max_marks"] = int(m_match.group(1))
            body = body[: m_match.start()].strip()
        co_match = co_pattern.search(current["question_text"])
        if co_match:
            current["co"] = f"CO:{co_match.group(1)}"
        current["question_text"] = body
        extracted_questions.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("OR"):
            continue
        if any(skip in line.upper() for skip in ["USN", "COURSE CODE", "INTERNAL ASSESSMENT", "NOTE:", "*****"]):
            continue

        m = inline_line.match(line) or inline_line_b.match(line)
        if m:
            flush()
            current = {
                "question_id": f"Q{m.group(1)}{m.group(2).upper()}",
                "question_text": m.group(3).strip(),
                "max_marks": 5,
                "co": "CO:1",
            }
            continue

        main_match = re.match(r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\b", line, re.I)
        if main_match and not re.search(r"[a-dA-D]", line[main_match.end() : main_match.end() + 6] or ""):
            flush()
            current = {
                "question_id": f"Q{main_match.group(1)}A",
                "question_text": "",
                "max_marks": 5,
                "co": "CO:1",
                "_main": main_match.group(1),
                "_open": True,
            }
            rest = line[main_match.end() :].strip(" .:-")
            if rest:
                current["question_text"] = rest
            continue

        sub_match = re.match(r"^([a-dA-D])\s*[).\]\-]\s*(.+)$", line, re.I)
        if sub_match:
            main_n = "1"
            if current:
                prev_id = current["question_id"]
                main_n = re.sub(r"^Q(\d).*", r"\1", prev_id)
                flush()
            current = {
                "question_id": f"Q{main_n}{sub_match.group(1).upper()}",
                "question_text": sub_match.group(2).strip(),
                "max_marks": 5,
                "co": "CO:1",
            }
            continue

        if current:
            current["question_text"] = (current["question_text"] + " " + line).strip()

    flush()
    return extracted_questions


def parse_teacher_questions(raw_text: str) -> List[Dict]:
    """Accepts pasted exam text or compact `Q1A | prompt | 5` rows."""
    compact = re.compile(
        r"^\s*(Q?\s*[1-3]\s*[A-Da-d])\s*[|:.\-]\s*(.+?)\s*[|:]\s*(\d+(?:\.\d+)?)\s*$"
    )
    rows = []
    for line in raw_text.splitlines():
        m = compact.match(line.strip())
        if not m:
            continue
        qid = re.sub(r"\s+", "", m.group(1).upper())
        if not qid.startswith("Q"):
            qid = "Q" + qid
        rows.append(
            {
                "question_id": qid,
                "question_text": m.group(2).strip(),
                "max_marks": int(float(m.group(3))),
                "co": "CO:1",
            }
        )
    if rows:
        return rows
    return parse_sdmcet_question_paper(raw_text)


def keywords(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOPWORDS}


def _match_orphan_subquestion(
    answer_text: str,
    sub_letter: str,
    rubrics: List[Dict],
    existing_keys: List[str],
) -> str:
    """
    Semantic Topic Routing for orphan sub-questions (e.g. a) or b) written without main number):
    Match a numbered rubric by fuzzy text similarity and prefer unused questions on ties.
    """
    candidates = [r for r in rubrics if str(r.get("question_id", "")).upper().endswith(sub_letter.upper())]
    if not candidates:
        return f"Q1{sub_letter.upper()}"

    probe_text = (answer_text or "").strip()[:250]
    if not probe_text:
        for c in candidates:
            qid = str(c["question_id"]).upper()
            if qid not in existing_keys:
                return qid
        return str(candidates[0]["question_id"]).upper()

    best_qid = str(candidates[0]["question_id"]).upper()
    best_score = -1.0
    for c in candidates:
        q_text = c.get("question_text", "")
        ratio = fuzz.token_set_ratio(probe_text.lower(), q_text.lower())
        qid = str(c["question_id"]).upper()
        bonus = 5.0 if qid not in existing_keys else 0.0
        score = ratio + bonus
        if score > best_score:
            best_score = score
            best_qid = qid
    return best_qid


def _semantic_heading_qid(line: str, rubrics: List[Dict]) -> Optional[str]:
    """Match a standalone student heading against teacher-configured question rubrics using keyword overlap and RapidFuzz."""
    raw_stripped = line.strip()
    # Prose sentences ending with a period or comma are body text, never section headers
    if raw_stripped.endswith(".") or raw_stripped.endswith(","):
        return None

    stripped = re.sub(r"^[#>*\-\s]+", "", line).strip(" .:;-*_")
    word_count = len(stripped.split())
    if word_count < 2 or word_count > 12:
        return None
    if re.match(r"^(?:page|internal assessment|usn|or|marks|note)\b", stripped, re.I):
        return None

    if not rubrics:
        return None

    line_l = stripped.lower()
    best_qid = None
    best_ratio = 0.0

    for r in rubrics:
        qid = str(r.get("question_id", "")).upper()
        if not re.match(r"^Q\d+[A-Z]$", qid):
            continue
        q_text = r.get("question_text", "")
        ratio = fuzz.token_set_ratio(line_l, q_text.lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_qid = qid

    if best_ratio >= 65.0:
        return best_qid
    return None


def merge_two_tier_margins(text: str) -> str:
    """Pre-processes pages to merge adjacent two-tier margin lines: e.g. Q1]\\n a] -> \\nQ1A) , 1]\\n (a) -> \\nQ1A) , 2]\\n (6) -> \\nQ2B) """
    def _repl(m):
        q_num = m.group(1)
        raw_sub = m.group(2).upper()
        sub = "B" if raw_sub == "6" else raw_sub
        return f"\nQ{q_num}{sub}) "

    pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?(\d+)\s*[\]).:-]?\s*\n\s*[([]?\s*([a-zA-Z6])\s*[)\]:.-]?\s*",
        re.IGNORECASE,
    )
    return pattern.sub(_repl, text)


def normalize_margin_markers(text: str) -> str:
    """Normalize handwritten / OCR-garbled margin markers to QnX) form."""
    # 2] (6) / 2](6) / 2 (6) / 2 [6] / Q2] (6) / 2] 6) / 2] 6] / 36) / 36]  -> Q2B) / Q3B)   (handwritten b read as 6)
    text = re.sub(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?([1-3])\s*[\]).:-]?\s*[([]?\s*6\s*[)\]:.-]",
        r"\nQ\1B) ",
        text,
        flags=re.IGNORECASE,
    )
    # Ans 1a: / Answer 1(a) / Ans 1(b) / Ans. 1 a / Ans 1-a
    text = re.sub(
        r"(?:^|\n)\s*Ans(?:wer)?\.?\s*(?:No\.?)?\s*(\d+)\s*[([.\-\s]*([a-zA-Z])\s*[)\]:.\-]?",
        lambda m: f"\nQ{m.group(1)}{m.group(2).upper()}) ",
        text,
        flags=re.IGNORECASE,
    )
    # Q1 (a) / Q1(a) / Question 1(a) / 1 (a) / 1(a) / 1] a / 3] a / 3) a / 1. a / 1 a) / 3b) / Q1] a] / 1] a]
    text = re.sub(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?(\d+)\s*[\]).:-]?\s*[([]?\s*([a-zA-Z])(?:\s*[)\]:.-]|\s+)",
        lambda m: f"\nQ{m.group(1)}{m.group(2).upper()}) ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _best_unlabeled_qid(block: str, rubrics: List[Dict], used: List[str]) -> Optional[str]:
    if not rubrics or not block.strip():
        return None
    words = keywords(block[:500])
    scored: List[Tuple[int, str]] = []
    for r in rubrics:
        qid = str(r["question_id"]).upper()
        overlap = len(words & keywords(r.get("question_text", "")))
        scored.append((overlap, qid))
    scored.sort(key=lambda x: (x[0], 0 if x[1] not in used else 1), reverse=True)
    if scored and scored[0][0] >= 2:
        return scored[0][1]
    unused = [r for r in rubrics if str(r["question_id"]).upper() not in used]
    if unused:
        return str(unused[0]["question_id"]).upper()
    return None


def chunk_student_booklet(
    raw_markdown: str,
    rubrics: Optional[List[Dict]] = None,
    margin_anchors: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Hybrid boundary detection & multi-page continuation:
      1. Merge two-tier margin markers across lines (Q1]\\n a] -> Q1A) ).
      2. Regex-normalize OCR margin quirks (2] (6) -> Q2B, 36) -> Q3B, 3] a -> Q3A).
      3. Track current_qid and transition standalone sub-letters (e.g. b] when current is Q1A -> Q1B).
      4. Treat standalone headings as semantic anchors against active teacher questions using RapidFuzz.
      5. Retain all numbered answer-part keys so unsupported questions can be manually assigned.
      6. Exclude Roman numerals and indented lists from triggering false question splits.
      7. Route orphan sub-questions using fuzzy text matching.
    """
    rubrics = rubrics or []
    if margin_anchors:
        anchor_text = "\n".join(a for a in margin_anchors if a)
        if anchor_text:
            raw_markdown = f"{anchor_text}\n\n{raw_markdown}"

    text = clean_page_markers(raw_markdown)
    text = re.sub(r"^\s*\|?\s*Internal Assessment\s*[-–—]?\s*[I1V]+\s*\n?", "", text, flags=re.IGNORECASE)
    text = merge_two_tier_margins(text)
    text = normalize_margin_markers(text)

    lines = text.splitlines()
    boundaries: List[Tuple[int, str]] = []
    last_main = None
    current_qid = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Exclude Roman numerals & indented sub-lists from triggering boundaries
        if ROMAN_NUMERAL_RE.match(line) or INDENTED_LIST_RE.match(raw_line):
            continue

        # Check if line is a generic continuation marker: "Contd...", "Continued", "(contd)"
        if re.match(r"^(?:#+\s*)?[([]?\s*(?:contd|continued|continuation)\b", line, re.IGNORECASE):
            continue

        # Standalone Sub-Letter Transition: line starts with b), b], (b), (6), 6)
        standalone_sub = re.match(
            r"^(?:#+\s*)?(?:Ans(?:wer)?\.?\s*)?(?:[([]\s*([a-zA-Z6])\s*[)\]]|([a-zA-Z6])\s*[)\]:.-])(?:\s+|$)(.*)$",
            line,
            re.IGNORECASE,
        )
        if standalone_sub:
            raw_let = (standalone_sub.group(1) or standalone_sub.group(2) or "A").upper()
            sub_let = "B" if raw_let == "6" else raw_let
            rest = (standalone_sub.group(3) or "").strip()

            target_qid = None
            if sub_let == "B" and current_qid and re.match(r"^Q\d+A$", current_qid):
                target_qid = current_qid[:-1] + "B"
            elif last_main:
                target_qid = f"Q{last_main}{sub_let}"
            else:
                target_qid = _match_orphan_subquestion(
                    rest or " ".join(lines[idx : idx + 4]),
                    sub_let,
                    rubrics,
                    [b[1] for b in boundaries],
                )

            if target_qid:
                current_qid = target_qid
                main_match = re.match(r"^Q(\d+)", target_qid)
                last_main = main_match.group(1) if main_match else last_main
                boundaries.append((idx, target_qid))
                continue

        # Standalone main question marker (multi-tier parent): e.g. "Question 1", "Q1", "Q1]", "1]", "[1]", "Ans 1:", "Q.1", "1."
        main_only = re.match(
            r"^(?:#+\s*)?(?:(?:Q(?:uestion)?\.?\s*(?:No\.?)?|Ans(?:wer)?\.?)\s*[([]?\s*(\d+)\s*[)\]:.-]?|"
            r"[([]\s*(\d+)\s*[)\]]|(\d+)\s*[)\]:.-])\s*$",
            line,
            re.IGNORECASE,
        )
        if main_only:
            last_main = main_only.group(1) or main_only.group(2) or main_only.group(3)
            continue

        # Question + Sub-question or Sub-question alone (tier 2: e.g. "1a)", "Q1A)", "Q2B)", "(a)", "a)")
        hm = re.match(
            r"^(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?"
            r"(?:"
            r"(\d+)\s*[)\].\-\s]*[([]?\s*([a-zA-Z6])\s*[)\].:\-\s]|"
            r"[([]\s*([a-zA-Z6])\s*[)\]]\s*|"
            r"([a-zA-Z])\s*[)\]:.-]\s*"
            r")\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if hm:
            q_num = hm.group(1)
            raw_sub = hm.group(2) or hm.group(3) or hm.group(4) or "A"
            sub_let = "B" if raw_sub.strip("()[] ") == "6" else raw_sub.strip("()[] ").upper()
            rest = (hm.group(5) or "").strip()
            if q_num:
                last_main = q_num
                qid = f"Q{q_num}{sub_let}"
            elif current_qid and sub_let == "B" and re.match(r"^Q\d+A$", current_qid):
                qid = current_qid[:-1] + "B"
            elif last_main:
                qid = f"Q{last_main}{sub_let}"
            else:
                qid = _match_orphan_subquestion(
                    rest or " ".join(lines[idx : idx + 4]),
                    sub_let,
                    rubrics,
                    [b[1] for b in boundaries],
                )
                maybe_n = re.match(r"^Q(\d+)", qid)
                if maybe_n:
                    last_main = maybe_n.group(1)
            current_qid = qid
            boundaries.append((idx, qid))
            continue

        semantic = _semantic_heading_qid(line, rubrics)
        if semantic:
            maybe_n = re.match(r"^Q(\d+)", semantic)
            if maybe_n:
                last_main = maybe_n.group(1)
            current_qid = semantic
            boundaries.append((idx, semantic))

    if not boundaries:
        raw_chunks = _split_unlabeled_booklet(text, rubrics)
    else:
        raw_chunks = {}
        for i, (line_idx, qid) in enumerate(boundaries):
            start = line_idx
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            block_lines = lines[start:end]
            # Drop the heading line's marker; keep topical heading words.
            if block_lines:
                block_lines = [re.sub(
                    r"^(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?"
                    r"(?:\d+\s*[)\].\-\s]*[(\[]?\s*[a-zA-Z6]\s*[)\].:\-\s]|(?:[a-zA-Z]|[([]?\s*6\s*[)\]])\s*[).:\]\-\s]?)\s*",
                    "",
                    block_lines[0],
                    flags=re.IGNORECASE,
                )] + block_lines[1:]
            content = "\n".join(block_lines).strip()
            if qid in raw_chunks:
                if content:
                    raw_chunks[qid] = f"{raw_chunks[qid]}\n\n{content}".strip()
            else:
                raw_chunks[qid] = content

        # Preamble before the first boundary: attach via keywords, not Q1A by default.
        first_idx = boundaries[0][0]
        preamble = "\n".join(lines[:first_idx]).strip()
        if preamble and len(preamble) > 40:
            pqid = _best_unlabeled_qid(preamble, rubrics, list(raw_chunks.keys()))
            if pqid:
                raw_chunks[pqid] = (preamble + "\n\n" + raw_chunks.get(pqid, "")).strip()

    # Keep unsupported numbered parts for the caller to report and expose for manual mapping.
    valid_chunks: Dict[str, str] = {}
    for k, v in raw_chunks.items():
        k_clean = k.strip().upper()
        if re.match(r"^Q\d+[A-Z]$", k_clean) or any(
            str(r.get("question_id", "")).strip().upper() == k_clean for r in rubrics
        ):
            if k_clean in valid_chunks:
                valid_chunks[k_clean] += "\n\n" + v
            else:
                valid_chunks[k_clean] = v
        else:
            match_main = re.match(r"^Q?(\d+)$", k_clean)
            target_main = match_main.group(1) if match_main else None
            cand_a = f"Q{target_main}A" if target_main else ""
            cand_b = f"Q{target_main}B" if target_main else ""

            text_a = ""
            text_b = ""
            for r in rubrics:
                qid_r = str(r.get("question_id", "")).upper()
                if qid_r == cand_a:
                    text_a = r.get("question_text", "")
                elif qid_r == cand_b:
                    text_b = r.get("question_text", "")

            sim_a = fuzz.token_set_ratio(v.lower(), text_a.lower())
            sim_b = fuzz.token_set_ratio(v.lower(), text_b.lower())

            resolved_key = (cand_a if sim_a >= sim_b else cand_b) if target_main else k_clean
            if resolved_key in valid_chunks:
                valid_chunks[resolved_key] += "\n\n" + v
            else:
                valid_chunks[resolved_key] = v

    return valid_chunks


def _split_unlabeled_booklet(text: str, rubrics: List[Dict]) -> Dict[str, str]:
    if not text.strip():
        return {}
    if not rubrics:
        return {}

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    assigned: Dict[str, List[str]] = {}
    for para in paragraphs:
        qid = _best_unlabeled_qid(para, rubrics, [])
        if not qid:
            continue
        assigned.setdefault(qid, []).append(para)

    chunks = {qid: "\n\n".join(parts) for qid, parts in assigned.items()}
    if chunks:
        return chunks

    qid = _best_unlabeled_qid(text, rubrics, []) or str(rubrics[0]["question_id"]).upper()
    return {qid: text.strip()}


def calculate_ia_total(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """
    SDMCET IA Rule:
      IA Total = max(Q1A + Q1B, Q2A + Q2B) + (Q3A + Q3B)   out of 20 Marks.
    """
    def g(*keys):
        return sum(float(scores_dict.get(k, 0.0) or 0.0) for k in keys)

    q1_total = g("Q1A", "Q1B")
    q2_total = g("Q2A", "Q2B")
    q3_total = g("Q3A", "Q3B")
    best_choice = max(q1_total, q2_total)
    chosen = "Q1" if q1_total >= q2_total else "Q2"
    total = round(best_choice + q3_total, 2)
    return {
        "q1_total": round(q1_total, 2),
        "q2_total": round(q2_total, 2),
        "q3_total": round(q3_total, 2),
        "chosen_pair": chosen,
        "ia_total": total,
        "max_marks": 20,
    }
