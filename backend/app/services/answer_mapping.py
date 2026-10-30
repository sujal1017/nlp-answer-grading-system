"""Maps OCR'd answer-sheet text onto exam questions.

Only markers that correspond to a real question label are accepted. Anything that
cannot be mapped confidently is surfaced for human confirmation - text is never
silently assigned to a guessed question.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

# optional prefix, number, optional separator, optional sub letter, closing punctuation
MARKER_RE = re.compile(
    r"^\s*(?P<prefix>(?:Q(?:uestion)?\.?\s*(?:No\.?)?|Ans(?:wer)?\.?)\s*)?"
    r"(?P<num>\d{1,2})\s*[\]\)\.\-:]?\s*\(?(?:(?P<sub>[a-hA-H])(?=[\]\)\.:]))?\)?\s*(?P<close>[\]\)\.:\-])?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
SUB_ONLY_RE = re.compile(r"^\s*\(?(?P<sub>[a-hA-H])\s*[\)\.\]:]\s*(?P<rest>.*)$")


@dataclass
class Segment:
    label: str
    lines: list = field(default_factory=list)
    pages: list = field(default_factory=list)
    occurrences: int = 1
    contiguous: bool = True

    @property
    def text(self):
        return "\n".join(self.lines).strip()


@dataclass
class MappingResult:
    segments: dict          # label -> Segment
    unmapped_text: str
    notes: list
    markers_found: int


def _label(num: str, sub: Optional[str]) -> str:
    return f"Q{int(num)}{(sub or '').upper()}"


def map_answers(pages: list[tuple[int, str]], question_labels: list[str]) -> MappingResult:
    labels = {l.upper() for l in question_labels}
    order = {l.upper(): i for i, l in enumerate(question_labels)}
    mains_with_subs = {re.match(r"Q\d+", l).group(0) for l in labels if re.fullmatch(r"Q\d+[A-H]", l)}
    segments: dict[str, Segment] = {}
    unmapped: list[str] = []
    notes: list[str] = []
    current: Optional[Segment] = None
    current_main: Optional[str] = None
    last_label: Optional[str] = None
    markers = 0

    def start(label: str, rest: str, page: int):
        nonlocal current, last_label
        if label in segments:
            seg = segments[label]
            seg.occurrences += 1
            if last_label != label:
                seg.contiguous = False
        else:
            seg = segments[label] = Segment(label=label)
        if rest:
            seg.lines.append(rest)
        if page not in seg.pages:
            seg.pages.append(page)
        current = seg
        last_label = label

    for page_no, text in pages:
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip():
                if current is not None:
                    current.lines.append("")
                continue
            m = MARKER_RE.match(line)
            handled = False
            if m:
                num, sub, prefix, rest = m.group("num"), m.group("sub"), m.group("prefix"), m.group("rest")
                has_punct = bool(re.match(r"^\s*(?:Q[^\d]*|Ans[^\d]*)?\d{1,2}\s*[\]\)\.\-:(]", line, re.IGNORECASE)) or bool(prefix)
                main = f"Q{int(num)}"
                lab = _label(num, sub)
                # A bare number ("2." / "3)") with no Q/Ans prefix and no sub-letter is usually a list item
                # inside an answer. It only counts as a question marker if it is a real leaf question that
                # comes LATER in the paper than the one being answered; it can never be a section heading.
                weak = not prefix and not sub
                if weak and not (lab in labels and main not in mains_with_subs
                                 and (current is None or order[lab] > order.get(last_label or "", -1))):
                    pass
                elif has_punct or sub:
                    if lab in labels and (sub or main not in mains_with_subs):
                        markers += 1
                        current_main = main
                        start(lab, rest.strip(), page_no)
                        handled = True
                    elif not sub and main in mains_with_subs:
                        # heading like "Q1" followed by "a) ..." on later lines
                        markers += 1
                        current_main = main
                        current = None
                        handled = True
                        if rest.strip():
                            sm = SUB_ONLY_RE.match(rest)
                            if sm and _label(num, sm.group("sub")) in labels:
                                start(_label(num, sm.group("sub")), sm.group("rest").strip(), page_no)
                    elif prefix:
                        notes.append(f"Page {page_no}: marker '{line.strip()[:30]}' does not match any question; kept as unmapped text.")
            if not handled and current_main is not None:
                sm = SUB_ONLY_RE.match(line)
                if sm and f"{current_main}{sm.group('sub').upper()}" in labels:
                    markers += 1
                    start(f"{current_main}{sm.group('sub').upper()}", sm.group("rest").strip(), page_no)
                    handled = True
            if handled:
                continue
            if current is not None:
                current.lines.append(line.strip())
                if page_no not in current.pages:
                    current.pages.append(page_no)
            else:
                unmapped.append(line.strip())

    for seg in segments.values():
        if seg.occurrences > 1:
            where = "non-contiguously " if not seg.contiguous else ""
            notes.append(f"{seg.label} was found {seg.occurrences} times ({where}pages {seg.pages}); confirm the mapping.")
    if markers == 0:
        notes.append("No question numbers were recognised; assign the text to questions manually.")
    return MappingResult(
        segments=segments,
        unmapped_text="\n".join(unmapped).strip(),
        notes=notes,
        markers_found=markers,
    )
