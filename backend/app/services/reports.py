"""PDF and CSV reports generated from saved evaluation data only."""
import csv
import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..models import Submission
from .workflow import submission_totals

CSV_COLUMNS = [
    "exam", "student", "question", "max_marks", "ai_marks", "human_marks", "final_marks",
    "marks_type", "evaluation_status", "approved", "judge", "flags", "reviewer_comment",
]


def _fmt(x):
    return "" if x is None else f"{x:g}"


def _marks_type(ev):
    if ev is None:
        return "none"
    if ev.approved:
        return "human-approved"
    if ev.human_total is not None:
        return "human-edited (not approved)"
    if ev.ai_total is not None:
        return "AI-generated (not approved)"
    return "none"


def submission_rows(sub: Submission) -> list[dict]:
    answers = {a.question_id: a for a in sub.answers}
    rows = []
    for q in sub.exam.questions:
        ans = answers.get(q.id)
        ev = ans.evaluation if ans else None
        rows.append({
            "exam": sub.exam.name, "student": sub.student_identifier, "question": q.label,
            "max_marks": _fmt(q.max_marks), "ai_marks": _fmt(ev.ai_total if ev else None),
            "human_marks": _fmt(ev.human_total if ev else None),
            "final_marks": _fmt(ev.final_total if ev and ev.approved else None),
            "marks_type": _marks_type(ev), "evaluation_status": ev.status if ev else "not_evaluated",
            "approved": "yes" if ev and ev.approved else "no", "judge": ev.judge_name if ev else "",
            "flags": " | ".join(ev.ai_flags or []) if ev else "",
            "reviewer_comment": (ev.reviewer_comment or "") if ev else "",
        })
    return rows


def _csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    return buf.getvalue()


def submission_csv(sub: Submission) -> str:
    return _csv(submission_rows(sub))


def exam_csv(exam) -> str:
    rows = []
    for sub in exam.submissions:
        rows.extend(submission_rows(sub))
    return _csv(rows)


def submission_pdf(sub: Submission) -> bytes:
    totals = submission_totals(sub)
    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small", fontSize=8.5, leading=11)
    p = lambda t, s=None: Paragraph(escape(str(t)), s or styles["BodyText"])
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"Evaluation report - {sub.exam.name}")
    story = [p(f"Evaluation Report - {sub.exam.name}", styles["Title"])]
    meta = [["Student", sub.student_identifier or "-"], ["Subject", sub.exam.subject or "-"],
            ["Total", f"{totals['total']:g} / {totals['max_total']:g}"
                      + (f"  ({totals['percentage']}%)" if totals["percentage"] is not None else "")],
            ["Status", "All marks human-approved" if totals["all_final"]
             else "INCOMPLETE: contains AI-generated or unreviewed marks"]]
    t = Table(meta, colWidths=[35 * mm, 140 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    story += [t, Spacer(1, 6 * mm)]
    story.append(p("AI-generated marks are advisory. Only marks labelled 'human-approved' are final. "
                   "This report does not replace examiner judgement.", small))
    story.append(Spacer(1, 4 * mm))

    answers = {a.question_id: a for a in sub.answers}
    for q in sub.exam.questions:
        ans = answers.get(q.id)
        ev = ans.evaluation if ans else None
        story.append(p(f"{q.label}  -  {_fmt(q.max_marks)} marks", styles["Heading3"]))
        story.append(p(q.text, small))
        if ev is None:
            story.append(p("Not evaluated.", small))
            continue
        story.append(p(f"AI marks: {_fmt(ev.ai_total) or 'none'}   |   Human marks: {_fmt(ev.human_total) or 'none'}   |   "
                       f"Final: {_fmt(ev.final_total) if ev.approved else 'not approved'}   |   Status: {ev.status}", small))
        res = ev.ai_result
        if res:
            rows = [["Criterion", "Awarded", "Max", "Concept coverage"]]
            for c in res["criteria"]:
                cov = "; ".join(f"{x['concept_text'] or x['concept_id']}: {x['status']}" for x in c["concepts"])
                rows.append([Paragraph(escape(c["name"]), small), _fmt(c["awarded"]), _fmt(c["max_marks"]),
                             Paragraph(escape(cov), small)])
            ct = Table(rows, colWidths=[40 * mm, 18 * mm, 14 * mm, 102 * mm], repeatRows=1)
            ct.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(ct)
            for title, items in (("Marks awarded because", res["awarded_because"]),
                                 ("Marks withheld because", res["withheld_because"]),
                                 ("Possibly incorrect statements", res["incorrect_statements"]),
                                 ("How to improve", res["improvements"])):
                if items:
                    story.append(p(title + ":", small))
                    for it in items:
                        story.append(p("- " + it, small))
        elif ev.ai_error:
            story.append(p(f"No AI marks: {ev.ai_error}", small))
        if ev.human_criterion_scores:
            story.append(p("Human-edited criterion marks: " + ", ".join(
                f"#{k}: {v:g}" for k, v in ev.human_criterion_scores.items()), small))
        if ev.reviewer_comment:
            story.append(p(f"Reviewer comment: {ev.reviewer_comment}", small))
        story.append(Spacer(1, 3 * mm))
    doc.build(story)
    return buf.getvalue()
