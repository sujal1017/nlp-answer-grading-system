import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plus, Trash2 } from "lucide-react";
import { api, waitForJob, type Concept, type Criterion, type Exam, type Question, type Submission } from "../api";
import { Badge, Button, Card, Empty, Field, Notice, Progress, Spinner, StatusBadge, inputCls, useLoad, fmt } from "../ui";

const QTYPES = ["definition", "explanation", "comparison", "application", "numerical", "algorithm", "long_answer", "unknown"];

export default function ExamSetup() {
  const { id } = useParams();
  const exam = useLoad(() => api.get<Exam>(`/exams/${id}`), [id]);
  const subs = useLoad(() => api.get<Submission[]>(`/submissions?exam_id=${id}`), [id]);
  const [msg, setMsg] = useState<{ kind: "info" | "error" | "success" | "warn"; text: string } | null>(null);
  const [job, setJob] = useState<{ p: number; m: string } | null>(null);
  const [newQ, setNewQ] = useState({ label: "", text: "", max_marks: "" });
  const [student, setStudent] = useState("");
  const nav = useNavigate();

  if (exam.loading) return <Spinner />;
  if (exam.error || !exam.data) return <Notice kind="error">{exam.error}</Notice>;
  const e = exam.data;
  const qs = e.questions ?? [];

  async function uploadPaper(file: File) {
    setMsg(null); setJob({ p: 0, m: "Uploading…" });
    try {
      const r = await api.upload<{ job: { id: number; status: string }; questions?: Question[] }>(`/exams/${id}/question-paper`, file);
      if (r.job.status !== "done") {
        const j = await waitForJob(r.job.id, (x) => setJob({ p: x.progress, m: x.message }));
        if (j.status === "failed") throw new Error(j.error || "Question extraction failed");
      }
      setMsg({ kind: "success", text: "Questions extracted. Review every question and mark value below; flagged ones need confirmation." });
      exam.reload();
    } catch (x: any) { setMsg({ kind: "error", text: x.message }); } finally { setJob(null); }
  }

  async function uploadSheet(file: File) {
    setMsg(null); setJob({ p: 0, m: "Uploading and reading pages…" });
    try {
      const r = await api.upload<{ submission: Submission; job: { id: number; status: string } }>(`/exams/${id}/submissions`, file, { student_identifier: student });
      if (r.job.status !== "done") await waitForJob(r.job.id, (x) => setJob({ p: x.progress, m: x.message }));
      nav(`/submissions/${r.submission.id}`);
    } catch (x: any) { setMsg({ kind: "error", text: x.message }); setJob(null); }
  }

  async function addQuestion() {
    try {
      await api.post(`/exams/${id}/questions`, { ...newQ, max_marks: newQ.max_marks === "" ? null : Number(newQ.max_marks) });
      setNewQ({ label: "", text: "", max_marks: "" }); exam.reload();
    } catch (x: any) { setMsg({ kind: "error", text: x.message }); }
  }

  const ready = qs.length > 0 && qs.every((q) => q.max_marks !== null);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">{e.name}</h1>
        <p className="text-sm text-slate-500">{e.subject} {e.code && `· ${e.code}`}{e.choice_groups.length > 0 && ` · choose-one groups: ${e.choice_groups.map((g) => g.join("/")).join(", ")}`}</p>
      </div>
      {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}
      {job && <Card><Progress value={job.p} label={job.m} /></Card>}

      <Card title="1. Question paper (PDF)">
        <input type="file" accept="application/pdf" className="text-sm" onChange={(ev) => ev.target.files?.[0] && uploadPaper(ev.target.files[0])} />
        <p className="mt-2 text-xs text-slate-500">Questions are extracted automatically. Nothing is invented: questions or marks that cannot be read reliably are flagged for you to confirm.</p>
        {e.documents?.filter((d) => d.kind === "question_paper").map((d) => <p key={d.id} className="mt-1 text-xs text-slate-600">Uploaded: {d.filename} ({d.page_count} pages)</p>)}
      </Card>

      <Card title={`2. Questions, reference answers and rubrics (${qs.length})`}>
        {qs.length === 0 ? <Empty>No questions yet. Upload a paper or add questions manually below.</Empty> : (
          <div className="space-y-3">{qs.map((q) => <QuestionEditor key={q.id} q={q} onChange={exam.reload} />)}</div>)}
        <div className="mt-4 rounded-md bg-slate-50 p-3">
          <p className="mb-2 text-sm font-medium">Add a question manually</p>
          <div className="grid gap-2 md:grid-cols-[100px_1fr_100px_auto]">
            <input className={inputCls} placeholder="Label e.g. Q1A" value={newQ.label} onChange={(x) => setNewQ({ ...newQ, label: x.target.value })} />
            <input className={inputCls} placeholder="Question text" value={newQ.text} onChange={(x) => setNewQ({ ...newQ, text: x.target.value })} />
            <input className={inputCls} placeholder="Marks" type="number" min="0" value={newQ.max_marks} onChange={(x) => setNewQ({ ...newQ, max_marks: x.target.value })} />
            <Button disabled={!newQ.label.trim() || !newQ.text.trim()} onClick={addQuestion}><Plus className="h-4 w-4" />Add</Button>
          </div>
        </div>
      </Card>

      <Card title="3. Upload a student's handwritten answer sheet (PDF)">
        {!ready && <div className="mb-3"><Notice kind="warn">Confirm marks for every question before uploading answer sheets.</Notice></div>}
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Student identifier (optional)"><input className={inputCls} value={student} onChange={(x) => setStudent(x.target.value)} /></Field>
          <input type="file" accept="application/pdf" disabled={qs.length === 0} className="text-sm" onChange={(ev) => ev.target.files?.[0] && uploadSheet(ev.target.files[0])} />
        </div>
      </Card>

      <Card title="Answer sheets for this exam" actions={<a className="text-sm text-indigo-700 hover:underline" href={`/api/exams/${id}/report.csv`}>Export exam CSV</a>}>
        {subs.loading ? <Spinner /> : !subs.data?.length ? <Empty>None uploaded yet.</Empty> : (
          <ul className="divide-y text-sm">{subs.data.map((s) => (
            <li key={s.id} className="flex items-center justify-between py-2">
              <Link className="text-indigo-700 hover:underline" to={`/submissions/${s.id}`}>{s.student_identifier || `Sheet #${s.id}`} — {s.filename}</Link>
              <span className="flex items-center gap-3"><span>{s.max_total ? `${fmt(s.total)}/${fmt(s.max_total)}` : ""}</span><StatusBadge s={s.status} /></span>
            </li>))}</ul>)}
      </Card>
      <Metrics examId={id!} />
    </div>
  );
}

function Metrics({ examId }: { examId: string }) {
  const m = useLoad(() => api.get<any>(`/exams/${examId}/metrics`), [examId]);
  if (m.loading || !m.data) return null;
  const d = m.data;
  return (
    <Card title="Human-vs-AI agreement">
      {!d.sufficient ? <Notice kind="info">{d.message || `Not enough human-reviewed answers yet (${d.n ?? 0}) for reliable metrics.`}</Notice> : (
        <dl className="grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
          {["n", "mae", "rmse", "pearson", "spearman", "exact_agreement", "within_tolerance"].map((k) => d[k] !== undefined && d[k] !== null && (
            <div key={k}><dt className="text-xs uppercase text-slate-500">{k.replace(/_/g, " ")}</dt><dd className="font-medium">{typeof d[k] === "number" ? Number(d[k].toFixed(3)) : String(d[k])}</dd></div>))}
        </dl>)}
    </Card>
  );
}

function QuestionEditor({ q, onChange }: { q: Question; onChange: () => void }) {
  const [open, setOpen] = useState(false);
  const [f, setF] = useState({ label: q.label, text: q.text, max_marks: q.max_marks ?? "", qtype: q.qtype });
  const [ref, setRef] = useState(q.reference_answer);
  const [crit, setCrit] = useState<Criterion[]>(q.criteria.length ? q.criteria : []);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const run = async (fn: () => Promise<unknown>, done: string) => {
    setErr(null); setOk(null);
    try { await fn(); setOk(done); onChange(); } catch (x: any) { setErr(x.message); }
  };
  const sum = crit.reduce((a, c) => a + (Number(c.max_marks) || 0), 0);

  return (
    <div className="rounded-md border border-slate-200">
      <button className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm" onClick={() => setOpen(!open)}>
        <span className="min-w-0"><b>{q.label}</b> <span className="text-slate-600">{q.text.slice(0, 110)}{q.text.length > 110 ? "…" : ""}</span></span>
        <span className="flex shrink-0 items-center gap-1.5">
          <Badge tone={q.max_marks === null ? "red" : "slate"}>{q.max_marks === null ? "marks?" : `${fmt(q.max_marks)} marks`}</Badge>
          <Badge tone={q.reference_approved ? "green" : "amber"}>{q.reference_approved ? "ref ✓" : q.reference_answer ? "ref unapproved" : "no ref"}</Badge>
          <Badge tone={q.criteria.length ? "green" : "amber"}>{q.criteria.length ? "rubric ✓" : "no rubric"}</Badge>
        </span>
      </button>
      {open && (
        <div className="space-y-4 border-t border-slate-100 p-3">
          {q.needs_review && <Notice kind="warn">{q.review_reasons.join(" ") || "Needs review."}</Notice>}
          {err && <Notice kind="error">{err}</Notice>}
          {ok && <Notice kind="success">{ok}</Notice>}
          <div className="grid gap-2 md:grid-cols-[100px_1fr_110px_160px]">
            <Field label="Label"><input className={inputCls} value={f.label} onChange={(x) => setF({ ...f, label: x.target.value })} /></Field>
            <Field label="Question text"><textarea className={inputCls} rows={2} value={f.text} onChange={(x) => setF({ ...f, text: x.target.value })} /></Field>
            <Field label="Max marks"><input className={inputCls} type="number" min="0" step="0.5" value={f.max_marks} onChange={(x) => setF({ ...f, max_marks: x.target.value })} /></Field>
            <Field label="Type"><select className={inputCls} value={f.qtype} onChange={(x) => setF({ ...f, qtype: x.target.value })}>{QTYPES.map((t) => <option key={t}>{t}</option>)}</select></Field>
          </div>
          <div className="flex gap-2">
            <Button variant="primary" onClick={() => run(() => api.patch(`/questions/${q.id}`, { ...f, max_marks: f.max_marks === "" ? null : Number(f.max_marks) }), "Question saved.")}>Save question</Button>
            <Button variant="danger" onClick={() => confirm(`Delete ${q.label}?`) && run(() => api.del(`/questions/${q.id}`), "Deleted.")}><Trash2 className="h-4 w-4" />Delete</Button>
          </div>

          <div>
            <Field label="Reference answer" hint={q.reference_source === "ai_draft" && !q.reference_approved ? "AI-generated draft: it is NOT used for grading until you approve it." : "Saving your own text marks it approved."}>
              <textarea className={inputCls} rows={4} value={ref} onChange={(x) => setRef(x.target.value)} />
            </Field>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button variant="primary" disabled={!ref.trim()} onClick={() => run(() => api.put(`/questions/${q.id}/reference`, { text: ref }), "Reference answer saved and approved.")}>Save reference</Button>
              {q.reference_answer && !q.reference_approved && <Button onClick={() => run(() => api.post(`/questions/${q.id}/reference/approve`), "Approved.")}>Approve AI draft</Button>}
              <Button onClick={() => run(async () => { const r = await api.post<Question>(`/questions/${q.id}/reference/draft`); setRef(r.reference_answer); }, "AI draft created. Review and approve it.")}>Generate AI draft</Button>
            </div>
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-sm font-medium">Marking rubric</span>
              <span className={`text-xs ${q.max_marks !== null && Math.abs(sum - q.max_marks) > 1e-6 ? "text-red-600" : "text-slate-500"}`}>criteria total {fmt(sum)} / question {fmt(q.max_marks)}</span>
            </div>
            <div className="space-y-2">{crit.map((c, i) => (
              <CriterionRow key={i} c={c} onChange={(n) => setCrit(crit.map((x, j) => (j === i ? n : x)))} onRemove={() => setCrit(crit.filter((_, j) => j !== i))} />))}</div>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setCrit([...crit, { name: "", max_marks: 1, concepts: [{ text: "", keywords: [] }] }])}><Plus className="h-4 w-4" />Add criterion</Button>
              <Button variant="primary" disabled={!crit.length} onClick={() => run(async () => {
                try { await api.put(`/questions/${q.id}/rubric`, { criteria: crit }); }
                catch (x: any) {
                  if (x.status === 409 && confirm(`${x.message}\n\nDiscard those evaluations and save the new rubric?`)) await api.put(`/questions/${q.id}/rubric?force=1`, { criteria: crit });
                  else throw x;
                }
              }, "Rubric saved.")}>Save rubric</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CriterionRow({ c, onChange, onRemove }: { c: Criterion; onChange: (c: Criterion) => void; onRemove: () => void }) {
  const setConcept = (i: number, n: Concept) => onChange({ ...c, concepts: c.concepts.map((x, j) => (j === i ? n : x)) });
  return (
    <div className="rounded-md bg-slate-50 p-2">
      <div className="flex gap-2">
        <input className={inputCls} placeholder="Criterion name" value={c.name} onChange={(x) => onChange({ ...c, name: x.target.value })} />
        <input className={`${inputCls} !w-24`} type="number" min="0" step="0.5" value={c.max_marks} onChange={(x) => onChange({ ...c, max_marks: Number(x.target.value) })} />
        <Button variant="danger" onClick={onRemove}><Trash2 className="h-4 w-4" /></Button>
      </div>
      {c.concepts.map((k, i) => (
        <div key={i} className="mt-1 grid gap-2 pl-3 md:grid-cols-[2fr_1fr_1fr_auto]">
          <input className={inputCls} placeholder="Expected concept (a full statement)" value={k.text} onChange={(x) => setConcept(i, { ...k, text: x.target.value })} />
          <input className={inputCls} placeholder="keywords, comma separated" value={k.keywords.join(", ")} onChange={(x) => setConcept(i, { ...k, keywords: x.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
          <input className={inputCls} placeholder="synonyms, comma separated" value={(k.synonyms ?? []).join(", ")} onChange={(x) => setConcept(i, { ...k, synonyms: x.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
          <Button onClick={() => onChange({ ...c, concepts: c.concepts.filter((_, j) => j !== i) })}>×</Button>
        </div>))}
      <button className="mt-1 pl-3 text-xs text-indigo-700 hover:underline" onClick={() => onChange({ ...c, concepts: [...c.concepts, { text: "", keywords: [] }] })}>+ add concept</button>
    </div>
  );
}
