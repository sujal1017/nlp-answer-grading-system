import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Download } from "lucide-react";
import { api, waitForJob, type Answer, type Evaluation, type Question, type Submission } from "../api";
import { Badge, Button, Card, Empty, Notice, Progress, Spinner, StatusBadge, fmt, inputCls, useLoad } from "../ui";

type Tab = "pages" | "mapping" | "evaluate" | "results";
type Msg = { kind: "info" | "error" | "success" | "warn"; text: string } | null;

export default function SubmissionPage() {
  const { id } = useParams();
  const nav = useNavigate();
  const sub = useLoad(() => api.get<Submission>(`/submissions/${id}`), [id]);
  const [tab, setTab] = useState<Tab>("pages");
  const [msg, setMsg] = useState<Msg>(null);
  const [job, setJob] = useState<{ p: number; m: string } | null>(null);

  if (sub.loading) return <Spinner />;
  if (sub.error || !sub.data) return <Notice kind="error">{sub.error}</Notice>;
  const s = sub.data;

  async function runJob(start: () => Promise<{ job: { id: number; status: string } }>, okText: string) {
    setMsg(null); setJob({ p: 0, m: "Starting…" });
    try {
      const r = await start();
      if (r.job.status !== "done") {
        const j = await waitForJob(r.job.id, (x) => setJob({ p: x.progress, m: x.message }));
        if (j.status === "failed") throw new Error(j.error || "Processing failed");
      }
      setMsg({ kind: "success", text: okText });
      await sub.reload();
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); } finally { setJob(null); }
  }

  async function remove() {
    if (!confirm("Delete this submission and its saved evaluations?")) return;
    try { await api.del(`/submissions/${s.id}`); nav(`/exams/${s.exam_id}`); } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  const tabs: [Tab, string][] = [["pages", "1. Pages & OCR"], ["mapping", "2. Answer mapping"], ["evaluate", "3. Evaluation"], ["results", "4. Results"]];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <Link to={`/exams/${s.exam_id}`} className="text-xs text-indigo-700 underline">← {s.exam_name}</Link>
          <h1 className="text-xl font-semibold">{s.student_identifier || "Unnamed student"} <StatusBadge s={s.status} /></h1>
          <p className="text-sm text-slate-500">{s.filename} · {s.page_count} page(s)</p>
        </div>
        <Button variant="danger" onClick={remove}>Delete submission</Button>
      </div>

      {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}
      {job && <Card><Progress value={job.p} label={job.m} /></Card>}
      {["uploaded", "ocr_running"].includes(s.status) && !job && (
        <Notice kind="warn">Text extraction has not finished. <button className="underline" onClick={() => sub.reload()}>Refresh</button> or re-run OCR from the Pages tab.</Notice>
      )}

      <div className="flex gap-1 border-b border-slate-200">
        {tabs.map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${tab === k ? "border-indigo-600 text-indigo-700" : "border-transparent text-slate-500 hover:text-slate-800"}`}>{label}</button>
        ))}
      </div>

      {tab === "pages" && <PagesTab s={s} reload={sub.reload} setMsg={setMsg}
        rerun={() => runJob(() => api.post(`/submissions/${s.id}/ocr`), "OCR finished. Review the text before grading.")} busy={!!job} />}
      {tab === "mapping" && <MappingTab s={s} reload={sub.reload} setMsg={setMsg} />}
      {tab === "evaluate" && <EvaluateTab s={s} reload={sub.reload} setMsg={setMsg} busy={!!job}
        run={(force) => runJob(() => api.post(`/submissions/${s.id}/evaluate`, { force }), "Evaluation finished. AI marks are suggestions until you approve them.")} />}
      {tab === "results" && <ResultsTab s={s} />}
    </div>
  );
}

// ---------------------------------------------------------------- pages / OCR review
function PagesTab({ s, reload, setMsg, rerun, busy }: { s: Submission; reload: () => void; setMsg: (m: Msg) => void; rerun: () => void; busy: boolean }) {
  const pages = s.pages ?? [];
  const [n, setN] = useState(pages[0]?.page_number ?? 1);
  const page = pages.find((p) => p.page_number === n);
  const [text, setText] = useState("");
  useEffect(() => { setText(page ? page.text_corrected ?? page.text_original : ""); }, [page?.page_number, page?.text_corrected, page?.text_original]);

  async function save(reviewed: boolean) {
    try {
      await api.patch(`/submissions/${s.id}/pages/${n}`, { text_corrected: text, reviewed });
      setMsg({ kind: "success", text: `Page ${n} saved${reviewed ? " and marked reviewed" : ""}.` });
      reload();
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  if (pages.length === 0) return <Empty>No pages have been extracted yet. <Button className="ml-2" onClick={rerun} busy={busy}>Run OCR</Button></Empty>;
  const changed = page && text !== (page.text_corrected ?? page.text_original);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {pages.map((p) => (
          <button key={p.page_number} onClick={() => setN(p.page_number)}
            className={`rounded px-2.5 py-1 text-sm ${p.page_number === n ? "bg-indigo-600 text-white" : "border border-slate-300 bg-white"}`}>
            {p.page_number}{p.reviewed ? " ✓" : ""}{p.error || p.warnings.length ? " ⚠" : ""}
          </button>
        ))}
        <Button className="ml-auto" onClick={rerun} busy={busy}>Re-run OCR</Button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={`Original page ${n}`}>
          <img src={`/api/submissions/${s.id}/pages/${n}/image`} alt={`Page ${n}`} className="max-h-[75vh] w-full rounded border border-slate-200 object-contain" />
        </Card>
        <Card title="Extracted text (editable)" actions={page && <Badge>{page.kind} · {page.method}</Badge>}>
          {page && (
            <div className="space-y-2">
              <p className="text-xs text-slate-500">Engine: {page.engine || "n/a"} · confidence: {page.confidence === null ? "not provided by engine" : fmt(page.confidence)}</p>
              {page.error && <Notice kind="error">{page.error}</Notice>}
              {page.warnings.map((w, i) => <Notice key={i} kind="warn">{w}</Notice>)}
              <textarea className={`${inputCls} h-[55vh] font-mono`} value={text} onChange={(e) => setText(e.target.value)} />
              <div className="flex gap-2">
                <Button onClick={() => save(false)} disabled={!changed}>Save correction</Button>
                <Button variant="primary" onClick={() => save(true)}>Save &amp; mark reviewed</Button>
              </div>
              {page.text_corrected !== null && <details className="text-xs text-slate-500"><summary className="cursor-pointer">Original OCR output</summary><pre className="mt-1 whitespace-pre-wrap">{page.text_original}</pre></details>}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- mapping
function MappingTab({ s, reload, setMsg }: { s: Submission; reload: () => void; setMsg: (m: Msg) => void }) {
  const questions = s.questions ?? [];
  const answers = s.answers ?? [];
  const unreviewed = (s.pages ?? []).filter((p) => !p.reviewed).length;

  async function remap() {
    if (!confirm("Re-map from the current page text? Existing mapping edits and evaluations may be replaced.")) return;
    try { await api.post(`/submissions/${s.id}/map`); setMsg({ kind: "success", text: "Answers re-mapped." }); reload(); } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  return (
    <div className="space-y-3">
      {unreviewed > 0 && <Notice kind="warn">{unreviewed} page(s) have not been marked reviewed. Correct OCR errors first, then re-map.</Notice>}
      {(s.mapping_notes ?? []).map((n, i) => <Notice key={i} kind="warn">{n}</Notice>)}
      <div className="flex justify-between">
        <p className="text-sm text-slate-500">Confirm each answer belongs to the right question. Unanswered and unreadable are kept distinct.</p>
        <Button onClick={remap}>Re-map from page text</Button>
      </div>
      {answers.length === 0 && <Empty>No answers have been mapped yet.</Empty>}
      {answers.map((a) => <AnswerRow key={a.id} a={a} questions={questions} reload={reload} setMsg={setMsg} />)}
      {s.unmapped_text?.trim() && <Card title="Text not assigned to any question"><pre className="whitespace-pre-wrap text-sm">{s.unmapped_text}</pre></Card>}
    </div>
  );
}

function AnswerRow({ a, questions, reload, setMsg }: { a: Answer; questions: Question[]; reload: () => void; setMsg: (m: Msg) => void }) {
  const [text, setText] = useState(a.text_corrected ?? a.text_original);
  const [qid, setQid] = useState(a.question_id);
  const [status, setStatus] = useState(a.status);
  useEffect(() => { setText(a.text_corrected ?? a.text_original); setQid(a.question_id); setStatus(a.status); }, [a.id, a.text_corrected, a.text_original, a.question_id, a.status]);

  async function save(confirm: boolean) {
    try {
      await api.patch(`/answers/${a.id}`, { text_corrected: text, question_id: qid, status, ...(confirm ? { mapping_confirmed: true } : {}) });
      setMsg({ kind: "success", text: `${a.question_label} updated.` });
      reload();
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  return (
    <Card title={<span>{a.question_label} <StatusBadge s={a.status} /> {a.mapping_confirmed && <Badge tone="green">confirmed</Badge>}</span>}
      actions={<span className="text-xs text-slate-500">pages: {a.pages.join(", ") || "—"}</span>}>
      <div className="space-y-2">
        {a.notes.map((n, i) => <Notice key={i} kind="warn">{n}</Notice>)}
        <textarea className={`${inputCls} h-28`} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="flex flex-wrap items-center gap-2">
          <select className={`${inputCls} w-auto`} value={qid} onChange={(e) => setQid(Number(e.target.value))}>
            {questions.map((q) => <option key={q.id} value={q.id}>{q.label}</option>)}
          </select>
          <select className={`${inputCls} w-auto`} value={status} onChange={(e) => setStatus(e.target.value)}>
            {["answered", "unanswered", "unreadable", "ambiguous"].map((x) => <option key={x}>{x}</option>)}
          </select>
          <Button onClick={() => save(false)}>Save</Button>
          <Button variant="primary" onClick={() => save(true)}>Save &amp; confirm mapping</Button>
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------- evaluation workspace
function EvaluateTab({ s, reload, setMsg, run, busy }: { s: Submission; reload: () => void; setMsg: (m: Msg) => void; run: (force: boolean) => void; busy: boolean }) {
  const qById = new Map((s.questions ?? []).map((q) => [q.id, q]));
  const answers = s.answers ?? [];
  const unapproved = (s.questions ?? []).filter((q) => !q.reference_approved || !q.max_marks);
  const unconfirmed = answers.filter((a) => a.status === "ambiguous" || (a.status === "answered" && !a.mapping_confirmed)).length;

  return (
    <div className="space-y-3">
      {unapproved.length > 0 && <Notice kind="warn">Questions without an approved reference answer or confirmed marks will be sent to manual grading: {unapproved.map((q) => q.label).join(", ")}.</Notice>}
      {unconfirmed > 0 && <Notice kind="warn">{unconfirmed} answer mapping(s) are not confirmed yet (Mapping tab).</Notice>}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" busy={busy} onClick={() => run(false)}>Evaluate answers</Button>
        <Button busy={busy} onClick={() => { if (confirm("Re-evaluate everything, replacing existing AI results (human overrides are kept in the record)?")) run(true); }}>Re-evaluate all</Button>
        <span className="text-xs text-slate-500">AI marks are suggestions. Nothing is final until you approve it.</span>
      </div>
      {answers.length === 0 && <Empty>Map answers to questions first.</Empty>}
      {answers.map((a) => <EvalCard key={a.id} a={a} q={qById.get(a.question_id)} reload={reload} setMsg={setMsg} />)}
    </div>
  );
}

function EvalCard({ a, q, reload, setMsg }: { a: Answer; q?: Question; reload: () => void; setMsg: (m: Msg) => void }) {
  const ev: Evaluation | null = a.evaluation;
  const ai = ev?.ai_result ?? null;
  const crits = q?.criteria ?? [];
  const initial = () => {
    const m: Record<string, string> = {};
    for (const c of crits) {
      const h = ev?.human_criterion_scores?.[String(c.id)];
      const aiScore = ai?.criteria.find((x) => x.criterion_id === c.id)?.awarded;
      m[String(c.id)] = h !== undefined ? String(h) : aiScore !== undefined ? String(aiScore) : "";
    }
    return m;
  };
  const [scores, setScores] = useState<Record<string, string>>(initial);
  const [total, setTotal] = useState(ev?.human_total !== null && ev?.human_total !== undefined ? String(ev.human_total) : "");
  const [comment, setComment] = useState(ev?.reviewer_comment ?? "");
  const [busy, setBusy] = useState(false);
  useEffect(() => { setScores(initial()); setTotal(ev?.human_total != null ? String(ev.human_total) : ""); setComment(ev?.reviewer_comment ?? ""); },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ev?.id, ev?.ai_total, ev?.human_total, ev?.approved]);

  async function submit(approve: boolean) {
    if (!ev) return;
    setBusy(true);
    try {
      const payload: any = { comment, approve };
      if (crits.length > 0) {
        const cs: Record<string, number> = {};
        for (const [k, v] of Object.entries(scores)) if (v !== "") cs[k] = Number(v);
        payload.criterion_scores = cs;
      } else if (total !== "") payload.total = Number(total);
      await api.patch(`/evaluations/${ev.id}/review`, payload);
      setMsg({ kind: "success", text: `${a.question_label}: ${approve ? "approved" : "review saved"}.` });
      reload();
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); } finally { setBusy(false); }
  }

  async function manual() {
    try { await api.post(`/answers/${a.id}/manual-evaluation`); reload(); } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  const answerText = a.text_corrected ?? a.text_original;
  return (
    <Card title={<span>{a.question_label} <StatusBadge s={ev?.status ?? "not_evaluated"} /> {ev?.approved && <Badge tone="green">approved</Badge>}</span>}
      actions={<span className="text-sm font-medium">{ev ? `Final ${fmt(ev.final_total)} · AI ${fmt(ev.ai_total)}` : ""} / {fmt(q?.max_marks)}</span>}>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-3 text-sm">
          <div><h3 className="font-medium">Question</h3><p className="whitespace-pre-wrap text-slate-700">{q?.text}</p></div>
          <div><h3 className="font-medium">Reference answer {q && !q.reference_approved && <Badge tone="amber">not approved</Badge>}</h3><p className="whitespace-pre-wrap text-slate-700">{q?.reference_answer || "—"}</p></div>
          <div><h3 className="font-medium">Student answer ({a.status})</h3><p className="whitespace-pre-wrap rounded bg-slate-50 p-2 text-slate-800">{answerText.trim() || "(no text)"}</p></div>
        </div>

        <div className="space-y-3 text-sm">
          {!ev && <><Notice kind="info">Not evaluated yet. Run evaluation above, or grade this answer manually.</Notice><Button onClick={manual}>Grade manually</Button></>}
          {ev?.ai_error && <Notice kind="error">{ev.ai_error}</Notice>}
          {ev?.ai_flags.map((f, i) => <Notice key={i} kind="warn">{f}</Notice>)}

          {ai && (
            <div className="space-y-2">
              {ai.criteria.map((c) => (
                <div key={c.criterion_id} className="rounded border border-slate-200 p-2">
                  <div className="flex justify-between font-medium"><span>{c.name}</span><span>{fmt(c.awarded)} / {fmt(c.max_marks)}</span></div>
                  {c.rationale && <p className="text-xs text-slate-500">{c.rationale}</p>}
                  <ul className="mt-1 space-y-1">
                    {c.concepts.map((k) => (
                      <li key={k.concept_id} className="text-xs">
                        <StatusBadge s={k.status} /> {k.concept_text}
                        {k.evidence && <q className="ml-1 italic text-slate-500">{k.evidence}</q>}
                        {k.comment && <span className="ml-1 text-slate-500">— {k.comment}</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              <Bullets title="Why marks were awarded" items={ai.awarded_because} />
              <Bullets title="Why marks were withheld" items={ai.withheld_because} />
              <Bullets title="Incorrect statements" items={ai.incorrect_statements} />
              <Bullets title="How to improve" items={ai.improvements} />
              <Bullets title="A stronger answer would include" items={ai.stronger_answer_outline} />
            </div>
          )}

          {ev && (
            <div className="space-y-2 rounded border border-indigo-200 bg-indigo-50/40 p-3">
              <h3 className="font-medium">Human review</h3>
              {crits.length > 0 ? (
                <div className="grid grid-cols-2 gap-2">
                  {crits.map((c) => (
                    <label key={c.id} className="text-xs">{c.name} (max {fmt(c.max_marks)})
                      <input type="number" step="0.25" min={0} max={c.max_marks} className={inputCls} value={scores[String(c.id)] ?? ""}
                        onChange={(e) => setScores({ ...scores, [String(c.id)]: e.target.value })} />
                    </label>
                  ))}
                </div>
              ) : (
                <label className="block text-xs">Total marks (max {fmt(q?.max_marks)})
                  <input type="number" step="0.25" min={0} max={q?.max_marks ?? undefined} className={inputCls} value={total} onChange={(e) => setTotal(e.target.value)} />
                </label>
              )}
              <textarea className={`${inputCls} h-16`} placeholder="Reviewer comment" value={comment} onChange={(e) => setComment(e.target.value)} />
              <div className="flex gap-2">
                <Button busy={busy} onClick={() => submit(false)}>Save review</Button>
                <Button variant="primary" busy={busy} onClick={() => submit(true)}>Approve final mark</Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function Bullets({ title, items }: { title: string; items: string[] }) {
  if (!items || items.length === 0) return null;
  return <div><h4 className="text-xs font-semibold uppercase text-slate-500">{title}</h4><ul className="list-disc pl-5 text-sm">{items.map((x, i) => <li key={i}>{x}</li>)}</ul></div>;
}

// ---------------------------------------------------------------- results
function ResultsTab({ s }: { s: Submission }) {
  const t = s.totals;
  if (!t) return <Empty>No results yet.</Empty>;
  const strengths: string[] = [], weaknesses: string[] = [];
  for (const a of s.answers ?? []) {
    for (const c of a.evaluation?.ai_result?.criteria ?? [])
      for (const k of c.concepts) {
        if (k.status === "correct") strengths.push(`${a.question_label}: ${k.concept_text}`);
        else if (["missing", "incorrect", "partial"].includes(k.status)) weaknesses.push(`${a.question_label}: ${k.concept_text} (${k.status})`);
      }
  }
  return (
    <div className="space-y-4">
      {!t.all_final && <Notice kind="warn">Not every question has an approved final mark. Totals below count only approved marks and reports will show the unapproved ones as pending.</Notice>}
      <Card title="Summary" actions={<>
        <a href={`/api/submissions/${s.id}/report.pdf`}><Button><Download className="h-4 w-4" />PDF report</Button></a>
        <a href={`/api/submissions/${s.id}/report.csv`}><Button><Download className="h-4 w-4" />CSV</Button></a>
      </>}>
        <p className="text-2xl font-semibold">{fmt(t.total)} <span className="text-base font-normal text-slate-500">/ {fmt(t.max_total)}{t.percentage !== null && ` · ${fmt(t.percentage)}%`}</span></p>
        <table className="mt-3 w-full text-sm">
          <thead><tr className="border-b text-left text-xs text-slate-500"><th className="py-1">Question</th><th>Marks</th><th>Max</th><th>Basis</th><th>Status</th></tr></thead>
          <tbody>
            {t.questions.map((q) => (
              <tr key={q.question_id} className="border-b border-slate-100">
                <td className="py-1">{q.label}</td><td>{fmt(q.marks)}</td><td>{fmt(q.max_marks)}</td><td>{q.kind}</td><td><StatusBadge s={q.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Strengths (concepts answered correctly)">{strengths.length ? <ul className="list-disc pl-5 text-sm">{strengths.map((x, i) => <li key={i}>{x}</li>)}</ul> : <p className="text-sm text-slate-500">None recorded.</p>}</Card>
        <Card title="Weaknesses (missing, partial or incorrect)">{weaknesses.length ? <ul className="list-disc pl-5 text-sm">{weaknesses.map((x, i) => <li key={i}>{x}</li>)}</ul> : <p className="text-sm text-slate-500">None recorded.</p>}</Card>
      </div>
    </div>
  );
}
