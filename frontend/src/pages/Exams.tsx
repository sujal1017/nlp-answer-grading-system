import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type Exam } from "../api";
import { Button, Card, Empty, Field, Notice, Spinner, inputCls, useLoad } from "../ui";

export default function Exams() {
  const nav = useNavigate();
  const { data, error, loading, reload } = useLoad(() => api.get<Exam[]>("/exams"), []);
  const [f, setF] = useState({ name: "", subject: "", code: "", choice: "" });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function create() {
    setBusy(true); setErr(null);
    try {
      // "Q1|Q2; Q3|Q4" => [["Q1","Q2"],["Q3","Q4"]] (answer any one of each group)
      const groups = f.choice.split(";").map((g) => g.split("|").map((x) => x.trim()).filter(Boolean)).filter((g) => g.length > 1);
      const e = await api.post<Exam>("/exams", { name: f.name, subject: f.subject, code: f.code, choice_groups: groups });
      nav(`/exams/${e.id}`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Examinations</h1>
      <Card title="New examination">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Name *"><input className={inputCls} value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></Field>
          <Field label="Subject"><input className={inputCls} value={f.subject} onChange={(e) => setF({ ...f, subject: e.target.value })} /></Field>
          <Field label="Course code"><input className={inputCls} value={f.code} onChange={(e) => setF({ ...f, code: e.target.value })} /></Field>
          <Field label="Choice groups (optional)" hint="Answer-any-one groups, e.g. Q1|Q2; Q3|Q4. Only the best of each group counts.">
            <input className={inputCls} value={f.choice} onChange={(e) => setF({ ...f, choice: e.target.value })} />
          </Field>
        </div>
        {err && <div className="mt-3"><Notice kind="error">{err}</Notice></div>}
        <div className="mt-3"><Button variant="primary" busy={busy} disabled={!f.name.trim()} onClick={create}>Create examination</Button></div>
      </Card>
      <Card title="All examinations">
        {loading ? <Spinner /> : error ? <Notice kind="error">{error}</Notice> : !data?.length ? <Empty>No examinations yet.</Empty> : (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500"><tr><th className="py-1">Name</th><th>Subject</th><th>Questions</th><th>Sheets</th><th /></tr></thead>
            <tbody className="divide-y">{data.map((e) => (
              <tr key={e.id}>
                <td className="py-2"><Link className="text-indigo-700 hover:underline" to={`/exams/${e.id}`}>{e.name}</Link></td>
                <td>{e.subject}</td><td>{e.question_count}</td><td>{e.submission_count}</td>
                <td className="text-right"><Button variant="danger" onClick={async () => {
                  if (!confirm(`Delete "${e.name}" and all its answer sheets? This cannot be undone.`)) return;
                  try { await api.del(`/exams/${e.id}`); reload(); } catch (x: any) { alert(x.message); }
                }}>Delete</Button></td>
              </tr>))}</tbody>
          </table>)}
      </Card>
    </div>
  );
}
