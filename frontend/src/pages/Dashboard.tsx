import { Link } from "react-router-dom";
import { api, type Submission } from "../api";
import { Card, Empty, Notice, Spinner, StatusBadge, useLoad, fmt } from "../ui";

interface Dash {
  exam_count: number; submission_count: number; processing: number; completed: number; needs_review: number;
  recent: Submission[]; review_queue: Submission[];
}

export default function Dashboard() {
  const { data, error, loading } = useLoad(() => api.get<Dash>("/dashboard"), []);
  if (loading) return <Spinner />;
  if (error || !data) return <Notice kind="error">{error}</Notice>;
  const stats = [
    ["Examinations", data.exam_count], ["Answer sheets", data.submission_count], ["Processing", data.processing],
    ["Needs review", data.needs_review], ["Completed", data.completed],
  ] as const;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Dashboard</h1>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        {stats.map(([k, v]) => (
          <div key={k} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="text-2xl font-semibold">{v}</div><div className="text-xs text-slate-500">{k}</div>
          </div>
        ))}
      </div>
      <Card title="Quick actions">
        <div className="flex flex-wrap gap-2 text-sm">
          <Link className="rounded-md bg-indigo-600 px-3 py-1.5 font-medium text-white" to="/exams">Create or open an examination</Link>
          <Link className="rounded-md border border-slate-300 px-3 py-1.5" to="/history">View history</Link>
          <Link className="rounded-md border border-slate-300 px-3 py-1.5" to="/settings">Settings</Link>
        </div>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Requires review">
          {data.review_queue.length === 0 ? <Empty>Nothing waiting for review.</Empty> : (
            <ul className="divide-y text-sm">{data.review_queue.map((s) => (
              <li key={s.id} className="flex items-center justify-between py-2">
                <Link className="text-indigo-700 hover:underline" to={`/submissions/${s.id}`}>{s.student_identifier || `Sheet #${s.id}`} · {s.exam_name}</Link>
                <StatusBadge s={s.status} />
              </li>))}</ul>)}
        </Card>
        <Card title="Recent evaluations">
          {data.recent.length === 0 ? <Empty>No answer sheets yet. Create an exam, add questions, then upload a sheet.</Empty> : (
            <ul className="divide-y text-sm">{data.recent.map((s) => (
              <li key={s.id} className="flex items-center justify-between py-2">
                <Link className="text-indigo-700 hover:underline" to={`/submissions/${s.id}`}>{s.student_identifier || `Sheet #${s.id}`} · {s.exam_name}</Link>
                <span className="flex items-center gap-2"><span className="text-slate-500">{s.max_total ? `${fmt(s.total)}/${fmt(s.max_total)}` : ""}</span><StatusBadge s={s.status} /></span>
              </li>))}</ul>)}
        </Card>
      </div>
    </div>
  );
}
