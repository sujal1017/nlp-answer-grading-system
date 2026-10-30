import { Link } from "react-router-dom";
import { api, type Submission } from "../api";
import { Card, Empty, Notice, Spinner, StatusBadge, useLoad, fmt } from "../ui";

export default function HistoryPage() {
  const { data, error, loading } = useLoad(() => api.get<Submission[]>("/submissions"), []);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">History</h1>
      <Card>
        {loading ? <Spinner /> : error ? <Notice kind="error">{error}</Notice> : !data?.length ? <Empty>No saved evaluations yet.</Empty> : (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500"><tr><th className="py-1">Student</th><th>Exam</th><th>Marks</th><th>Status</th><th>Needs review</th><th>Reports</th></tr></thead>
            <tbody className="divide-y">{data.map((s) => (
              <tr key={s.id}>
                <td className="py-2"><Link className="text-indigo-700 hover:underline" to={`/submissions/${s.id}`}>{s.student_identifier || `Sheet #${s.id}`}</Link></td>
                <td>{s.exam_name}</td>
                <td>{s.max_total ? `${fmt(s.total)} / ${fmt(s.max_total)}` : "—"}</td>
                <td><StatusBadge s={s.status} /></td>
                <td>{s.needs_review || 0}</td>
                <td className="space-x-3"><a className="text-indigo-700 hover:underline" href={`/api/submissions/${s.id}/report.pdf`}>PDF</a><a className="text-indigo-700 hover:underline" href={`/api/submissions/${s.id}/report.csv`}>CSV</a></td>
              </tr>))}</tbody>
          </table>)}
      </Card>
    </div>
  );
}
