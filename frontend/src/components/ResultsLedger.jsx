import { useEffect, useState, useMemo } from "react";
import {
  Download,
  RefreshCw,
  Search,
  CheckCircle2,
  Clock,
  Award,
  Users,
  FileSpreadsheet,
} from "lucide-react";
import { getStudentResults } from "../services/api";

export default function ResultsLedger({ examId, onSelectStudent }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const fetchResults = async () => {
    if (!examId) return;
    setLoading(true);
    setError("");
    try {
      const { data } = await getStudentResults(examId);
      setResults(data.results || []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to load results ledger.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, [examId]);

  const filteredResults = useMemo(() => {
    if (!searchTerm.trim()) return results;
    const term = searchTerm.toLowerCase();
    return results.filter((r) => r.student_id?.toLowerCase().includes(term));
  }, [results, searchTerm]);

  // Summary Metrics
  const metrics = useMemo(() => {
    if (!results.length) return null;
    const total = results.length;
    const verified = results.filter((r) => r.status === "verified").length;
    const scores = results.map((r) => r.final_ia_total || 0);
    const avg = (scores.reduce((a, b) => a + b, 0) / total).toFixed(1);
    const max = Math.max(...scores).toFixed(1);
    const min = Math.min(...scores).toFixed(1);
    return { total, verified, avg, max, min };
  }, [results]);

  const handleDownloadCsv = () => {
    if (!results.length) return;
    const headers = [
      "Student USN",
      "Q1A",
      "Q1B",
      "Q2A",
      "Q2B",
      "Q3A",
      "Q3B",
      "Q1 Total",
      "Q2 Total",
      "Best (Q1, Q2)",
      "Q3 Total",
      "Final IA Total",
      "Max Marks",
      "Verified Submissions",
      "Answered Count",
      "Status",
      "Last Updated",
    ];

    const rows = results.map((r) => {
      const bestModule1 = Math.max(r.q1_total || 0, r.q2_total || 0);
      return [
        r.student_id,
        r.q1a_score ?? 0,
        r.q1b_score ?? 0,
        r.q2a_score ?? 0,
        r.q2b_score ?? 0,
        r.q3a_score ?? 0,
        r.q3b_score ?? 0,
        r.q1_total ?? 0,
        r.q2_total ?? 0,
        bestModule1,
        r.q3_total ?? 0,
        r.final_ia_total ?? 0,
        r.max_ia_marks ?? 20,
        r.verified_count ?? 0,
        r.total_questions_answered ?? 0,
        r.status || "evaluated",
        r.updated_at || "",
      ];
    });

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((row) => row.map((cell) => `"${cell}"`).join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${examId}_marks_ledger.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Top Action & KPI Bar */}
      <div className="surface-card flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5 text-indigo-600" />
            <h3 className="text-lg font-bold text-slate-900">SDMCET IA Student Marks Ledger</h3>
            <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 rounded-full">
              Choice Rule: max(Q1, Q2) + Q3
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Official consolidated assessment ledger for exam <strong>{examId}</strong>. Synchronized automatically from AI evaluation and teacher verification.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={fetchResults}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            type="button"
            onClick={handleDownloadCsv}
            disabled={loading || !results.length}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-3.5 py-2 text-xs font-bold shadow-sm transition-colors"
          >
            <Download className="w-4 h-4" />
            Download Marks CSV
          </button>
        </div>
      </div>

      {/* Quick Summary Cards */}
      {metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Total Booklets</p>
              <p className="text-xl font-bold text-slate-900">{metrics.total}</p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Fully Verified</p>
              <p className="text-xl font-bold text-emerald-700">{metrics.verified}</p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Average IA Score</p>
              <p className="text-xl font-bold text-slate-900">{metrics.avg} <span className="text-xs font-normal text-slate-500">/ 20</span></p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Score Range</p>
              <p className="text-xl font-bold text-slate-900">{metrics.min} – {metrics.max}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Ledger Table Card */}
      <div className="surface-card p-0 overflow-hidden">
        {/* Table Filter Toolbar */}
        <div className="p-4 border-b border-slate-200 bg-slate-50/50 flex flex-wrap items-center justify-between gap-3">
          <div className="relative flex-1 max-w-xs">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by USN / Student ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-300 bg-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
            />
          </div>
          <div className="text-xs text-slate-500">
            Showing <strong>{filteredResults.length}</strong> of <strong>{results.length}</strong> students
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-50 text-red-700 text-xs border-b border-red-100">
            {error}
          </div>
        )}

        {/* Responsive Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-100/80 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-3 px-4">Student USN</th>
                <th className="py-3 px-2 text-center">Q1A</th>
                <th className="py-3 px-2 text-center">Q1B</th>
                <th className="py-3 px-2 text-center">Q2A</th>
                <th className="py-3 px-2 text-center">Q2B</th>
                <th className="py-3 px-2 text-center">Q3A</th>
                <th className="py-3 px-2 text-center">Q3B</th>
                <th className="py-3 px-3 text-center bg-indigo-50/50 text-indigo-900 border-x border-indigo-100">
                  Best (Q1, Q2)
                </th>
                <th className="py-3 px-3 text-center bg-indigo-50/50 text-indigo-900 border-r border-indigo-100">
                  Q3 Total
                </th>
                <th className="py-3 px-4 text-center bg-indigo-100/60 text-indigo-950 font-black">
                  Final Total (/20)
                </th>
                <th className="py-3 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {filteredResults.length === 0 ? (
                <tr>
                  <td colSpan="11" className="py-8 text-center text-slate-400">
                    {loading ? "Loading student scorecards..." : "No student scorecards found for this exam."}
                  </td>
                </tr>
              ) : (
                filteredResults.map((r) => {
                  const q1 = (r.q1a_score || 0) + (r.q1b_score || 0);
                  const q2 = (r.q2a_score || 0) + (r.q2b_score || 0);
                  const best = Math.max(q1, q2);
                  const q3 = (r.q3a_score || 0) + (r.q3b_score || 0);
                  const chosenModule = q1 >= q2 ? "Q1" : "Q2";
                  const isVerified = r.status === "verified";

                  return (
                    <tr
                      key={r.student_id}
                      className="hover:bg-slate-50 transition-colors cursor-pointer group"
                      onClick={() => onSelectStudent?.(r.student_id)}
                      title="Click to view and review this student's answer booklet"
                    >
                      <td className="py-3 px-4 font-mono font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                        {r.student_id}
                      </td>
                      <td className={`py-3 px-2 text-center font-mono ${chosenModule === "Q1" ? "font-semibold text-slate-800" : "text-slate-400"}`}>
                        {r.q1a_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className={`py-3 px-2 text-center font-mono ${chosenModule === "Q1" ? "font-semibold text-slate-800" : "text-slate-400"}`}>
                        {r.q1b_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className={`py-3 px-2 text-center font-mono ${chosenModule === "Q2" ? "font-semibold text-slate-800" : "text-slate-400"}`}>
                        {r.q2a_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className={`py-3 px-2 text-center font-mono ${chosenModule === "Q2" ? "font-semibold text-slate-800" : "text-slate-400"}`}>
                        {r.q2b_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className="py-3 px-2 text-center font-mono font-semibold text-slate-800">
                        {r.q3a_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className="py-3 px-2 text-center font-mono font-semibold text-slate-800">
                        {r.q3b_score?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className="py-3 px-3 text-center font-mono font-bold bg-indigo-50/30 text-indigo-900 border-x border-indigo-100">
                        {best.toFixed(1)}{" "}
                        <span className="text-[10px] font-normal text-indigo-500">({chosenModule})</span>
                      </td>
                      <td className="py-3 px-3 text-center font-mono font-bold bg-indigo-50/30 text-indigo-900 border-r border-indigo-100">
                        {q3.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-center font-mono font-extrabold text-sm bg-indigo-100/40 text-indigo-950">
                        {r.final_ia_total?.toFixed(1) ?? "0.0"}
                      </td>
                      <td className="py-3 px-3 text-center">
                        {isVerified ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                            Verified
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
                            <Clock className="w-3 h-3 text-amber-600" />
                            Evaluated
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
