import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Clock,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sparkles,
  Check,
  AlertCircle,
  FileCheck,
} from "lucide-react";
import { getIaTotal, getSubmissions, verifyAnswer } from "../services/api";

export default function ReviewGrade({ examId, refreshKey, onCalibrated }) {
  const [students, setStudents] = useState([]);
  const [selected, setSelected] = useState("");
  const [rows, setRows] = useState([]);
  const [ia, setIa] = useState(null);
  const [drafts, setDrafts] = useState({});
  const [activePageIdx, setActivePageIdx] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [savingQid, setSavingQid] = useState(null);

  const load = async (studentId) => {
    if (!examId) return;
    try {
      const { data } = await getSubmissions(examId, studentId || undefined);
      setStudents(data.students || []);
      const pick = studentId || data.students?.[0] || "";
      setSelected(pick);
      const filtered = pick ? data.submissions.filter((s) => s.student_id === pick) : data.submissions;
      setRows(filtered);

      const next = {};
      filtered.forEach((s) => {
        next[s.question_id] = {
          teacher_score: s.teacher_score ?? s.ai_score ?? 0,
          feedback: s.ai_feedback || "",
        };
      });
      setDrafts(next);

      if (pick) {
        const iaRes = await getIaTotal(examId, pick);
        setIa(iaRes.data);
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    }
  };

  useEffect(() => {
    load().catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examId, refreshKey]);

  const images = useMemo(() => {
    const urls = rows[0]?.image_urls || [];
    return urls;
  }, [rows]);

  const approve = async (qid) => {
    setError("");
    setMessage("");
    setSavingQid(qid);
    try {
      const { data } = await verifyAnswer({
        exam_id: examId,
        student_id: selected,
        question_id: qid,
        teacher_score: Number(drafts[qid]?.teacher_score ?? 0),
        feedback: drafts[qid]?.feedback || "",
      });
      setMessage(
        `Question ${qid} verified! Rubric weights continuously calibrated and logged to training dataset.`
      );
      await load(selected);
      onCalibrated?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setSavingQid(null);
    }
  };

  if (!examId) {
    return (
      <div className="surface-card text-center py-12 text-slate-500 text-sm">
        Please select or create an assessment workspace in Question Studio.
      </div>
    );
  }

  if (!students.length) {
    return (
      <div className="surface-card text-center py-12 text-slate-500 text-sm">
        No evaluated booklets found for this exam. Grade a student booklet in <b>02. Batch Grading</b> to view side-by-side verification.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top Banner: Student Selector & SDMCET IA Choice Rule Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-4 surface-card flex flex-col justify-between">
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Select Student USN
            </label>
            <select
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold bg-white"
              value={selected}
              onChange={(e) => {
                setActivePageIdx(0);
                load(e.target.value);
              }}
            >
              {students.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Showing {rows.length} extracted answers across {images.length} handwritten booklet pages.
          </p>
        </div>

        <div className="lg:col-span-8 surface-card flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 rounded-full">
              SDMCET IA Choice Rule
            </span>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              IA Total = max(Q1A + Q1B, Q2A + Q2B) + (Q3A + Q3B) / 20 Marks
            </p>
            <div className="flex flex-wrap gap-3 mt-2 text-xs">
              <span className="bg-slate-100 rounded-md px-2.5 py-1 text-slate-700">
                Q1 Pair: <b>{ia?.q1_total ?? 0}</b> / 10
              </span>
              <span className="bg-slate-100 rounded-md px-2.5 py-1 text-slate-700">
                Q2 Pair: <b>{ia?.q2_total ?? 0}</b> / 10
              </span>
              <span className="bg-indigo-50 border border-indigo-200 rounded-md px-2.5 py-1 text-indigo-800">
                Best Choice: <b>{ia?.chosen_pair ?? "—"} ({Math.max(ia?.q1_total ?? 0, ia?.q2_total ?? 0)})</b>
              </span>
              <span className="bg-slate-100 rounded-md px-2.5 py-1 text-slate-700">
                Compulsory Q3: <b>{ia?.q3_total ?? 0}</b> / 10
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="text-xs font-semibold text-slate-500 block uppercase tracking-wide">Final IA Marks</span>
            <span className="text-3xl font-extrabold text-indigo-600 tracking-tight">
              {ia?.ia_total ?? 0} <span className="text-sm font-semibold text-slate-400">/ 20</span>
            </span>
          </div>
        </div>
      </div>

      {message && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3.5 flex items-center gap-2 text-sm text-emerald-800">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{message}</span>
        </div>
      )}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-3.5 flex items-center gap-2 text-sm text-red-800">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Verification Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* LEFT COLUMN: Handwritten Booklet Viewer */}
        <aside className="xl:col-span-5 surface-card space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h4 className="font-bold text-slate-900 text-sm">Original Handwritten Booklet</h4>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setZoomLevel((z) => Math.max(60, z - 20))}
                className="p-1 rounded hover:bg-slate-100 text-slate-500"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono text-slate-600">{zoomLevel}%</span>
              <button
                type="button"
                onClick={() => setZoomLevel((z) => Math.min(180, z + 20))}
                className="p-1 rounded hover:bg-slate-100 text-slate-500"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => setZoomLevel(100)}
                className="p-1 rounded hover:bg-slate-100 text-slate-500"
                title="Reset Zoom"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {images.length > 0 ? (
            <div>
              {/* Page Navigator Tabs */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1 overflow-x-auto py-1 max-w-[280px]">
                  {images.map((_, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setActivePageIdx(idx)}
                      className={`text-xs px-2.5 py-1 rounded-md font-semibold transition-colors ${
                        activePageIdx === idx
                          ? "bg-indigo-600 text-white shadow-sm"
                          : "bg-slate-100 hover:bg-slate-200 text-slate-600"
                      }`}
                    >
                      P{idx + 1}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    disabled={activePageIdx === 0}
                    onClick={() => setActivePageIdx((i) => Math.max(0, i - 1))}
                    className="p-1 rounded border border-slate-200 hover:bg-slate-50 disabled:opacity-30"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="text-xs text-slate-500 font-medium">
                    {activePageIdx + 1} of {images.length}
                  </span>
                  <button
                    type="button"
                    disabled={activePageIdx === images.length - 1}
                    onClick={() => setActivePageIdx((i) => Math.min(images.length - 1, i + 1))}
                    className="p-1 rounded border border-slate-200 hover:bg-slate-50 disabled:opacity-30"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Main Image View */}
              <div className="overflow-auto max-h-[640px] border border-slate-200 rounded-xl bg-slate-100 p-2 flex justify-center">
                <img
                  src={images[activePageIdx]}
                  alt={`Student Answer Page ${activePageIdx + 1}`}
                  style={{ width: `${zoomLevel}%` }}
                  className="rounded-lg shadow-sm border border-slate-300 transition-all bg-white"
                />
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500 py-8 text-center">No images found for this submission.</p>
          )}
        </aside>

        {/* RIGHT COLUMN: AI Grading & Teacher Override */}
        <div className="xl:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-bold text-slate-900 text-sm">
              Extracted Question Answers &amp; Continuous Calibration
            </h4>
            <span className="text-xs text-slate-500">
              Teacher adjustments calibrate weights online
            </span>
          </div>

          {rows.map((sub) => {
            const isVerified = Boolean(sub.teacher_verified);
            const isSaving = savingQid === sub.question_id;

            return (
              <article key={sub.question_id} className="surface-card space-y-3">
                {/* Header row */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-base text-slate-900 tracking-tight">
                      Question {sub.question_id}
                    </span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                      Max: 5 Marks
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 inline-flex items-center gap-1">
                      <Sparkles className="w-3 h-3 text-indigo-600" />
                      AI Score: <b>{sub.ai_score ?? 0}</b> / 5
                    </span>
                    <span
                      className={`text-xs font-semibold px-2.5 py-1 rounded-full inline-flex items-center gap-1 ${
                        isVerified
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-amber-100 text-amber-800"
                      }`}
                    >
                      {isVerified ? (
                        <>
                          <Check className="w-3 h-3" /> Teacher Verified
                        </>
                      ) : (
                        <>
                          <Clock className="w-3 h-3" /> Pending Review
                        </>
                      )}
                    </span>
                  </div>
                </div>

                {/* Transcribed Answer Box */}
                <div>
                  <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 block mb-1">
                    Digitized Handwritten Content (Qwen2-VL)
                  </label>
                  <pre className="text-xs leading-relaxed font-mono whitespace-pre-wrap bg-slate-50 border border-slate-200 rounded-lg p-3 max-h-44 overflow-y-auto text-slate-800">
                    {sub.transcribed_text}
                  </pre>
                </div>

                {/* Matched vs Missing Rubric Points */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                  <div className="bg-emerald-50/60 border border-emerald-100 rounded-lg p-2.5">
                    <span className="text-xs font-bold text-emerald-800 block mb-1">
                      Matched Rubric Points ({sub.matched_points?.length || 0})
                    </span>
                    {sub.matched_points?.length ? (
                      <ul className="text-xs text-emerald-950 space-y-1 list-disc pl-4">
                        {sub.matched_points.map((pt, i) => (
                          <li key={i}>{pt}</li>
                        ))}
                      </ul>
                    ) : (
                      <span className="text-xs text-slate-400 italic">None matched</span>
                    )}
                  </div>

                  <div className="bg-red-50/60 border border-red-100 rounded-lg p-2.5">
                    <span className="text-xs font-bold text-red-800 block mb-1">
                      Missing Rubric Points ({sub.missing_points?.length || 0})
                    </span>
                    {sub.missing_points?.length ? (
                      <ul className="text-xs text-red-950 space-y-1 list-disc pl-4">
                        {sub.missing_points.map((pt, i) => (
                          <li key={i}>{pt}</li>
                        ))}
                      </ul>
                    ) : (
                      <span className="text-xs text-emerald-700 italic">All rubric criteria covered!</span>
                    )}
                  </div>
                </div>

                {/* Teacher Score & Remarks Input */}
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2">
                  <div className="sm:col-span-1">
                    <label className="text-xs font-semibold text-slate-600 block mb-1">
                      Approved Score
                    </label>
                    <input
                      type="number"
                      step="0.5"
                      min="0"
                      max="5"
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-center text-slate-900 bg-white"
                      value={drafts[sub.question_id]?.teacher_score ?? 0}
                      onChange={(e) =>
                        setDrafts((d) => ({
                          ...d,
                          [sub.question_id]: { ...d[sub.question_id], teacher_score: e.target.value },
                        }))
                      }
                    />
                  </div>

                  <div className="sm:col-span-3">
                    <label className="text-xs font-semibold text-slate-600 block mb-1">
                      Teacher Feedback / Examiner Remarks
                    </label>
                    <textarea
                      className="w-full rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-800 min-h-[42px] bg-white"
                      value={drafts[sub.question_id]?.feedback ?? ""}
                      placeholder="Add examiner feedback or notes..."
                      onChange={(e) =>
                        setDrafts((d) => ({
                          ...d,
                          [sub.question_id]: { ...d[sub.question_id], feedback: e.target.value },
                        }))
                      }
                    />
                  </div>
                </div>

                {/* Approve Button */}
                <button
                  type="button"
                  onClick={() => approve(sub.question_id)}
                  disabled={isSaving}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-sm font-semibold py-2.5 transition-colors shadow-sm disabled:opacity-50"
                >
                  <FileCheck className="w-4 h-4" />
                  {isSaving
                    ? "Updating & Calibrating..."
                    : `Approve Grade & Calibrate Rubric (${sub.question_id})`}
                </button>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
