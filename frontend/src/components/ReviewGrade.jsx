import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Clock,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sparkles,
  Check,
  AlertCircle,
  FileCheck,
  Scissors,
  RefreshCw,
  Trash2,
  Zap,
  Search,
  CornerDownRight,
  FileSpreadsheet,
} from "lucide-react";
import {
  getBooklet,
  getSubmissions,
  verifyAnswer,
  splitSubmission,
  evaluateStudent,
  deleteSubmission,
} from "../services/api";

const ALL_QUESTION_IDS = ["Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"];

function defaultTargetFor(qid) {
  if (qid === "Q1A") return "Q2A";
  if (qid === "Q2A") return "Q2B";
  if (qid === "Q2B") return "Q3A";
  if (qid === "Q3A") return "Q3B";
  if (qid === "Q1B") return "Q2A";
  return "Q1A";
}

export default function ReviewGrade({ examId, refreshKey, onCalibrated, initialStudentId, onGoToLedger }) {
  const [students, setStudents] = useState([]);
  const [selected, setSelected] = useState(initialStudentId || "");
  const [booklet, setBooklet] = useState(null);
  const [rows, setRows] = useState([]);
  const [ia, setIa] = useState(null);
  const [drafts, setDrafts] = useState({});
  const [activePageIdx, setActivePageIdx] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // Per-card interactive states
  const [cardTexts, setCardTexts] = useState({});
  const [cursorPositions, setCursorPositions] = useState({});
  const [splitTargets, setSplitTargets] = useState({});

  // Loading states
  const [isEvaluatingAll, setIsEvaluatingAll] = useState(false);
  const [evaluatingCard, setEvaluatingCard] = useState({});
  const [splittingCard, setSplittingCard] = useState({});
  const [deletingCard, setDeletingCard] = useState({});
  const [savingCard, setSavingCard] = useState({});

  const load = async (studentId) => {
    if (!examId) return;
    try {
      const { data: subData } = await getSubmissions(examId, studentId || undefined);
      const studentList = subData.students || [];
      setStudents(studentList);
      const pick = studentId || initialStudentId || selected || studentList[0] || "";
      setSelected(pick);

      if (pick) {
        try {
          const { data: bData } = await getBooklet(examId, pick);
          setBooklet(bData.booklet || null);
          const subs = bData.submissions || [];
          setRows(subs);
          if (bData.ia_breakdown) {
            setIa(bData.ia_breakdown);
          }
          const nextDrafts = {};
          const nextTexts = {};
          subs.forEach((s) => {
            nextDrafts[s.question_id] = {
              teacher_score: s.teacher_score ?? s.ai_score ?? 0,
              feedback: s.ai_feedback || "",
            };
            nextTexts[s.question_id] = s.transcribed_text || "";
          });
          setDrafts(nextDrafts);
          setCardTexts(nextTexts);
        } catch {
          const filtered = (subData.submissions || []).filter((s) => s.student_id === pick);
          setRows(filtered);
          const nextDrafts = {};
          const nextTexts = {};
          filtered.forEach((s) => {
            nextDrafts[s.question_id] = {
              teacher_score: s.teacher_score ?? s.ai_score ?? 0,
              feedback: s.ai_feedback || "",
            };
            nextTexts[s.question_id] = s.transcribed_text || "";
          });
          setDrafts(nextDrafts);
          setCardTexts(nextTexts);
        }
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
    if (booklet?.image_urls && booklet.image_urls.length > 0) {
      return booklet.image_urls;
    }
    const urls = rows[0]?.image_urls || [];
    return urls;
  }, [booklet, rows]);

  const handleCursorChange = (qid, pos) => {
    setCursorPositions((prev) => ({ ...prev, [qid]: pos }));
  };

  // Step 2: Teacher Cursor Split or Reassign
  const handleSplitCard = async (qid) => {
    const pos = cursorPositions[qid] ?? 0;
    const targetQid = splitTargets[qid] || defaultTargetFor(qid);
    const textToSplit = cardTexts[qid] ?? "";

    if (pos < 0) return;
    setError("");
    setMessage("");
    setSplittingCard((prev) => ({ ...prev, [qid]: true }));

    try {
      const { data } = await splitSubmission({
        exam_id: examId,
        student_id: selected,
        source_question_id: qid,
        target_question_id: targetQid,
        split_index: pos,
        current_text: textToSplit,
      });

      setRows(data.submissions || []);
      if (data.ia_breakdown) {
        setIa(data.ia_breakdown);
      }

      const nextDrafts = { ...drafts };
      const nextTexts = { ...cardTexts };
      (data.submissions || []).forEach((s) => {
        nextDrafts[s.question_id] = {
          teacher_score: s.teacher_score ?? s.ai_score ?? 0,
          feedback: s.ai_feedback || "",
        };
        nextTexts[s.question_id] = s.transcribed_text || "";
      });
      setDrafts(nextDrafts);
      setCardTexts(nextTexts);

      if (pos === 0) {
        setMessage(
          `Reassigned entire text of Question ${qid} to Question ${targetQid}! Empty ${qid} card removed.`
        );
      } else {
        setMessage(
          `Split Question ${qid} at character ${pos}! Second half moved to Question ${targetQid}.`
        );
      }
      onCalibrated?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setSplittingCard((prev) => ({ ...prev, [qid]: false }));
    }
  };

  // Step 3: Evaluate All Assigned Questions Against Rubrics
  const handleEvaluateAll = async () => {
    if (!selected) return;
    setError("");
    setMessage("");
    setIsEvaluatingAll(true);
    try {
      const { data } = await evaluateStudent({
        exam_id: examId,
        student_id: selected,
        question_id: null,
      });
      setRows(data.submissions || []);
      if (data.ia_breakdown) {
        setIa(data.ia_breakdown);
      }
      const nextDrafts = { ...drafts };
      (data.submissions || []).forEach((s) => {
        nextDrafts[s.question_id] = {
          teacher_score: s.teacher_score ?? s.ai_score ?? 0,
          feedback: s.ai_feedback || "",
        };
      });
      setDrafts(nextDrafts);
      setMessage(
        `⚡ Successfully evaluated all ${data.submissions.length} question answers against active rubrics!`
      );
      onCalibrated?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setIsEvaluatingAll(false);
    }
  };

  // Evaluate single question card on demand
  const handleEvaluateSingleCard = async (qid) => {
    setError("");
    setMessage("");
    setEvaluatingCard((prev) => ({ ...prev, [qid]: true }));
    try {
      const { data } = await evaluateStudent({
        exam_id: examId,
        student_id: selected,
        question_id: qid,
      });
      setRows(data.submissions || []);
      if (data.ia_breakdown) {
        setIa(data.ia_breakdown);
      }
      const nextDrafts = { ...drafts };
      (data.submissions || []).forEach((s) => {
        nextDrafts[s.question_id] = {
          teacher_score: s.teacher_score ?? s.ai_score ?? 0,
          feedback: s.ai_feedback || "",
        };
      });
      setDrafts(nextDrafts);
      setMessage(`Evaluated Question ${qid} against its rubric criteria!`);
      onCalibrated?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setEvaluatingCard((prev) => ({ ...prev, [qid]: false }));
    }
  };

  // Delete card cleanly
  const handleDeleteCard = async (qid) => {
    if (!window.confirm(`Delete question card ${qid}?`)) return;
    setError("");
    setMessage("");
    setDeletingCard((prev) => ({ ...prev, [qid]: true }));
    try {
      const { data } = await deleteSubmission(examId, selected, qid);
      setRows(data.submissions || []);
      if (data.ia_breakdown) {
        setIa(data.ia_breakdown);
      }
      setMessage(`Deleted Question ${qid} card. SDMCET IA total updated.`);
      onCalibrated?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setDeletingCard((prev) => ({ ...prev, [qid]: false }));
    }
  };

  // Teacher verify / approve score & calibrate rubric weights
  const approve = async (qid) => {
    setError("");
    setMessage("");
    setSavingCard((prev) => ({ ...prev, [qid]: true }));
    try {
      await verifyAnswer({
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
      setSavingCard((prev) => ({ ...prev, [qid]: false }));
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
        No evaluated booklets found for this exam. Upload or grade a student booklet in <b>02. Batch Grading</b> to view interactive mapping.
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
            Showing {rows.length} question cards across {images.length} handwritten pages.
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
          <div className="flex items-center gap-4">
            {onGoToLedger && (
              <button
                type="button"
                onClick={onGoToLedger}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition-colors"
              >
                <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-600" />
                View Marks Ledger
              </button>
            )}
            <div className="text-right">
              <span className="text-xs font-semibold text-slate-500 block uppercase tracking-wide">Final IA Marks</span>
              <span className="text-3xl font-extrabold text-indigo-600 tracking-tight">
                {ia?.ia_total ?? 0} <span className="text-sm font-semibold text-slate-400">/ 20</span>
              </span>
            </div>
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

      {/* Main Two-Column Layout */}
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
              <div className="overflow-auto max-h-[700px] border border-slate-200 rounded-xl bg-slate-100 p-2 flex justify-center">
                <img
                  src={images[activePageIdx]}
                  alt={`Student Answer Page ${activePageIdx + 1}`}
                  style={{ width: `${zoomLevel}%` }}
                  className="rounded-lg shadow-sm border border-slate-300 transition-all bg-white"
                />
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500 py-8 text-center">No booklet images found.</p>
          )}
        </aside>

        {/* RIGHT COLUMN: Question Cards & On-Demand Rubric Evaluator */}
        <div className="xl:col-span-7 space-y-4">
          {/* Step 3: Global Evaluate All Assigned Questions Button */}
          <div className="surface-card p-4 bg-gradient-to-r from-indigo-900 to-slate-900 text-white rounded-xl shadow-md flex flex-wrap items-center justify-between gap-3 border border-indigo-700">
            <div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                <h4 className="font-bold text-sm text-white">
                  Step 3: Automated Rubric Evaluation
                </h4>
              </div>
              <p className="text-xs text-slate-300 mt-0.5">
                Evaluates all assigned question cards against atomic rubrics using EduEval (MiniLM-L6 + Fuzzy Keyword).
              </p>
            </div>
            <button
              type="button"
              disabled={isEvaluatingAll || rows.length === 0}
              onClick={handleEvaluateAll}
              className="bg-amber-400 hover:bg-amber-500 text-slate-950 font-extrabold text-xs px-4 py-2.5 rounded-lg inline-flex items-center gap-2 shadow transition-all disabled:opacity-50"
            >
              {isEvaluatingAll ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
                  Evaluating All Cards...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 text-slate-950 fill-slate-950" />
                  ⚡ Evaluate All Answers Against Rubrics
                </>
              )}
            </button>
          </div>

          <div className="flex items-center justify-between pt-1">
            <h4 className="font-bold text-slate-900 text-sm">
              Question Cards ({rows.length}) &amp; Cursor Splitter
            </h4>
            <span className="text-xs text-slate-500">
              Click textarea to place split cursor (Char 0 reassigns card)
            </span>
          </div>

          {/* Question Cards List */}
          {rows.map((sub) => {
            const isVerified = Boolean(sub.teacher_verified);
            const isSplitting = splittingCard[sub.question_id];
            const isEvaluating = evaluatingCard[sub.question_id];
            const isDeleting = deletingCard[sub.question_id];
            const isSaving = savingCard[sub.question_id];

            const pos = cursorPositions[sub.question_id] ?? 0;
            const targetQid = splitTargets[sub.question_id] || defaultTargetFor(sub.question_id);
            const textLength = cardTexts[sub.question_id]?.length || 0;

            return (
              <article key={sub.question_id} className="surface-card space-y-3 border border-slate-200">
                {/* Header row */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <span className="font-extrabold text-base text-slate-900 tracking-tight">
                      Question {sub.question_id}
                    </span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                      Max: 5 Marks
                    </span>
                  </div>

                  <div className="flex items-center gap-2 flex-wrap">
                    {/* On-demand single card evaluation button */}
                    <button
                      type="button"
                      disabled={isEvaluating}
                      onClick={() => handleEvaluateSingleCard(sub.question_id)}
                      className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 inline-flex items-center gap-1 transition-colors"
                      title={`Re-evaluate Question ${sub.question_id}`}
                    >
                      {isEvaluating ? (
                        <RefreshCw className="w-3 h-3 animate-spin" />
                      ) : (
                        <Search className="w-3 h-3" />
                      )}
                      <span>Evaluate This Question</span>
                    </button>

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

                    {/* Delete card */}
                    <button
                      type="button"
                      disabled={isDeleting}
                      onClick={() => handleDeleteCard(sub.question_id)}
                      className="p-1 rounded hover:bg-red-50 text-slate-400 hover:text-red-600 transition-colors"
                      title={`Delete Question ${sub.question_id}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Editable Textarea */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                      Transcribed Text ({textLength} characters)
                    </label>
                    <span className="text-[11px] text-slate-400 font-mono">
                      Click anywhere to place split cursor
                    </span>
                  </div>
                  <textarea
                    rows={8}
                    value={cardTexts[sub.question_id] ?? ""}
                    onChange={(e) =>
                      setCardTexts((prev) => ({ ...prev, [sub.question_id]: e.target.value }))
                    }
                    onSelect={(e) => handleCursorChange(sub.question_id, e.target.selectionStart)}
                    onClick={(e) => handleCursorChange(sub.question_id, e.target.selectionStart)}
                    onKeyUp={(e) => handleCursorChange(sub.question_id, e.target.selectionStart)}
                    placeholder="Transcribed student text..."
                    className="w-full text-xs font-mono leading-relaxed p-3 rounded-lg border border-slate-300 bg-slate-50/50 text-slate-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                {/* Cursor Split Toolbar */}
                <div className="p-3 bg-slate-100 rounded-xl border border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <Scissors className="w-4 h-4 text-indigo-600 shrink-0" />
                    <span className="font-mono text-slate-700">
                      Cursor: <b>Char {pos}</b> / {textLength}
                    </span>
                    {pos === 0 && (
                      <span className="text-[11px] font-bold text-amber-700 bg-amber-100 px-2 py-0.5 rounded-full">
                        Start of text (Reassigns card)
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-slate-600 font-medium">Target:</span>
                    <select
                      value={targetQid}
                      onChange={(e) =>
                        setSplitTargets((prev) => ({ ...prev, [sub.question_id]: e.target.value }))
                      }
                      className="bg-white border border-slate-300 text-xs font-bold rounded-lg px-2.5 py-1.5 text-slate-900"
                    >
                      {ALL_QUESTION_IDS.filter((q) => q !== sub.question_id).map((qid) => (
                        <option key={qid} value={qid}>
                          Question {qid}
                        </option>
                      ))}
                    </select>

                    <button
                      type="button"
                      disabled={isSplitting}
                      onClick={() => handleSplitCard(sub.question_id)}
                      className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg inline-flex items-center gap-1.5 shadow-sm transition-colors disabled:opacity-50"
                    >
                      {isSplitting ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      ) : pos === 0 ? (
                        <CornerDownRight className="w-3.5 h-3.5" />
                      ) : (
                        <Scissors className="w-3.5 h-3.5" />
                      )}
                      <span>
                        {pos === 0
                          ? `↪️ Reassign Entire Text to ${targetQid}`
                          : `✂️ Split from Char ${pos} -> Assign to ${targetQid}`}
                      </span>
                    </button>
                  </div>
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
