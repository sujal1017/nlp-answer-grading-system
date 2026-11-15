import { useEffect, useState } from "react";
import { Plus, Trash2, Sparkles, Save, BookOpen, Scale, AlertCircle, CheckCircle } from "lucide-react";
import { parseQuestions, updateRubric, upsertExam } from "../services/api";

const SDMCET_SAMPLE_TEXT = `Q1 (a) Define research and explain its major objectives. 5 Marks CO-1
Q1 (b) Describe the systematic procedure involved in reviewing the literature. 5 Marks CO-2
Q2 (a) Classify and explain the different types of research with examples. 5 Marks CO-1
Q2 (b) Discuss the essential features of a good research design. 5 Marks CO-2
Q3 (a) Explain the various techniques used for defining a research problem. 5 Marks CO-3
Q3 (b) Discuss the important concepts associated with research design. 5 Marks CO-4`;

const emptyQuestion = (id = "Q1A") => ({
  question_id: id,
  question_text: "",
  max_marks: 5,
  key_points: [
    { point: "Core conceptual definition and explanation", weight: 1.25 },
    { point: "Key characteristics and distinguishing components", weight: 1.25 },
    { point: "Systematic methodology or procedure", weight: 1.25 },
    { point: "Academic significance and concrete examples", weight: 1.25 },
  ],
});

export default function QuestionStudio({ exam, onSaved }) {
  const [examId, setExamId] = useState(exam?.exam_id || "IA1_RM_2026");
  const [title, setTitle] = useState(exam?.exam_title || "22URMK506 – Research Methodology & IPR");
  const [paste, setPaste] = useState("");
  const [questions, setQuestions] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (exam?.exam_id) setExamId(exam.exam_id);
    if (exam?.exam_title) setTitle(exam.exam_title);
    if (exam?.rubrics?.length) {
      setQuestions(
        exam.rubrics.map((r) => ({
          question_id: r.question_id,
          question_text: r.question_text,
          max_marks: r.max_marks,
          key_points: r.key_points || [],
        }))
      );
    }
  }, [exam]);

  const updateQ = (idx, patch) => {
    setQuestions((prev) => prev.map((q, i) => (i === idx ? { ...q, ...patch } : q)));
  };

  const handleParse = async () => {
    setError("");
    setMessage("");
    setBusy(true);
    try {
      const { data } = await parseQuestions(paste);
      if (!data.questions?.length) {
        setError("Could not parse questions. Use Q1 (a) ... 5 Marks, or Q1A | prompt | 5.");
      } else {
        setQuestions(data.questions);
        setMessage(`Parsed ${data.questions.length} questions and generated atomic rubrics with calibrated weights.`);
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleLoadSample = () => {
    setExamId("IA1_RM_2026");
    setTitle("22URMK506 – Research Methodology & IPR");
    setPaste(SDMCET_SAMPLE_TEXT);
    setMessage("Loaded SDMCET IA-1 Question Paper template. Click 'Parse & generate rubrics' to build criteria.");
  };

  const handleSaveAll = async () => {
    setError("");
    setMessage("");
    if (!questions.length) {
      setError("Add or paste at least one question.");
      return;
    }
    setBusy(true);
    try {
      const { data } = await upsertExam({
        exam_id: examId,
        exam_title: title,
        questions,
        replace_rubrics: true,
      });
      setQuestions(data.questions);
      setMessage(`Successfully saved exam ${data.exam_id} with ${data.questions.length} weighted rubrics.`);
      onSaved?.(data.exam_id);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleSaveOne = async (q) => {
    setError("");
    setBusy(true);
    try {
      await updateRubric(examId, q.question_id, {
        question_text: q.question_text,
        max_marks: q.max_marks,
        key_points: q.key_points,
      });
      setMessage(`Updated question ${q.question_id} rubric.`);
      onSaved?.(examId);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const setPoint = (qIdx, pIdx, patch) => {
    const next = questions[qIdx].key_points.map((p, i) => (i === pIdx ? { ...p, ...patch } : p));
    updateQ(qIdx, { key_points: next });
  };

  const addPoint = (qIdx) => {
    const q = questions[qIdx];
    const nextPoints = [...(q.key_points || []), { point: "Additional rubric criterion", weight: 1.0 }];
    updateQ(qIdx, { key_points: nextPoints });
  };

  const removePoint = (qIdx, pIdx) => {
    const nextPoints = questions[qIdx].key_points.filter((_, i) => i !== pIdx);
    updateQ(qIdx, { key_points: nextPoints });
  };

  const rebalanceWeights = (qIdx) => {
    const q = questions[qIdx];
    const n = Math.max(q.key_points.length, 1);
    const base = Math.round((q.max_marks / n) * 100) / 100;
    const balanced = q.key_points.map((pt, i) => ({
      ...pt,
      weight: i === 0 ? Math.round((q.max_marks - base * (n - 1)) * 100) / 100 : base,
    }));
    updateQ(qIdx, { key_points: balanced });
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
      {/* LEFT: Setup and Input */}
      <section className="xl:col-span-5 surface-card">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-slate-900">Teacher Question Studio</h3>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
            No OCR on Question Paper
          </span>
        </div>
        <p className="text-sm text-slate-500 mt-1">
          Paste questions directly with marks. The backend automatically constructs atomic Bloom criteria with calibrated weights.
        </p>

        <div className="mt-4 space-y-3">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500">Exam Identifier</label>
            <input
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold"
              value={examId}
              onChange={(e) => setExamId(e.target.value.toUpperCase())}
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500">Subject / Course Title</label>
            <input
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500">Paste Questions &amp; Marks</label>
              <button
                type="button"
                onClick={handleLoadSample}
                className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 inline-flex items-center gap-1"
              >
                <BookOpen className="w-3.5 h-3.5" />
                Load SDMCET IA-1 Sample
              </button>
            </div>
            <textarea
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono min-h-[190px] leading-relaxed"
              placeholder={SDMCET_SAMPLE_TEXT}
              value={paste}
              onChange={(e) => setPaste(e.target.value)}
            />
          </div>
        </div>

        <div className="flex gap-2 mt-4">
          <button
            type="button"
            onClick={handleParse}
            disabled={busy || !paste.trim()}
            className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-600 text-white font-semibold py-2.5 text-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-sm"
          >
            <Sparkles className="w-4 h-4" />
            Parse &amp; generate rubrics
          </button>
          <button
            type="button"
            onClick={() => setQuestions((q) => [...q, emptyQuestion(`Q${q.length + 1}A`)])}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 px-3.5 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            <Plus className="w-4 h-4" />
            Add Row
          </button>
        </div>

        <button
          type="button"
          onClick={handleSaveAll}
          disabled={busy || !questions.length}
          className="mt-3 w-full inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 text-white font-semibold py-2.5 text-sm hover:bg-slate-800 disabled:opacity-50 transition-colors shadow-sm"
        >
          <Save className="w-4 h-4" />
          Save exam &amp; calibrated rubrics ({questions.length} Questions)
        </button>

        {message && (
          <p className="mt-3 text-sm text-emerald-800 bg-emerald-50 border border-emerald-200 rounded-lg p-3">
            {message}
          </p>
        )}
        {error && (
          <p className="mt-3 text-sm text-red-800 bg-red-50 border border-red-200 rounded-lg p-3">
            {error}
          </p>
        )}
      </section>

      {/* RIGHT: Rubric Review and Fine-Tuning */}
      <section className="xl:col-span-7 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-slate-900">
            Atomic Rubrics &amp; Calibrated Weights ({questions.length})
          </h3>
          <span className="text-xs text-slate-500 font-medium">
            Weights auto-balance to match Max Marks
          </span>
        </div>

        {!questions.length && (
          <div className="surface-card text-center py-12 text-slate-500 text-sm">
            No questions configured yet. Paste question paper text on the left or click <b>Load SDMCET IA-1 Sample</b>.
          </div>
        )}

        {questions.map((q, idx) => {
          const totalWeight = Math.round(
            (q.key_points || []).reduce((sum, pt) => sum + (Number(pt.weight) || 0), 0) * 100
          ) / 100;
          const isBalanced = Math.abs(totalWeight - Number(q.max_marks)) < 0.05;

          return (
            <article key={`${q.question_id}-${idx}`} className="surface-card">
              <div className="flex flex-wrap gap-2 items-center justify-between pb-3 border-b border-slate-100">
                <div className="flex gap-2 items-center">
                  <input
                    className="w-20 rounded-lg border border-slate-300 px-2.5 py-1 text-sm font-bold text-indigo-700 uppercase"
                    value={q.question_id}
                    onChange={(e) => updateQ(idx, { question_id: e.target.value.toUpperCase() })}
                  />
                  <div className="flex items-center gap-1">
                    <input
                      type="number"
                      min={1}
                      max={20}
                      className="w-16 rounded-lg border border-slate-300 px-2 py-1 text-sm text-center font-semibold"
                      value={q.max_marks}
                      onChange={(e) => updateQ(idx, { max_marks: Number(e.target.value) })}
                    />
                    <span className="text-xs font-semibold text-slate-500">Marks</span>
                  </div>

                  <span
                    className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full ${
                      isBalanced
                        ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        : "bg-amber-50 text-amber-700 border border-amber-200"
                    }`}
                  >
                    {isBalanced ? <CheckCircle className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                    Total: {totalWeight} / {q.max_marks}M
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => rebalanceWeights(idx)}
                    title="Evenly balance point weights"
                    className="text-xs font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 border border-slate-200 rounded-lg px-2.5 py-1 inline-flex items-center gap-1"
                  >
                    <Scale className="w-3.5 h-3.5" />
                    Balance
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSaveOne(q)}
                    className="text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg px-2.5 py-1"
                  >
                    Save {q.question_id}
                  </button>
                  <button
                    type="button"
                    onClick={() => setQuestions((prev) => prev.filter((_, i) => i !== idx))}
                    className="text-slate-400 hover:text-red-600 p-1"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="mt-3">
                <label className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Question Prompt</label>
                <textarea
                  className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm min-h-[60px]"
                  value={q.question_text}
                  onChange={(e) => updateQ(idx, { question_text: e.target.value })}
                />
              </div>

              <div className="mt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                    Atomic Rubric Key Points &amp; Weights
                  </span>
                  <button
                    type="button"
                    onClick={() => addPoint(idx)}
                    className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold inline-flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" />
                    Add Criterion
                  </button>
                </div>

                <div className="space-y-2">
                  {(q.key_points || []).map((pt, pIdx) => (
                    <div key={pIdx} className="flex items-center gap-2">
                      <span className="text-xs font-mono font-semibold text-slate-400 w-5">{pIdx + 1}.</span>
                      <input
                        className="flex-1 rounded-lg border border-slate-200 px-3 py-1.5 text-sm"
                        value={pt.point || ""}
                        onChange={(e) => setPoint(idx, pIdx, { point: e.target.value })}
                        placeholder="Evaluation criterion..."
                      />
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          step="0.25"
                          min="0"
                          max={q.max_marks}
                          className="w-16 rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-center font-semibold"
                          value={pt.weight ?? 0}
                          onChange={(e) => setPoint(idx, pIdx, { weight: Number(e.target.value) })}
                        />
                        <span className="text-xs text-slate-400">M</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => removePoint(idx, pIdx)}
                        className="text-slate-400 hover:text-red-500 p-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </article>
          );
        })}
      </section>
    </div>
  );
}
