import { useEffect, useState } from "react";
import {
  Rocket,
  Upload,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Layers,
  Plus,
  Trash2,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import {
  getExamRubrics,
  getJob,
  getSampleBooklets,
  gradeBooklet,
  gradeManualAnswers,
  gradeSampleBooklet,
} from "../services/api";

export default function BatchGrading({ examId, onFinished, onGoToReview }) {
  const [usn, setUsn] = useState("2SD23CS082");
  const [files, setFiles] = useState([]);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState("test1.pdf");
  const [maxPages, setMaxPages] = useState(0); // 0 = all
  const [useSample, setUseSample] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState(null);
  const [rubrics, setRubrics] = useState([]);
  const [manualAnswers, setManualAnswers] = useState([]);
  const [manualOpen, setManualOpen] = useState(false);
  const [manualBusy, setManualBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getSampleBooklets()
      .then(({ data }) => {
        setSamples(data.samples || []);
        if (data.samples?.length) {
          setSelectedSample(data.samples[0].filename);
        }
      })
      .catch(() => {});
  }, []);

  const poll = async (jobId) => {
    for (;;) {
      const { data } = await getJob(jobId);
      setProgress(data.progress || data.status);
      if (data.status === "done") return data;
      if (data.status === "error") throw new Error(data.error || "Grading failed");
      await new Promise((r) => setTimeout(r, 1200));
    }
  };

  const run = async () => {
    setError("");
    setResult(null);
    if (!examId) {
      setError("Select or create an exam in Question Studio first.");
      return;
    }
    const cleanUsn = usn.trim().toUpperCase();
    if (!cleanUsn) {
      setError("Please provide a student USN.");
      return;
    }

    setBusy(true);
    setProgress("Initiating booklet processing...");

    try {
      let jobId = null;
      if (useSample) {
        setProgress(`Loading server sample '${selectedSample}'...`);
        const { data } = await gradeSampleBooklet(examId, {
          student_id: cleanUsn,
          sample_name: selectedSample,
          max_pages: maxPages > 0 ? maxPages : null,
        });
        jobId = data.job_id;
        setProgress(`Queued ${data.pages} pages for TrOCR and BERT`);
      } else {
        if (!files.length) {
          throw new Error("Select at least one PDF or image file.");
        }
        setProgress("Uploading booklet...");
        const { data } = await gradeBooklet(examId, cleanUsn, files);
        jobId = data.job_id;
        setProgress(`Queued ${data.pages} pages`);
      }

      const job = await poll(jobId);
      setResult(job);
      if (job.transcript) {
        const { data: rubricData } = await getExamRubrics(examId);
        setRubrics(rubricData || []);
        const detected = (job.segments || []).map((segment, index) => ({
          id: `${Date.now()}-${index}`,
          question_id: segment.mapped ? segment.question_id : "",
          answer: segment.answer || "",
        }));
        setManualAnswers(
          detected.length
            ? detected
            : [{ id: `${Date.now()}-0`, question_id: "", answer: job.transcript || "" }],
        );
        setManualOpen(Boolean(job.needs_review));
      }
      onFinished?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const updateManualAnswer = (id, patch) => {
    setManualAnswers((current) =>
      current.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  };

  const moveManualAnswer = (index, direction) => {
    const target = index + direction;
    if (target < 0 || target >= manualAnswers.length) return;
    setManualAnswers((current) => {
      const updated = [...current];
      [updated[index], updated[target]] = [updated[target], updated[index]];
      return updated;
    });
  };

  const submitManualAnswers = async () => {
    setError("");
    const answers = manualAnswers.map(({ question_id, answer }) => ({
      question_id: question_id.trim().toUpperCase(),
      answer: answer.trim(),
    }));
    if (!answers.length || answers.some((item) => !item.question_id || !item.answer)) {
      setError("Assign a question and enter non-empty text for every answer, or remove unused answer fields.");
      return;
    }
    if (new Set(answers.map((item) => item.question_id)).size !== answers.length) {
      setError("Each question can be assigned once. Combine text for the same question into one answer field.");
      return;
    }

    setManualBusy(true);
    try {
      const { data } = await gradeManualAnswers(examId, {
        exam_id: examId,
        student_id: result.student_id,
        answers,
      });
      if (data.answers_evaluated_successfully !== data.answers_submitted) {
        throw new Error(
          `Only ${data.answers_evaluated_successfully} of ${data.answers_submitted} submitted answers were evaluated. Results were not finalized.`,
        );
      }
      setResult((current) => ({
        ...current,
        needs_review: false,
        results: data.submissions,
        chunks: data.submissions.map((item) => item.question_id),
        total_score: data.total_score,
        max_possible_score: data.max_possible_score,
        ia: data.ia,
        message: `Manual review complete. All ${data.answers_evaluated_successfully} submitted answers were evaluated.`,
        diagnostics: {
          ...current.diagnostics,
          answers_mapped_to_rubrics: data.answers_passed_to_grader,
          answers_passed_to_grader: data.answers_passed_to_grader,
          answers_evaluated_successfully: data.answers_evaluated_successfully,
          answers_submitted_manually: data.answers_submitted,
          skipped: [],
        },
      }));
      setManualOpen(false);
      onFinished?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setManualBusy(false);
    }
  };

  return (
    <div className="surface-card max-w-4xl mx-auto">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <h3 className="text-lg font-bold text-slate-900">Batch Student Booklet Digitization &amp; Grading</h3>
          <p className="text-sm text-slate-500 mt-0.5">
            Target Exam: <span className="font-mono font-bold text-indigo-700">{examId || "—"}</span>. OCR and BERT run locally on CUDA when available, otherwise CPU.
          </p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
          TrOCR OCR · BERT grading
        </span>
      </div>

      <div className="mt-5 space-y-4">
        {/* USN Input */}
        <div>
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Student USN</label>
          <input
            className="mt-1 w-full max-w-sm rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold tracking-wide"
            value={usn}
            onChange={(e) => setUsn(e.target.value.toUpperCase())}
            placeholder="e.g. 2SD23CS082"
          />
        </div>

        {/* Source Mode Selector */}
        <div>
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-500 block mb-2">Booklet Source</label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setUseSample(true)}
              className={`p-3.5 rounded-xl border text-left transition-all ${
                useSample
                  ? "border-indigo-600 bg-indigo-50/60 ring-2 ring-indigo-600/20"
                  : "border-slate-200 hover:border-slate-300 bg-white"
              }`}
            >
              <div className="flex items-center gap-2">
                <Layers className={`w-4 h-4 ${useSample ? "text-indigo-600" : "text-slate-500"}`} />
                <span className="text-sm font-bold text-slate-800">Use Pre-loaded Test Booklet</span>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Instant test using SDMCET answer booklet stored on the system ({samples.length} files available)
              </p>
            </button>

            <button
              type="button"
              onClick={() => setUseSample(false)}
              className={`p-3.5 rounded-xl border text-left transition-all ${
                !useSample
                  ? "border-indigo-600 bg-indigo-50/60 ring-2 ring-indigo-600/20"
                  : "border-slate-200 hover:border-slate-300 bg-white"
              }`}
            >
              <div className="flex items-center gap-2">
                <Upload className={`w-4 h-4 ${!useSample ? "text-indigo-600" : "text-slate-500"}`} />
                <span className="text-sm font-bold text-slate-800">Upload New PDF / Images</span>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Choose a scanned student booklet PDF or individual handwritten page photos
              </p>
            </button>
          </div>
        </div>

        {/* Configuration based on selection */}
        {useSample ? (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-semibold text-slate-600 block mb-1">Select Sample PDF</label>
              <select
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white"
                value={selectedSample}
                onChange={(e) => setSelectedSample(e.target.value)}
              >
                {samples.map((s) => (
                  <option key={s.filename} value={s.filename}>
                    {s.filename} ({s.size_mb} MB) — {s.description}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-600 block mb-1">Pages to Grade</label>
              <select
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white"
                value={maxPages}
                onChange={(e) => setMaxPages(Number(e.target.value))}
              >
                <option value={0}>All pages (Full booklet)</option>
                <option value={2}>First 2 pages (Fastest test)</option>
                <option value={4}>First 4 pages (Standard test)</option>
              </select>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-lg p-6 cursor-pointer hover:bg-white transition-colors">
              <Upload className="w-8 h-8 text-slate-400 mb-2" />
              <span className="text-sm font-semibold text-slate-700">
                {files.length ? `${files.length} file(s) selected` : "Drop PDF or page photos here, or browse"}
              </span>
              <span className="text-xs text-slate-400 mt-1">Accepts .pdf, .png, .jpg, .jpeg</span>
              <input
                type="file"
                multiple
                accept=".pdf,image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
            </label>
          </div>
        )}

        {/* Action Button */}
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 text-sm disabled:opacity-50 transition-all shadow-md"
        >
          <Rocket className="w-4 h-4" />
          {busy ? progress || "Processing with TrOCR and BERT..." : "Run TrOCR OCR & BERT Grading"}
        </button>

        {/* Live Progress Banner */}
        {busy && (
          <div className="rounded-xl border border-indigo-200 bg-indigo-50/70 p-4 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm font-semibold text-indigo-900">{progress}</p>
            </div>
            <p className="text-xs text-indigo-700 mt-1 ml-7">
              OCR and grading status is reported separately. Models load on first use and are released after grading.
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-red-900">Grading Error</p>
              <p className="text-xs text-red-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Evaluation Result Summary */}
        {result?.status === "done" && (
          <div className={`rounded-xl border p-5 space-y-3 ${
            result.needs_review ? "border-amber-200 bg-amber-50" : "border-emerald-200 bg-emerald-50"
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {result.needs_review ? (
                  <AlertTriangle className="w-5 h-5 text-amber-600" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                )}
                <p className="font-bold text-slate-950">
                  {result.needs_review ? "OCR needs review" : `BERT grading complete for ${result.student_id}`}
                </p>
              </div>
              <span className="text-xs font-semibold bg-white/70 text-slate-800 px-3 py-1 rounded-full">
                {result.diagnostics?.answers_detected_by_segmentation ?? result.chunks?.length ?? 0} Answer Sections
              </span>
            </div>

            <p className="text-sm text-slate-800">{result.progress}</p>
            {result.message && <p className="text-sm text-slate-700">{result.message}</p>}

            {result.diagnostics && (
              <div className="grid grid-cols-2 md:grid-cols-6 gap-2 rounded-lg border border-slate-200 bg-white p-3 text-center text-xs">
                <div><span className="block text-slate-500">OCR text lines</span><b>{result.diagnostics.ocr_nonempty_lines ?? 0}</b></div>
                <div><span className="block text-slate-500">Answer sections detected</span><b>{result.diagnostics.answers_detected_by_segmentation ?? 0}</b></div>
                <div><span className="block text-slate-500">Mapped to rubrics</span><b>{result.diagnostics.answers_mapped_to_rubrics ?? 0}</b></div>
                <div><span className="block text-slate-500">Passed to grader</span><b>{result.diagnostics.answers_passed_to_grader ?? 0}</b></div>
                <div><span className="block text-slate-500">Evaluated successfully</span><b>{result.diagnostics.answers_evaluated_successfully ?? 0}</b></div>
                <div><span className="block text-slate-500">Submitted manually</span><b>{result.diagnostics.answers_submitted_manually ?? "—"}</b></div>
              </div>
            )}
            {!!result.diagnostics?.detected_question_ids?.length && (
              <p className="text-xs text-slate-600">
                Detected question numbers: {result.diagnostics.detected_question_ids.join(", ")}.
              </p>
            )}

            {!!result.diagnostics?.missing_question_ids?.length && (
              <p className="text-xs text-slate-600">
                No answer section was detected for: {result.diagnostics.missing_question_ids.join(", ")}.
              </p>
            )}
            {!!result.diagnostics?.skipped?.length && (
              <div className="rounded-lg border border-amber-200 bg-white p-3 text-xs text-amber-900">
                <b>Sections requiring review</b>
                <ul className="mt-1 list-disc pl-5">
                  {result.diagnostics.skipped.map((item, index) => (
                    <li key={`${item.question_id || "section"}-${index}`}>
                      {item.question_id || "Answer section"}: {item.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.transcript && (
              <details className="rounded-lg border border-slate-200 bg-white p-3" open>
                <summary className="cursor-pointer text-sm font-semibold text-slate-800">
                  Extracted OCR text
                </summary>
                <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-slate-700">
                  {result.transcript}
                </pre>
              </details>
            )}

            {!!result.results?.length && (
              <div className="space-y-2">
                <div className="rounded-lg border border-emerald-200 bg-white p-3 text-sm font-bold text-emerald-900">
                  Total score: {result.total_score ?? result.results.reduce((sum, item) => sum + Number(item.ai_score || 0), 0)}
                  {" / "}{result.max_possible_score ?? "—"}
                </div>
                {result.results.map((submission) => (
                  <article key={submission.question_id} className="rounded-lg border border-slate-200 bg-white p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                      <b>{submission.question_id}</b>
                      <span>
                        BERT marks: <b>{submission.ai_score ?? 0} / {submission.max_marks ?? "—"}</b>
                      </span>
                      <span>Similarity: {Math.round((submission.similarity_score ?? 0) * 100)}%</span>
                      <span>Keyword coverage: {Math.round((submission.keyword_coverage ?? 0) * 100)}%</span>
                    </div>
                    {submission.ai_feedback && (
                      <p className="mt-2 text-xs text-slate-600">{submission.ai_feedback}</p>
                    )}
                  </article>
                ))}
              </div>
            )}

            {result.transcript && (
              <section className="space-y-3 rounded-xl border border-amber-300 bg-white p-4">
                <div>
                  <h4 className="font-bold text-slate-900">Manual answer separation</h4>
                  <p className="mt-1 text-xs text-slate-600">
                    {result.needs_review
                      ? "Check every detected section, edit or split the OCR text, and assign each answer to its rubric question."
                      : "If the transcript contains an answer that automatic segmentation missed, open this workflow to add or correct answer sections."}
                    {" "}Submitting replaces the automatic results; each non-empty field is graded separately.
                  </p>
                </div>
                {!result.needs_review && (
                  <button
                    type="button"
                    onClick={() => setManualOpen((open) => !open)}
                    className="rounded-lg border border-amber-300 px-3 py-2 text-xs font-semibold text-amber-900"
                  >
                    {manualOpen ? "Close manual answer editor" : "Edit / separate answers manually"}
                  </button>
                )}
                {(result.needs_review || manualOpen) && <>
                <div className="space-y-3">
                  {manualAnswers.map((item, index) => (
                    <article key={item.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                        <label className="text-xs font-semibold text-slate-700">
                          Answer {index + 1}
                          <select
                            className="ml-2 min-w-36 rounded-md border border-slate-300 bg-white px-2 py-1.5"
                            value={item.question_id}
                            onChange={(event) => updateManualAnswer(item.id, { question_id: event.target.value })}
                          >
                            <option value="">Assign question…</option>
                            {rubrics.map((rubric) => (
                              <option key={rubric.question_id} value={rubric.question_id}>
                                {rubric.question_id}
                              </option>
                            ))}
                          </select>
                        </label>
                        <div className="flex items-center gap-1">
                          <button type="button" aria-label="Move answer up" disabled={index === 0} onClick={() => moveManualAnswer(index, -1)} className="rounded p-1 text-slate-500 disabled:opacity-30"><ChevronUp className="h-4 w-4" /></button>
                          <button type="button" aria-label="Move answer down" disabled={index === manualAnswers.length - 1} onClick={() => moveManualAnswer(index, 1)} className="rounded p-1 text-slate-500 disabled:opacity-30"><ChevronDown className="h-4 w-4" /></button>
                          <button type="button" aria-label="Delete answer" onClick={() => setManualAnswers((current) => current.filter((entry) => entry.id !== item.id))} className="rounded p-1 text-red-600"><Trash2 className="h-4 w-4" /></button>
                        </div>
                      </div>
                      <textarea
                        className="min-h-28 w-full rounded-md border border-slate-300 bg-white p-2 text-sm"
                        value={item.answer}
                        onChange={(event) => updateManualAnswer(item.id, { answer: event.target.value })}
                        placeholder="Paste or edit the extracted answer text"
                      />
                    </article>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => setManualAnswers((current) => [...current, { id: `${Date.now()}-${Math.random()}`, question_id: "", answer: "" }])}
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700"
                  >
                    <Plus className="h-4 w-4" /> Add answer
                  </button>
                  <button
                    type="button"
                    disabled={manualBusy || !manualAnswers.length}
                    onClick={submitManualAnswers}
                    className="rounded-lg bg-amber-700 px-4 py-2 text-xs font-semibold text-white disabled:opacity-50"
                  >
                    {manualBusy ? "Evaluating every answer…" : "Confirm answers & evaluate"}
                  </button>
                </div>
                </>}
              </section>
            )}

            {result.ia && (
              <div className="bg-white border border-emerald-200 rounded-lg p-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                <div>
                  <span className="text-slate-500 block">Q1 Pair</span>
                  <span className="font-bold text-slate-800 text-sm">{result.ia.q1_total} / 10</span>
                </div>
                <div>
                  <span className="text-slate-500 block">Q2 Pair</span>
                  <span className="font-bold text-slate-800 text-sm">{result.ia.q2_total} / 10</span>
                </div>
                <div>
                  <span className="text-slate-500 block">Compulsory Q3</span>
                  <span className="font-bold text-slate-800 text-sm">{result.ia.q3_total} / 10</span>
                </div>
                <div className="border-l border-emerald-100 pl-2">
                  <span className="text-emerald-700 font-semibold block">SDMCET IA Total</span>
                  <span className="font-extrabold text-indigo-700 text-base">{result.ia.ia_total} / 20</span>
                </div>
              </div>
            )}

            {onGoToReview && (
              <button
                type="button"
                onClick={onGoToReview}
                className="mt-2 w-full inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold py-2 text-sm shadow-sm"
              >
                Proceed to Side-by-Side Review &amp; Calibration
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
