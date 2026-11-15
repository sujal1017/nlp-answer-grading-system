import { useEffect, useState } from "react";
import { Rocket, Upload, FileText, CheckCircle2, AlertTriangle, ArrowRight, Layers } from "lucide-react";
import { getJob, getSampleBooklets, gradeBooklet, gradeSampleBooklet } from "../services/api";

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
        setProgress(`Queued ${data.pages} booklet pages for VLM inference`);
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
      onFinished?.();
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="surface-card max-w-4xl mx-auto">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <h3 className="text-lg font-bold text-slate-900">Batch Student Booklet Digitization &amp; Grading</h3>
          <p className="text-sm text-slate-500 mt-0.5">
            Target Exam: <span className="font-mono font-bold text-indigo-700">{examId || "—"}</span>. Runs locally with CUDA acceleration on NVIDIA RTX 4050.
          </p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
          Qwen2-VL-2B (CUDA)
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
          {busy ? progress || "Processing Booklet with Qwen2-VL..." : "Run Multi-Page OCR & Automated Evaluation"}
        </button>

        {/* Live Progress Banner */}
        {busy && (
          <div className="rounded-xl border border-indigo-200 bg-indigo-50/70 p-4 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm font-semibold text-indigo-900">{progress}</p>
            </div>
            <p className="text-xs text-indigo-700 mt-1 ml-7">
              Executing temperature=0.0 greedy decoding on RTX 4050 with negative prompt filters to eliminate hallucinations.
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
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                <p className="font-bold text-emerald-950">Evaluation Complete for {result.student_id}</p>
              </div>
              <span className="text-xs font-semibold bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full">
                {result.chunks?.length || 0} Questions Segmented
              </span>
            </div>

            <p className="text-sm text-emerald-900">
              Extracted student answers: <span className="font-mono font-bold">{(result.chunks || []).join(", ")}</span>
            </p>

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
