import { useCallback, useEffect, useState } from "react";
import { BadgeCheck, Cpu, FileText, ScanLine, Zap, RefreshCw, FileSpreadsheet } from "lucide-react";
import BatchGrading from "./components/BatchGrading.jsx";
import KpiCards from "./components/KpiCards.jsx";
import QuestionStudio from "./components/QuestionStudio.jsx";
import ReviewGrade from "./components/ReviewGrade.jsx";
import ResultsLedger from "./components/ResultsLedger.jsx";
import { getExam, getExams, getTelemetry, unloadEngine } from "./services/api";

const TABS = [
  { id: "studio", label: "01. Question Studio", icon: FileText },
  { id: "grade", label: "02. Batch Grading", icon: ScanLine },
  { id: "review", label: "03. Review & Calibrate", icon: BadgeCheck },
  { id: "ledger", label: "04. Marks Ledger", icon: FileSpreadsheet },
];

export default function App() {
  const [tab, setTab] = useState("studio");
  const [exams, setExams] = useState([]);
  const [examId, setExamId] = useState("");
  const [exam, setExam] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [flushing, setFlushing] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState("");

  const reload = useCallback(async (preferredId) => {
    try {
      const [{ data: examList }, { data: tel }] = await Promise.all([getExams(), getTelemetry()]);
      setExams(examList);
      setTelemetry(tel);
      const nextId = preferredId || examId || examList[0]?.exam_id || "";
      setExamId(nextId);
      if (nextId) {
        const { data } = await getExam(nextId);
        setExam({ ...data.exam, rubrics: data.rubrics, kpis: data.kpis });
      } else {
        setExam(null);
      }
    } catch {
      // Fallback if backend is warming up
    }
  }, [examId]);

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!examId) return;
    getExam(examId)
      .then(({ data }) => setExam({ ...data.exam, rubrics: data.rubrics, kpis: data.kpis }))
      .catch(() => setExam(null));
  }, [examId, refreshKey]);

  const bump = () => setRefreshKey((n) => n + 1);

  const handleFlush = async () => {
    setFlushing(true);
    try {
      const { data } = await unloadEngine();
      setTelemetry(data);
    } finally {
      setFlushing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="max-w-[1440px] mx-auto px-4 py-6 md:px-6">
        {/* Institutional Header Banner */}
        <header className="rounded-2xl bg-gradient-to-r from-slate-900 via-slate-850 to-indigo-950 text-white px-6 py-5 flex flex-wrap items-center justify-between gap-4 border border-slate-700 shadow-xl">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-tight">SDMCET — Automated Exam Evaluation Platform</span>
              <span className="text-[10px] font-bold uppercase tracking-wider bg-indigo-500/30 text-indigo-200 border border-indigo-400/30 px-2 py-0.5 rounded">
                IA Evaluation
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-1">
              Offline Qwen2-VL Digitization · Teacher Question Studio · Hybrid Semantic Boundary Chunker · SDMCET Choice Rules
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              className="rounded-lg bg-slate-800 border border-slate-600 text-sm font-semibold px-3 py-2 text-white shadow-inner"
              value={examId}
              onChange={(e) => setExamId(e.target.value)}
            >
              {exams.length === 0 && <option value="">No exams created</option>}
              {exams.map((e) => (
                <option key={e.exam_id} value={e.exam_id}>
                  {e.exam_id} — {e.exam_title || e.subject}
                </option>
              ))}
            </select>

            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-400/10 text-emerald-300 text-xs font-semibold px-3 py-1.5 shadow-sm">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              {telemetry?.cuda_available ? "CUDA Accelerated" : "CPU Fallback"} · {telemetry?.device_name || "RTX 4050"}
            </span>
          </div>
        </header>

        {/* Telemetry and Controls Bar */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-200 rounded-xl px-4 py-2.5 shadow-sm text-xs text-slate-600">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5 font-semibold text-slate-800">
              <Zap className="w-3.5 h-3.5 text-indigo-600" />
              <span>EvalAI Local Engine</span>
            </div>
            <span className="text-slate-300">|</span>
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              <span className="font-mono">{telemetry?.model || "Qwen2-VL-2B-Instruct"}</span>
            </div>
            <span className="text-slate-300">|</span>
            <span>Target: <b>{telemetry?.target_device || "NVIDIA RTX 4050 (6GB)"}</b></span>
            <span className="text-slate-300">|</span>
            {telemetry?.model_loaded ? (
              <span className="text-emerald-700 font-semibold bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                Active in VRAM ({telemetry.vram_allocated_gb} GB)
              </span>
            ) : (
              <span className="text-slate-400">VRAM Released (Ready on demand)</span>
            )}
          </div>

          <button
            type="button"
            onClick={handleFlush}
            disabled={flushing}
            className="inline-flex items-center gap-1.5 text-xs font-semibold rounded-lg border border-slate-300 bg-slate-50 hover:bg-slate-100 px-3 py-1.5 text-slate-700 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${flushing ? "animate-spin" : ""}`} />
            Flush GPU VRAM Cache
          </button>
        </div>

        {/* Institutional KPI Cards */}
        <div className="mt-5">
          <KpiCards examId={examId} kpis={exam?.kpis} />
        </div>

        {/* Tab Navigation */}
        <nav className="flex flex-wrap gap-2 bg-slate-200/80 p-1.5 rounded-xl mb-6 w-fit shadow-inner">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold transition-all ${
                  active
                    ? "bg-white text-indigo-700 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <Icon className="w-4 h-4" />
                {t.label}
              </button>
            );
          })}
        </nav>

        {/* Tab Content */}
        {tab === "studio" && (
          <QuestionStudio
            exam={exam}
            onSaved={(id) => {
              reload(id);
              bump();
            }}
          />
        )}
        {tab === "grade" && (
          <BatchGrading
            examId={examId}
            onFinished={bump}
            onGoToReview={() => {
              bump();
              setTab("review");
            }}
          />
        )}
        {tab === "review" && (
          <ReviewGrade
            examId={examId}
            refreshKey={refreshKey}
            onCalibrated={bump}
            initialStudentId={selectedStudent}
            onGoToLedger={() => setTab("ledger")}
          />
        )}
        {tab === "ledger" && (
          <ResultsLedger
            examId={examId}
            onSelectStudent={(sid) => {
              setSelectedStudent(sid);
              setTab("review");
            }}
          />
        )}
      </div>
    </div>
  );
}
