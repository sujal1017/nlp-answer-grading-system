import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, Field, Notice, Spinner, inputCls, useLoad } from "../ui";

type Values = Record<string, string | number>;
interface SettingsResp { settings: Values; choices: Record<string, string[]> }
interface SystemInfo {
  tesseract: boolean; torch: boolean; transformers: boolean; sentence_transformers: boolean;
  external_llm_configured: boolean; max_upload_mb: number; notes: string[];
}

const GROUPS: { title: string; keys: string[] }[] = [
  { title: "OCR configuration", keys: ["ocr_engine", "ocr_model_id", "ocr_dpi"] },
  { title: "AI model configuration", keys: ["embedder", "embedder_model", "judge", "max_grading_attempts"] },
  { title: "Processing limits", keys: ["max_pages"] },
  { title: "Data retention", keys: ["retention_days"] },
  { title: "Review & metrics", keys: ["mark_tolerance", "min_metric_samples"] },
  { title: "Concept credit (fraction of criterion marks)", keys: ["credit_correct", "credit_partial", "credit_mentioned", "credit_incorrect", "credit_missing"] },
];

const HINTS: Record<string, string> = {
  ocr_engine: "tesseract works on CPU. qwen2vl needs torch + transformers and a model download. remote_vlm uses an external service.",
  ocr_dpi: "Render resolution for scanned pages (72–400).",
  embedder: "auto uses sentence-transformers if installed, otherwise a lexical fallback (not semantic).",
  judge: "heuristic runs locally. external_llm needs EXTERNAL_LLM_URL/MODEL configured on the server; page text is sent to that service.",
  max_pages: "Maximum pages accepted per PDF.",
  retention_days: "Submissions older than this are deleted when you run the purge. 0 keeps everything.",
  mark_tolerance: "Marks difference counted as agreement in AI-vs-human metrics.",
  min_metric_samples: "Metrics are not shown below this many human-reviewed answers.",
};

export default function SettingsPage() {
  const s = useLoad(() => api.get<SettingsResp>("/settings"), []);
  const sys = useLoad(() => api.get<SystemInfo>("/system"), []);
  const [form, setForm] = useState<Values>({});
  const [msg, setMsg] = useState<{ kind: "success" | "error" | "info"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (s.data) setForm(s.data.settings); }, [s.data]);

  if (s.loading) return <Spinner />;
  if (s.error || !s.data) return <Notice kind="error">{s.error}</Notice>;
  const choices = s.data.choices;

  async function save() {
    setBusy(true); setMsg(null);
    try {
      const r = await api.put<{ settings: Values }>("/settings", form);
      setForm(r.settings);
      setMsg({ kind: "success", text: "Settings saved." });
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); } finally { setBusy(false); }
  }

  async function purge() {
    if (!confirm("Delete all submissions older than the retention period? This cannot be undone.")) return;
    try {
      const r = await api.post<{ deleted: number; message?: string }>("/maintenance/purge");
      setMsg({ kind: "info", text: r.message || `Deleted ${r.deleted} old submission(s).` });
    } catch (e: any) { setMsg({ kind: "error", text: e.message }); }
  }

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-xl font-semibold">Settings</h1>
      {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}

      <Card title="Detected on this server">
        {sys.data ? (
          <div className="space-y-2 text-sm">
            <div className="flex flex-wrap gap-2">
              {([["Tesseract", sys.data.tesseract], ["PyTorch", sys.data.torch], ["Transformers", sys.data.transformers],
                ["Sentence-Transformers", sys.data.sentence_transformers], ["External LLM", sys.data.external_llm_configured]] as [string, boolean][])
                .map(([n, ok]) => <Badge key={n} tone={ok ? "green" : "slate"}>{n}: {ok ? "available" : "not available"}</Badge>)}
              <Badge>Upload limit {sys.data.max_upload_mb} MB</Badge>
            </div>
            <ul className="list-disc pl-5 text-xs text-slate-500">{sys.data.notes.map((n) => <li key={n}>{n}</li>)}</ul>
          </div>
        ) : <Spinner />}
      </Card>

      {GROUPS.map((g) => (
        <Card key={g.title} title={g.title}>
          <div className="grid gap-3 sm:grid-cols-2">
            {g.keys.map((k) => (
              <Field key={k} label={k.replace(/_/g, " ")} hint={HINTS[k]}>
                {choices[k] ? (
                  <select className={inputCls} value={String(form[k] ?? "")} onChange={(e) => setForm({ ...form, [k]: e.target.value })}>
                    {choices[k].map((c) => <option key={c}>{c}</option>)}
                  </select>
                ) : (
                  <input className={inputCls} value={String(form[k] ?? "")} onChange={(e) => setForm({ ...form, [k]: e.target.value })} />
                )}
              </Field>
            ))}
          </div>
        </Card>
      ))}

      <div className="flex gap-2">
        <Button variant="primary" busy={busy} onClick={save}>Save settings</Button>
        <Button variant="danger" onClick={purge}>Purge expired submissions</Button>
      </div>
    </div>
  );
}
