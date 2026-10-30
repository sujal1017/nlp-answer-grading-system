import { useCallback, useEffect, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";

export function Card({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <div className="flex gap-2">{actions}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

const btn = "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50";
export function Button({ variant = "secondary", busy, children, ...p }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger"; busy?: boolean }) {
  const v = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700",
    secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
    danger: "border border-red-300 bg-white text-red-700 hover:bg-red-50",
  }[variant];
  return (
    <button {...p} disabled={p.disabled || busy} className={`${btn} ${v} ${p.className ?? ""}`}>
      {busy && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

export const inputCls = "w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block font-medium text-slate-700">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

const tones: Record<string, string> = {
  green: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  amber: "bg-amber-50 text-amber-800 ring-amber-200",
  red: "bg-red-50 text-red-700 ring-red-200",
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
  blue: "bg-indigo-50 text-indigo-700 ring-indigo-200",
};
export function Badge({ tone = "slate", children }: { tone?: keyof typeof tones; children: ReactNode }) {
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${tones[tone]}`}>{children}</span>;
}

export function statusTone(s: string) {
  if (["reviewed", "approved", "correct", "done", "answered", "ai_graded"].includes(s)) return "green";
  if (["needs_review", "manual_required", "partial", "mentioned", "ambiguous", "uploaded", "ocr_running", "running", "queued"].includes(s)) return "amber";
  if (["failed", "incorrect", "missing", "unreadable"].includes(s)) return "red";
  return "slate";
}
export const StatusBadge = ({ s }: { s: string }) => <Badge tone={statusTone(s) as any}>{s.replace(/_/g, " ")}</Badge>;

export function Notice({ kind = "info", children }: { kind?: "info" | "warn" | "error" | "success"; children: ReactNode }) {
  const m = {
    info: ["border-indigo-200 bg-indigo-50 text-indigo-900", Info],
    warn: ["border-amber-200 bg-amber-50 text-amber-900", AlertTriangle],
    error: ["border-red-200 bg-red-50 text-red-900", XCircle],
    success: ["border-emerald-200 bg-emerald-50 text-emerald-900", CheckCircle2],
  }[kind] as [string, typeof Info];
  const Icon = m[1];
  return (
    <div className={`flex gap-2 rounded-md border px-3 py-2 text-sm ${m[0]}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 break-words">{children}</div>
    </div>
  );
}

export const Spinner = ({ label = "Loading…" }: { label?: string }) => (
  <div className="flex items-center gap-2 p-6 text-sm text-slate-500"><Loader2 className="h-4 w-4 animate-spin" />{label}</div>
);
export const Empty = ({ children }: { children: ReactNode }) => <div className="rounded-md border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">{children}</div>;

export function Progress({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      <div className="h-2 overflow-hidden rounded bg-slate-200"><div className="h-full bg-indigo-600 transition-all" style={{ width: `${Math.round(value * 100)}%` }} /></div>
      {label && <p className="mt-1 text-xs text-slate-500">{label}</p>}
    </div>
  );
}

/** Load data from an async function with loading/error state and a reload handle. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const load = useCallback(async () => {
    try {
      setError(null);
      setData(await fn());
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, deps);
  useEffect(() => { setLoading(true); load(); }, [load]);
  return { data, error, loading, reload: load, setData };
}

export const fmt = (n: number | null | undefined) => (n === null || n === undefined ? "—" : Number(n.toFixed(2)).toString());
