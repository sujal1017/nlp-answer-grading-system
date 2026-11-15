import { Award, CheckCircle2, FileText, FolderOpen } from "lucide-react";

export default function KpiCards({ examId, kpis }) {
  const items = [
    {
      label: "Active Exam Workspace",
      value: examId || "—",
      sub: "Institutional Assessment",
      icon: FolderOpen,
      color: "text-indigo-600 bg-indigo-50 border-indigo-100",
    },
    {
      label: "Configured Questions",
      value: kpis?.questions ?? 0,
      sub: "Atomic weighted rubrics",
      icon: FileText,
      color: "text-blue-600 bg-blue-50 border-blue-100",
    },
    {
      label: "Booklets Digitized",
      value: kpis?.booklets ?? 0,
      sub: "Multi-page student booklets",
      icon: Award,
      color: "text-purple-600 bg-purple-50 border-purple-100",
    },
    {
      label: "Teacher Verified Answers",
      value: `${kpis?.verified ?? 0} / ${kpis?.answers ?? 0}`,
      sub: `${kpis?.answers ? Math.round(((kpis.verified || 0) / kpis.answers) * 100) : 0}% verification rate`,
      icon: CheckCircle2,
      color: "text-emerald-600 bg-emerald-50 border-emerald-100",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <div
            key={item.label}
            className="bg-white border border-slate-200 rounded-xl px-5 py-4 shadow-sm hover:shadow-md transition-shadow flex items-start justify-between"
          >
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                {item.label}
              </p>
              <p className="mt-1 text-2xl font-bold text-slate-900 tracking-tight">
                {item.value}
              </p>
              <p className="mt-1 text-xs text-slate-500">{item.sub}</p>
            </div>
            <div className={`p-2.5 rounded-xl border ${item.color}`}>
              <Icon className="w-5 h-5" />
            </div>
          </div>
        );
      })}
    </div>
  );
}
