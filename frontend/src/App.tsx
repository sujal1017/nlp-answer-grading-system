import { NavLink, Route, Routes } from "react-router-dom";
import { BarChart3, FileText, History as HistoryIcon, LayoutDashboard, Settings as SettingsIcon } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Exams from "./pages/Exams";
import ExamSetup from "./pages/ExamSetup";
import HistoryPage from "./pages/History";
import SubmissionPage from "./pages/Submission";
import SettingsPage from "./pages/Settings";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/exams", label: "Examinations", icon: FileText },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export default function App() {
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-56 shrink-0 border-r border-slate-200 bg-white md:block">
        <div className="flex items-center gap-2 px-4 py-4 text-sm font-bold text-slate-900"><BarChart3 className="h-5 w-5 text-indigo-600" />Answer Sheet Evaluation</div>
        <nav className="space-y-1 px-2">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium ${isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100"}`}>
              <Icon className="h-4 w-4" />{label}
            </NavLink>
          ))}
        </nav>
        <p className="px-4 pt-6 text-xs text-slate-400">AI marks are suggestions. A human reviewer approves every final mark.</p>
      </aside>
      <main className="min-w-0 flex-1 p-4 md:p-6">
        <nav className="mb-4 flex gap-3 text-sm md:hidden">
          {nav.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className="text-indigo-700 underline">{n.label}</NavLink>)}
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/exams" element={<Exams />} />
          <Route path="/exams/:id" element={<ExamSetup />} />
          <Route path="/submissions/:id" element={<SubmissionPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<p className="text-sm text-slate-500">Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
