// Thin typed client for the Flask API. No mock data anywhere: every call hits the backend.
export class ApiError extends Error {
  status: number;
  details?: unknown;
  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function request<T>(method: string, url: string, body?: unknown, form?: FormData): Promise<T> {
  const init: RequestInit = { method };
  if (form) init.body = form;
  else if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  let res: Response;
  try {
    res = await fetch("/api" + url, init);
  } catch {
    throw new ApiError("Cannot reach the backend. Is the Flask server running on port 5000?", 0);
  }
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const det = data?.details;
    const extra = Array.isArray(det) ? " " + det.join("; ") : "";
    throw new ApiError((data?.error || data?.message || `Request failed (${res.status})`) + extra, res.status, det);
  }
  return data as T;
}

export const api = {
  get: <T>(u: string) => request<T>("GET", u),
  post: <T>(u: string, b?: unknown) => request<T>("POST", u, b ?? {}),
  put: <T>(u: string, b: unknown) => request<T>("PUT", u, b),
  patch: <T>(u: string, b: unknown) => request<T>("PATCH", u, b),
  del: <T>(u: string) => request<T>("DELETE", u),
  upload: <T>(u: string, file: File, fields: Record<string, string> = {}) => {
    const f = new FormData();
    f.append("file", file);
    Object.entries(fields).forEach(([k, v]) => f.append(k, v));
    return request<T>("POST", u, undefined, f);
  },
};

/** Poll a background job until it finishes. */
export async function waitForJob(id: number, onProgress?: (j: Job) => void): Promise<Job> {
  for (;;) {
    const j = await api.get<Job>(`/jobs/${id}`);
    onProgress?.(j);
    if (j.status === "done" || j.status === "failed") return j;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// ------------------------------------------------------------------ types
export interface Job { id: number; kind: string; ref_id: number; status: string; progress: number; message: string; error?: string | null }
export interface Concept { text: string; keywords: string[]; synonyms?: string[] }
export interface Criterion { id?: number; name: string; description?: string; max_marks: number; concepts: Concept[] }
export interface Question {
  id: number; exam_id: number; label: string; text: string; max_marks: number | null; marks_confirmed: boolean;
  qtype: string; source_page: number | null; source: string; needs_review: boolean; review_reasons: string[];
  reference_answer: string; reference_source: string; reference_approved: boolean;
  criteria: Criterion[]; rubric_total: number;
}
export interface Exam {
  id: number; name: string; subject: string; code: string; notes: string; choice_groups: string[][];
  question_count: number; submission_count: number; questions?: Question[]; documents?: { id: number; kind: string; filename: string; page_count: number }[];
}
export interface PageText {
  page_number: number; kind: string; method: string; engine: string; text_original: string; text_corrected: string | null;
  confidence: number | null; warnings: string[]; error: string | null; reviewed: boolean;
}
export interface ConceptAssessment { concept_id: string; concept_text: string; criterion_id: number; status: string; evidence: string | null; comment: string }
export interface CriterionAssessment { criterion_id: number; name: string; awarded: number; max_marks: number; concepts: ConceptAssessment[]; rationale: string }
export interface AiResult {
  total: number; max_total: number; criteria: CriterionAssessment[]; awarded_because: string[]; withheld_because: string[];
  incorrect_statements: string[]; improvements: string[]; stronger_answer_outline: string[]; flags: string[];
}
export interface Evaluation {
  id: number; answer_id: number; status: string; ai_total: number | null; ai_result: AiResult | null; ai_error: string | null;
  ai_flags: string[]; judge: string; embedder: string; human_criterion_scores: Record<string, number> | null;
  human_total: number | null; reviewer_comment: string | null; approved: boolean; final_total: number | null;
}
export interface Answer {
  id: number; question_id: number; question_label: string; text_original: string; text_corrected: string | null;
  pages: number[]; status: string; mapping_confirmed: boolean; notes: string[]; evaluation: Evaluation | null;
}
export interface Totals {
  total: number; max_total: number; percentage: number | null; all_final: boolean;
  questions: { question_id: number; label: string; max_marks: number | null; marks: number | null; kind: string; status: string; counted: boolean }[];
}
export interface Submission {
  id: number; exam_id: number; exam_name: string; student_identifier: string; status: string; page_count: number; filename: string;
  created_at: string; mapping_notes: string[]; pages?: PageText[]; answers?: Answer[]; unmapped_text?: string;
  totals?: Totals; questions?: Question[]; total?: number; max_total?: number; percentage?: number | null; needs_review?: number;
}
