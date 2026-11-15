import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 600000,
});

export const getTelemetry = () => api.get("/telemetry");
export const unloadEngine = () => api.post("/engine/unload");
export const getExams = () => api.get("/exams");
export const getExam = (examId) => api.get(`/exams/${examId}`);
export const upsertExam = (payload) => api.post("/exams", payload);
export const parseQuestions = (text) => api.post("/parse-questions", { text });
export const updateRubric = (examId, questionId, payload) =>
  api.put(`/exams/${examId}/rubrics/${questionId}`, payload);
export const getSubmissions = (examId, studentId) =>
  api.get(`/exams/${examId}/submissions`, { params: studentId ? { student_id: studentId } : {} });
export const gradeBooklet = (examId, studentId, files) => {
  const form = new FormData();
  form.append("student_id", studentId);
  files.forEach((f) => form.append("files", f));
  return api.post(`/exams/${examId}/grade`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};
export const getSampleBooklets = () => api.get("/sample-booklets");
export const gradeSampleBooklet = (examId, payload) =>
  api.post(`/exams/${examId}/grade-sample`, payload);
export const getJob = (jobId) => api.get(`/jobs/${jobId}`);
export const verifyAnswer = (payload) => api.post("/verify", payload);
export const getIaTotal = (examId, studentId) => api.get(`/exams/${examId}/ia/${studentId}`);

export default api;
