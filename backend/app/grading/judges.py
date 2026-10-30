"""Judges decide, per rubric concept, how well the student's answer covers it.

A judge returns *only* concept assessments (status + evidence + comment). Marks are
always computed afterwards by the engine from the rubric, so a judge can never award
more than a criterion's maximum.

HeuristicJudge   - offline. Combines (1) key-term/synonym presence, (2) embedding
                   similarity of the best-matching answer sentence, (3) a negation
                   check for possible contradictions. Anything uncertain is flagged.
                   It cannot verify calculations, so numerical questions are routed
                   to manual grading by the engine.
ExternalLLMJudge - optional; calls an OpenAI-compatible endpoint you configure.
"""
import json
import re
from typing import Optional

import requests

from .embeddings import BaseEmbedder
from .schema import AnswerSpec, QuestionSpec
from .textutil import content_stems, has_negation, split_sentences, stem, tokens


class JudgeError(RuntimeError):
    """Judge could not produce an assessment (service down, bad credentials, ...)."""


class HeuristicJudge:
    name = "heuristic"
    supported_qtypes = {"definition", "explanation", "comparison", "application", "long_answer", "algorithm", "unknown"}

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    # -- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _phrase_present(phrase: str, answer_stems: list[str]) -> bool:
        ps = [stem(t) for t in tokens(phrase)]
        if not ps:
            return False
        n = len(ps)
        return any(answer_stems[i:i + n] == ps for i in range(len(answer_stems) - n + 1))

    def assess(self, question: QuestionSpec, answer: AnswerSpec) -> dict:
        sentences = split_sentences(answer.text)
        # sliding 2-sentence windows allow explanations that span a sentence boundary
        windows = list(sentences) + [f"{a} {b}" for a, b in zip(sentences, sentences[1:])]
        answer_stems = [stem(t) for t in tokens(answer.text)]
        concepts_out, flags = [], []
        emb = self.embedder
        for crit in question.criteria:
            for c in crit.concepts:
                terms = [t for t in (list(c.keywords) + list(c.synonyms)) if str(t).strip()]
                term_hit = any(self._phrase_present(t, answer_stems) for t in terms)
                key_stems = sorted({stem(t) for t in tokens(" ".join(map(str, c.keywords))) if len(t) > 1}
                                   or set(content_stems(c.text)))
                best_sim, best_win, best_cov = 0.0, "", 0.0
                if windows:
                    sims = emb.pairwise(c.text, windows)
                    for win, sim in zip(windows, sims):
                        wstems = set(content_stems(win))
                        cov = (sum(1 for k in key_stems if k in wstems) / len(key_stems)) if key_stems else 0.0
                        score = sim + 0.15 * cov
                        if score > best_sim + 0.15 * best_cov:
                            best_sim, best_win, best_cov = sim, win, cov
                words_in_best = len(tokens(best_win))
                status, comment = "missing", "No part of the answer addresses this concept."
                # Similarity is one signal only: the lexical fallback additionally requires key-term
                # coverage because it measures vocabulary overlap, not meaning.
                explicit_kw = len([k for k in c.keywords if str(k).strip()]) >= 2
                strong = words_in_best >= 5 and (
                    (best_sim >= emb.hi and (emb.semantic or best_cov >= 0.5))
                    # the evaluator named the key terms; a full sentence containing all of them counts
                    or (explicit_kw and best_cov >= 0.99 and best_sim >= emb.lo and words_in_best >= 6))
                weak = words_in_best >= 5 and (best_sim >= emb.lo or best_cov >= 0.5)
                if best_win and strong:
                    status, comment = "correct", "The answer explains this concept."
                elif best_win and weak:
                    status, comment = "partial", "The answer touches on this concept but the explanation is incomplete."
                elif term_hit:
                    status = "mentioned"
                    comment = "The term is mentioned but not explained."
                    best_win = best_win or self._sentence_with_term(sentences, terms)
                # contradiction heuristic: supports/negates the concept the opposite way
                neg_sentence = self._best_single_sentence(emb, c.text, best_win, sentences)
                if status in ("correct", "partial") and has_negation(neg_sentence) != has_negation(c.text):
                    status = "incorrect"
                    comment = ("The related statement may contradict the expected concept (negation mismatch); "
                               "a human should verify.")
                    flags.append(f"possible_contradiction:{c.id}")
                elif status in ("correct", "partial") and abs(best_sim - (emb.hi if status == "correct" else emb.lo)) < 0.04:
                    flags.append(f"borderline:{c.id}")
                concepts_out.append({
                    "criterion_id": crit.id, "concept_id": c.id, "status": status,
                    "evidence": best_win.strip() if status != "missing" and best_win else None,
                    "comment": comment,
                })
        return {"concepts": concepts_out, "flags": flags,
                "incorrect_statements": [
                    x["evidence"] for x in concepts_out if x["status"] == "incorrect" and x["evidence"]]}

    @staticmethod
    def _best_single_sentence(emb, concept_text: str, window: str, sentences: list[str]) -> str:
        """The single sentence inside `window` most similar to the concept (negation is checked
        per sentence, not across a two-sentence window, to avoid false contradictions)."""
        inside = [s for s in sentences if s and s in window]
        if not inside:
            return window
        sims = emb.pairwise(concept_text, inside)
        return inside[max(range(len(inside)), key=lambda i: sims[i])]

    @staticmethod
    def _sentence_with_term(sentences: list[str], terms: list[str]) -> str:
        for s in sentences:
            st = [stem(t) for t in tokens(s)]
            if any(HeuristicJudge._phrase_present(t, st) for t in terms):
                return s
        return ""


class ExternalLLMJudge:
    """OpenAI-compatible chat-completions judge. Disabled unless configured by environment.

    PRIVACY: answers are sent to the configured server. Use only a service you are
    permitted to send student data to.
    """

    name = "external_llm"
    supported_qtypes = {"definition", "explanation", "comparison", "application", "long_answer",
                        "algorithm", "numerical", "unknown"}

    def __init__(self, url: str, api_key: str, model: str, timeout: int = 120):
        if not (url and model):
            raise JudgeError("EXTERNAL_LLM_URL and EXTERNAL_LLM_MODEL must be set")
        self.url, self.api_key, self.model, self.timeout = url, api_key, model, timeout

    def _chat(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise JudgeError(f"external LLM request failed: {exc}") from exc

    def draft_reference_answer(self, question_text: str, max_marks: Optional[float]) -> str:
        return self._chat(
            "You are an expert examiner. Write a concise model answer suitable for the marks available.",
            f"Question ({max_marks or '?'} marks): {question_text}",
        ).strip()

    def assess(self, question: QuestionSpec, answer: AnswerSpec) -> dict:
        concept_lines = [
            f'- criterion_id={crit.id} concept_id="{c.id}": {c.text}'
            for crit in question.criteria for c in crit.concepts
        ]
        system = (
            "You are a careful examiner. For EACH listed concept decide the status of the student's answer: "
            "correct | partial | mentioned | incorrect | missing | not_applicable. 'mentioned' = term present but not "
            "explained. Accept valid synonyms and valid alternative explanations. Do not reward length or penalise "
            "grammar. Give 'evidence' as an EXACT quote from the student's answer (null if missing). Reply with JSON only: "
            '{"concepts":[{"criterion_id":int,"concept_id":str,"status":str,"evidence":str|null,"comment":str}],'
            '"incorrect_statements":[str],"flags":[str]}'
        )
        user = (f"Question ({question.qtype}, {question.max_marks} marks): {question.text}\n"
                f"Approved reference answer: {question.reference_answer}\nConcepts:\n" + "\n".join(concept_lines)
                + f"\n\nStudent answer:\n\"\"\"{answer.text}\"\"\"")
        raw = self._chat(system, user)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise JudgeError("external LLM did not return JSON")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise JudgeError(f"external LLM returned invalid JSON: {exc}") from exc
