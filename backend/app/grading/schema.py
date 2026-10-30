"""Typed specs (engine inputs) and the validated structured output of an evaluation."""
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

ConceptStatus = Literal["correct", "partial", "mentioned", "incorrect", "missing", "not_applicable"]
STATUS_NEEDS_EVIDENCE = {"correct", "partial", "mentioned", "incorrect"}


# ---------------- inputs (plain dataclasses: engine is independent of Flask/DB) ----------------
@dataclass
class ConceptSpec:
    id: str
    text: str
    keywords: list = field(default_factory=list)
    synonyms: list = field(default_factory=list)


@dataclass
class CriterionSpec:
    id: int
    name: str
    max_marks: float
    description: str = ""
    concepts: list = field(default_factory=list)  # list[ConceptSpec]


@dataclass
class QuestionSpec:
    id: int
    label: str
    text: str
    qtype: str
    max_marks: float
    reference_answer: str
    reference_approved: bool
    criteria: list = field(default_factory=list)  # list[CriterionSpec]


@dataclass
class AnswerSpec:
    text: str
    status: str = "answered"  # answered | unanswered | unreadable | ambiguous
    mapping_confirmed: bool = True


# ---------------- validated output ----------------
class ConceptAssessment(BaseModel):
    concept_id: str
    concept_text: str = ""
    criterion_id: int
    status: ConceptStatus
    evidence: Optional[str] = None
    comment: str = ""


class CriterionAssessment(BaseModel):
    criterion_id: int
    name: str
    awarded: float = Field(ge=0)
    max_marks: float = Field(ge=0)
    concepts: list[ConceptAssessment]
    rationale: str


class EvaluationResult(BaseModel):
    question_id: int
    total: float = Field(ge=0)
    max_total: float = Field(ge=0)
    criteria: list[CriterionAssessment]
    awarded_because: list[str] = []
    withheld_because: list[str] = []
    incorrect_statements: list[str] = []
    improvements: list[str] = []
    stronger_answer_outline: list[str] = []
    flags: list[str] = []


@dataclass
class Outcome:
    status: str  # ai_graded | needs_review | manual_required
    result: Optional[EvaluationResult] = None
    flags: list = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0
    judge_name: str = ""
    embedder_name: str = ""
