"""Evaluation of the copilot: execution accuracy (level 1) and an LLM judge (level 2), on a golden set of questions."""
from .execution import Verdict, execution_verdict, wilson
from .golden import Golden, GoldenCase, load_golden
from .judge import Judge, calibrate
from .runner import CaseResult, EvalReport, evaluate

__all__ = ["CaseResult", "EvalReport", "Golden", "GoldenCase", "Judge", "Verdict", "calibrate", "evaluate", "execution_verdict",
           "load_golden", "wilson"]
