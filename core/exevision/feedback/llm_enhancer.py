# core/exevision/feedback/llm_enhancer.py
"""LLM post-processor: transforms template-based feedback into natural coaching sentences."""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from core.exevision.feedback.engine import FeedbackResult, RepFeedback

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert exercise coach. Rewrite the following template-based feedback "
    "as 1-2 natural, encouraging coaching sentences. Be concise and direct. "
    "Do NOT add any information not present in the original feedback."
)

_HUMAN_TEMPLATE = (
    "Exercise: {exercise}\n"
    "Rep score: {score}/100 (performance tier: {tier})\n"
    "Issues detected: {issues}\n"
    "Template feedback: {template_text}\n\n"
    "Rewrite as natural coaching:"
)

_SESSION_SYSTEM_PROMPT = (
    "You are an expert exercise coach reviewing a complete workout session. "
    "You will receive structured per-rep data: each rep's overall score, tier, "
    "sub-metric scores, and detected issues/wins. Write a single cohesive "
    "'Coach's Notes' summary (3-5 sentences) that:\n"
    "1. Identifies concrete trends across reps (a metric improving or declining, "
    "a recurring issue) and references actual metric names and values from the data.\n"
    "2. Ends with 1-2 specific, actionable recommendations for the next session.\n"
    "Do NOT invent data not present in the input. Be direct, specific, and encouraging."
)

_SESSION_HUMAN_TEMPLATE = (
    "Exercise: {exercise}\n"
    "Reps: {rep_count}\n"
    "Average score: {avg_score}/100\n"
    "Trajectory: {trajectory}\n"
    "Per-rep data (JSON):\n{session_digest}\n\n"
    "Write the Coach's Notes:"
)


class LLMFeedbackEnhancer:
    """Wraps a LangChain LCEL chain to rewrite RepFeedback.text as natural language.

    Args:
        api_key: DeepSeek API key. Caller must pass os.getenv("DEEPSEEK_API_KEY").
        model: DeepSeek model name. "deepseek-chat" = DeepSeek-V3.
        _chain: Optional pre-built rep chain for testing (bypasses LLM construction).
        _session_chain: Optional pre-built session chain for testing.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        _chain: Any = None,
        _session_chain: Any = None,
    ) -> None:
        if _chain is not None or _session_chain is not None:
            self.chain = _chain
            self.session_chain = _session_chain
            return
        if not api_key:
            raise ValueError("api_key must be a non-empty string (set DEEPSEEK_API_KEY env var)")
        from langchain_deepseek import ChatDeepSeek
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatDeepSeek(model=model, temperature=0.7, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_TEMPLATE),
        ])
        self.chain = prompt | llm | StrOutputParser()

        session_prompt = ChatPromptTemplate.from_messages([
            ("system", _SESSION_SYSTEM_PROMPT),
            ("human", _SESSION_HUMAN_TEMPLATE),
        ])
        self.session_chain = session_prompt | llm | StrOutputParser()

    def enhance_rep(self, rep: RepFeedback, exercise: str) -> str:
        """Return LLM-rewritten text for one rep, or original text on any failure."""
        issues = [
            item["metric_key"]
            for item in rep.items
            if item.get("type") == "issue" and item.get("metric_key")
        ]
        try:
            return self.chain.invoke({
                "exercise": exercise,
                "score": int(rep.score),
                "tier": rep.tier,
                "issues": ", ".join(issues) if issues else "none",
                "template_text": rep.text,
            })
        except Exception as exc:
            logger.warning("LLM enhance_rep failed for rep %d: %s", rep.rep_id, exc)
            return rep.text

    def enhance_result(self, result: FeedbackResult) -> FeedbackResult:
        """Return new FeedbackResult with all rep texts replaced. Original is not mutated."""
        enhanced_reps = [
            dataclasses.replace(rep, text=self.enhance_rep(rep, result.exercise))
            for rep in result.reps
        ]
        return dataclasses.replace(result, reps=enhanced_reps)

    def enhance_session(self, result: FeedbackResult) -> FeedbackResult:
        """Return new FeedbackResult with session.coach_text replaced by a data-grounded
        summary across all reps, or unchanged on any failure."""
        try:
            new_coach_text = self.session_chain.invoke({
                "exercise": result.exercise,
                "rep_count": len(result.reps),
                "avg_score": round(result.session.avg_score),
                "trajectory": result.session.trajectory,
                "session_digest": json.dumps(result.session.session_digest),
            })
            new_session = dataclasses.replace(result.session, coach_text=new_coach_text)
            return dataclasses.replace(result, session=new_session)
        except Exception as exc:
            logger.warning("LLM enhance_session failed: %s", exc)
            return result
