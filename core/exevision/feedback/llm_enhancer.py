# core/exevision/feedback/llm_enhancer.py
"""LLM post-processor: transforms template-based feedback into natural coaching sentences."""
from __future__ import annotations

import dataclasses
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


class LLMFeedbackEnhancer:
    """Wraps a LangChain LCEL chain to rewrite RepFeedback.text as natural language.

    Args:
        api_key: DeepSeek API key. Caller must pass os.getenv("DEEPSEEK_API_KEY").
        model: DeepSeek model name. "deepseek-chat" = DeepSeek-V3.
        _chain: Optional pre-built chain for testing (bypasses LLM construction).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        _chain: Any = None,
    ) -> None:
        if _chain is not None:
            self.chain = _chain
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
