"""
LLM Insights Generator — calls OpenAI to generate business insights
from aggregated transaction metrics. Core of the ELT pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OpenAI client (graceful fallback if missing)
# ─────────────────────────────────────────────

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed — LLM calls will use mock responses.")


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial data analyst AI assistant embedded in a 
real-time transaction monitoring pipeline. Your role is to analyse aggregated transaction 
metrics and generate concise, actionable business insights for stakeholders.

Always respond in valid JSON with this exact structure:
{
  "insight_text": "<2-3 sentence executive summary>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "recommendations": ["<action 1>", "<action 2>"],
  "risk_level": "<low|medium|high|critical>"
}

Be precise, data-driven, and highlight anomalies or opportunities."""

USER_PROMPT_TEMPLATE = """Analyse the following transaction metrics from the last window 
and provide business insights:

METRICS:
- Window: {window_start} → {window_end}
- Total Transactions: {total_transactions}
- Total Amount: ${total_amount:,.2f}
- Average Transaction: ${avg_amount:,.2f}
- Maximum Transaction: ${max_amount:,.2f}
- Unique Users: {unique_users}
- Fraud Detected: {fraud_count} ({fraud_rate:.1%} fraud rate)
- Event Breakdown: {event_type_breakdown}
- Device Breakdown: {device_breakdown}
- Top Locations: {location_breakdown}

Generate insights, identify risks, and provide recommendations."""


# ─────────────────────────────────────────────
# Mock LLM response (used when OpenAI unavailable)
# ─────────────────────────────────────────────

def _mock_insight(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic mock insight for testing without API key."""
    fraud_rate = metrics.get("fraud_rate", 0)
    risk = "critical" if fraud_rate > 0.15 else "high" if fraud_rate > 0.08 else "medium" if fraud_rate > 0.03 else "low"
    return {
        "insight_text": (
            f"Processed {metrics.get('total_transactions', 0)} transactions "
            f"totalling ${metrics.get('total_amount', 0):,.2f}. "
            f"Fraud rate is {fraud_rate:.1%}, flagged as {risk} risk."
        ),
        "key_findings": [
            f"Average transaction value: ${metrics.get('avg_amount', 0):,.2f}",
            f"Fraud count: {metrics.get('fraud_count', 0)} transactions",
            f"Peak device: {list(metrics.get('device_breakdown', {}).keys())[0] if metrics.get('device_breakdown') else 'N/A'}",
        ],
        "recommendations": [
            "Review flagged transactions with fraud_probability > 0.85",
            "Investigate API-device transactions originating from unknown locations",
        ],
        "risk_level": risk,
    }


# ─────────────────────────────────────────────
# LLM Insights Engine
# ─────────────────────────────────────────────

class LLMInsightsEngine:
    """
    Wraps OpenAI API to generate structured business insights
    from aggregated streaming metrics.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> None:
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._client: Optional[Any] = None

        key = api_key or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and key:
            self._client = OpenAI(api_key=key)
            logger.info("LLMInsightsEngine connected to OpenAI (%s).", model)
        else:
            logger.warning("LLMInsightsEngine running in MOCK mode (no API key or openai package).")

    def generate_insight(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured business insight from metrics dict.

        Args:
            metrics: Output of aggregate_batch() from streaming layer.

        Returns:
            dict with insight_text, key_findings, recommendations, risk_level.
        """
        if self._client is None:
            return _mock_insight(metrics)

        prompt = USER_PROMPT_TEMPLATE.format(
            window_start=metrics.get("window_start", "N/A"),
            window_end=metrics.get("window_end", "N/A"),
            total_transactions=metrics.get("total_transactions", 0),
            total_amount=float(metrics.get("total_amount", 0)),
            avg_amount=float(metrics.get("avg_amount", 0)),
            max_amount=float(metrics.get("max_amount", 0)),
            unique_users=metrics.get("unique_users", 0),
            fraud_count=metrics.get("fraud_count", 0),
            fraud_rate=float(metrics.get("fraud_rate", 0)),
            event_type_breakdown=json.dumps(metrics.get("event_type_breakdown", {})),
            device_breakdown=json.dumps(metrics.get("device_breakdown", {})),
            location_breakdown=json.dumps(metrics.get("location_breakdown", {})),
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            parsed   = json.loads(raw_text)
            logger.info("LLM insight generated (risk=%s).", parsed.get("risk_level", "?"))
            return parsed
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON: %s", exc)
            return _mock_insight(metrics)
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            return _mock_insight(metrics)

    def generate_alert_summary(self, alerts: List[Dict[str, Any]]) -> str:
        """Summarise a list of alerts into a human-readable Slack/email message."""
        if not alerts:
            return "No alerts in this window."

        if self._client is None:
            lines = [f"• [{a.get('severity','?').upper()}] {a.get('title','Alert')}: {a.get('description','')}" for a in alerts[:5]]
            return "Alert Summary:\n" + "\n".join(lines)

        alert_text = "\n".join(
            f"- [{a.get('severity','?')}] {a.get('title','')}: {a.get('description','')}"
            for a in alerts[:10]
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise security analyst. Summarise these alerts in 3-4 sentences for a Slack notification."},
                    {"role": "user",   "content": f"Alerts:\n{alert_text}"},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Alert summary generation failed: %s", exc)
            return alert_text

    def batch_generate(self, metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights for multiple metric windows."""
        return [self.generate_insight(m) for m in metrics_list]


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = LLMInsightsEngine()
    sample_metrics = {
        "window_start": "2024-01-15T14:00:00",
        "window_end":   "2024-01-15T14:01:00",
        "total_transactions": 342,
        "total_amount":       187_430.50,
        "avg_amount":         548.04,
        "max_amount":         45_000.00,
        "unique_users":       89,
        "fraud_count":        14,
        "fraud_rate":         0.041,
        "event_type_breakdown": {"purchase": 180, "transfer": 95, "withdrawal": 67},
        "device_breakdown":     {"mobile": 200, "desktop": 100, "api": 42},
        "location_breakdown":   {"New York": 80, "Los Angeles": 60, "Unknown": 35},
    }
    result = engine.generate_insight(sample_metrics)
    print(json.dumps(result, indent=2))
