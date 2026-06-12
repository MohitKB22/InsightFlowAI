"""
Alerting Engine — evaluates processed events against rule thresholds
and dispatches notifications via Slack, email, and webhooks.
"""
from __future__ import annotations

import json
import logging
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

import requests

from backend.models import Alert, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Alert rules
# ─────────────────────────────────────────────

class AlertRule:
    """A single named alert rule with threshold and severity."""

    def __init__(
        self,
        name: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        condition: Callable[[Dict[str, Any]], bool],
        title_template: str,
        description_template: str,
    ) -> None:
        self.name                 = name
        self.alert_type           = alert_type
        self.severity             = severity
        self.condition            = condition
        self.title_template       = title_template
        self.description_template = description_template

    def evaluate(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Return an Alert if the condition fires, else None."""
        try:
            if not self.condition(event):
                return None
        except Exception:
            return None

        return Alert(
            alert_id    = str(uuid.uuid4()),
            alert_type  = self.alert_type,
            severity    = self.severity,
            title       = self.title_template.format(**event),
            description = self.description_template.format(**event),
            user_id     = event.get("user_id"),
            event_id    = event.get("event_id"),
            value       = event.get("transaction_amount"),
            threshold   = None,
        )


def build_default_rules(
    fraud_threshold: float = 0.85,
    high_amount: float = 10_000.0,
    failed_attempts: int = 5,
) -> List[AlertRule]:
    """Factory for the default rule set."""
    return [
        AlertRule(
            name="FRAUD_HIGH_PROB",
            alert_type=AlertType.FRAUD_DETECTED,
            severity=AlertSeverity.CRITICAL,
            condition=lambda e: float(e.get("fraud_probability", 0)) >= fraud_threshold,
            title_template="🚨 High-Probability Fraud — User {user_id}",
            description_template=(
                "Transaction {event_id} from user {user_id} scored "
                "{fraud_probability:.0%} fraud probability. "
                "Amount: ${transaction_amount:,.2f} | Location: {location}"
            ),
        ),
        AlertRule(
            name="HIGH_TRANSACTION",
            alert_type=AlertType.HIGH_TRANSACTION,
            severity=AlertSeverity.HIGH,
            condition=lambda e: float(e.get("transaction_amount", 0)) >= high_amount,
            title_template="💰 High-Value Transaction — ${transaction_amount:,.2f}",
            description_template=(
                "User {user_id} made a ${transaction_amount:,.2f} transaction "
                "via {device_type} from {location}."
            ),
        ),
        AlertRule(
            name="FAILED_ATTEMPTS",
            alert_type=AlertType.FAILED_ATTEMPTS,
            severity=AlertSeverity.MEDIUM,
            condition=lambda e: int(e.get("failed_attempts_1h", 0)) >= failed_attempts,
            title_template="🔐 Repeated Failed Logins — User {user_id}",
            description_template=(
                "User {user_id} has {failed_attempts_1h} failed login "
                "attempts in the last hour."
            ),
        ),
        AlertRule(
            name="ANOMALY_NIGHT_API",
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.MEDIUM,
            condition=lambda e: (
                e.get("device_type") == "api"
                and int(e.get("is_night", 0)) == 1
                and float(e.get("transaction_amount", 0)) > 1_000
            ),
            title_template="🌙 Night-time API Transaction — User {user_id}",
            description_template=(
                "Large API transaction (${transaction_amount:,.2f}) at night "
                "by {user_id} from {location}."
            ),
        ),
        AlertRule(
            name="VELOCITY_SPIKE",
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.HIGH,
            condition=lambda e: int(e.get("txn_count_1h", 0)) >= 20,
            title_template="⚡ Transaction Velocity Spike — User {user_id}",
            description_template=(
                "User {user_id} has {txn_count_1h} transactions in the last hour. "
                "Possible automated activity."
            ),
        ),
    ]


# ─────────────────────────────────────────────
# Notification dispatchers
# ─────────────────────────────────────────────

class SlackNotifier:
    def __init__(self, token: Optional[str], channel: str = "#data-alerts") -> None:
        self.token   = token
        self.channel = channel

    def send(self, alert: Alert) -> bool:
        if not self.token:
            logger.debug("Slack: no token — skipping. Alert: %s", alert.title)
            return False
        icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            alert.severity.value, "⚠️"
        )
        payload = {
            "channel": self.channel,
            "text": f"{icon} *{alert.title}*\n{alert.description}",
            "attachments": [
                {
                    "color": {"critical": "danger", "high": "warning",
                              "medium": "warning", "low": "good"}.get(alert.severity.value, "warning"),
                    "fields": [
                        {"title": "Severity",  "value": alert.severity.value.upper(), "short": True},
                        {"title": "Type",      "value": alert.alert_type.value,        "short": True},
                        {"title": "User",      "value": alert.user_id or "N/A",         "short": True},
                        {"title": "Triggered", "value": alert.triggered_at.isoformat(), "short": True},
                    ],
                }
            ],
        }
        try:
            resp = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
                timeout=5,
            )
            if resp.ok and resp.json().get("ok"):
                logger.info("Slack alert sent: %s", alert.title)
                return True
            logger.warning("Slack error: %s", resp.text[:200])
        except Exception as exc:
            logger.error("Slack send failed: %s", exc)
        return False


class EmailNotifier:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: Optional[str],
        password: Optional[str],
        to_address: Optional[str],
    ) -> None:
        self.smtp_host  = smtp_host
        self.smtp_port  = smtp_port
        self.username   = username
        self.password   = password
        self.to_address = to_address

    def send(self, alert: Alert) -> bool:
        if not all([self.username, self.password, self.to_address]):
            logger.debug("Email: credentials missing — skipping. Alert: %s", alert.title)
            return False
        try:
            msg            = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg["From"]    = self.username
            msg["To"]      = self.to_address

            html_body = f"""
            <html><body>
            <h2 style="color:{'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'}">
                {alert.title}
            </h2>
            <p>{alert.description}</p>
            <table border="1" cellpadding="5">
                <tr><td><b>Severity</b></td><td>{alert.severity.value}</td></tr>
                <tr><td><b>Type</b></td><td>{alert.alert_type.value}</td></tr>
                <tr><td><b>User</b></td><td>{alert.user_id or 'N/A'}</td></tr>
                <tr><td><b>Time</b></td><td>{alert.triggered_at}</td></tr>
            </table>
            </body></html>
            """
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, self.to_address, msg.as_string())
            logger.info("Email alert sent: %s", alert.title)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False


class WebhookNotifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            return False
        try:
            payload = {
                "alert_id":    alert.alert_id,
                "type":        alert.alert_type.value,
                "severity":    alert.severity.value,
                "title":       alert.title,
                "description": alert.description,
                "user_id":     alert.user_id,
                "triggered_at": alert.triggered_at.isoformat(),
            }
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            if resp.ok:
                logger.info("Webhook alert delivered: %s", alert.title)
                return True
            logger.warning("Webhook error: %s", resp.status_code)
        except Exception as exc:
            logger.error("Webhook failed: %s", exc)
        return False


# ─────────────────────────────────────────────
# Alert Engine — ties everything together
# ─────────────────────────────────────────────

class AlertEngine:
    """
    Evaluates processed events against all rules,
    stores triggered alerts, and dispatches notifications.
    """

    def __init__(
        self,
        rules: Optional[List[AlertRule]] = None,
        slack_token: Optional[str] = None,
        slack_channel: str = "#data-alerts",
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        alert_email: Optional[str] = None,
        webhook_url: Optional[str] = None,
        fraud_threshold: float = 0.85,
        high_amount: float = 10_000.0,
        failed_attempts: int = 5,
    ) -> None:
        self.rules: List[AlertRule] = rules or build_default_rules(
            fraud_threshold, high_amount, failed_attempts
        )
        self.alert_history: List[Alert] = []

        self._slack   = SlackNotifier(slack_token, slack_channel)
        self._email   = EmailNotifier(smtp_host, smtp_port, smtp_user, smtp_password, alert_email)
        self._webhook = WebhookNotifier(webhook_url)

    def evaluate(self, event: Dict[str, Any]) -> List[Alert]:
        """
        Run all rules against a single processed event.
        Returns list of triggered alerts (and dispatches notifications).
        """
        triggered: List[Alert] = []
        for rule in self.rules:
            alert = rule.evaluate(event)
            if alert:
                triggered.append(alert)
                self.alert_history.append(alert)
                self._dispatch(alert)
        return triggered

    def evaluate_batch(self, events: List[Dict[str, Any]]) -> List[Alert]:
        all_alerts: List[Alert] = []
        for event in events:
            all_alerts.extend(self.evaluate(event))
        return all_alerts

    def _dispatch(self, alert: Alert) -> None:
        """Fan-out notifications to all configured channels."""
        logger.info(
            "[ALERT %s] %s — %s",
            alert.severity.value.upper(),
            alert.title,
            alert.description[:80],
        )
        self._slack.send(alert)
        if alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            self._email.send(alert)
        self._webhook.send(alert)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [a.model_dump() for a in self.alert_history[-limit:]]

    def get_alert_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in self.alert_history:
            key = a.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = AlertEngine()
    test_events = [
        {
            "event_id": "evt-001", "user_id": "user_0001",
            "transaction_amount": 25000.0, "fraud_probability": 0.92,
            "device_type": "api", "location": "Unknown",
            "failed_attempts_1h": 8, "txn_count_1h": 25,
            "is_night": 1,
        },
        {
            "event_id": "evt-002", "user_id": "user_0050",
            "transaction_amount": 150.0, "fraud_probability": 0.12,
            "device_type": "mobile", "location": "New York",
            "failed_attempts_1h": 1, "txn_count_1h": 3,
            "is_night": 0,
        },
    ]
    for ev in test_events:
        alerts = engine.evaluate(ev)
        print(f"Event {ev['event_id']}: {len(alerts)} alert(s) triggered")
    print("\nAlert counts:", engine.get_alert_counts())
