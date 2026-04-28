"""SMTP notification handler for SafeSteer-IN pipeline events."""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SMTPConfig:
    """SMTP server configuration."""

    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
    use_tls: bool = True
    timeout: int = 10


def load_json_config(path: Path) -> dict:
    """Load JSON config; return an empty dict when missing/invalid."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read SMTP config %s: %s", path, exc)
        return {}


def _normalize_recipients(raw: str | List[str] | None) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [x.strip() for x in raw if x and x.strip()]


class SMTPNotifier:
    """Send pipeline notifications via SMTP."""

    TEMPLATES = {
        "pipeline_complete": """
<html>
  <body style="font-family: Arial, sans-serif;">
    <h2 style="color: #198754;">SafeSteer-IN pipeline stage completed</h2>
    <table style="border-collapse: collapse; margin: 12px 0;">
      <tr><td style="padding: 6px; font-weight: bold;">Run ID:</td><td style="padding: 6px;"><code>{run_id}</code></td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Stage:</td><td style="padding: 6px;">{stage}</td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Status:</td><td style="padding: 6px;">{status}</td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Timestamp:</td><td style="padding: 6px;">{timestamp}</td></tr>
    </table>
    <p><a href="{mlflow_uri}">Open MLflow run details</a></p>
    <p style="color: #6c757d;">{error_msg}</p>
  </body>
</html>
""",
        "pipeline_error": """
<html>
  <body style="font-family: Arial, sans-serif;">
    <h2 style="color: #dc3545;">SafeSteer-IN pipeline error</h2>
    <table style="border-collapse: collapse; margin: 12px 0;">
      <tr><td style="padding: 6px; font-weight: bold;">Run ID:</td><td style="padding: 6px;"><code>{run_id}</code></td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Stage:</td><td style="padding: 6px;">{stage}</td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Status:</td><td style="padding: 6px;">FAILED</td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Error:</td><td style="padding: 6px;"><code>{error_msg}</code></td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Timestamp:</td><td style="padding: 6px;">{timestamp}</td></tr>
    </table>
    <p><a href="{mlflow_uri}">Open MLflow run details</a></p>
  </body>
</html>
""",
        "evaluation_complete": """
<html>
  <body style="font-family: Arial, sans-serif;">
    <h2 style="color: #198754;">SafeSteer-IN evaluation complete</h2>
    <table style="border-collapse: collapse; margin: 12px 0;">
      <tr><td style="padding: 6px; font-weight: bold;">Model:</td><td style="padding: 6px;">{model_key}</td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Run ID:</td><td style="padding: 6px;"><code>{run_id}</code></td></tr>
      <tr><td style="padding: 6px; font-weight: bold;">Timestamp:</td><td style="padding: 6px;">{timestamp}</td></tr>
    </table>
    <p><a href="{mlflow_uri}">Open MLflow results</a></p>
  </body>
</html>
""",
    }

    def __init__(
        self,
        smtp_server: str | None = None,
        smtp_port: int | None = None,
        sender_email: str | None = None,
        sender_password: str | None = None,
        use_tls: bool = True,
        timeout: int = 10,
        config_path: str | Path | None = None,
    ) -> None:
        cfg_data = {}
        if config_path:
            cfg_data = load_json_config(Path(config_path))

        self.config = SMTPConfig(
            smtp_server=smtp_server
            or cfg_data.get("smtp_server")
            or os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(
                smtp_port
                if smtp_port is not None
                else cfg_data.get("smtp_port", os.getenv("SMTP_PORT", 587))
            ),
            sender_email=sender_email
            or cfg_data.get("sender_email")
            or os.getenv("SENDER_EMAIL", ""),
            sender_password=sender_password
            or cfg_data.get("sender_password")
            or cfg_data.get("password")
            or os.getenv("SENDER_PASSWORD", ""),
            use_tls=bool(cfg_data.get("use_tls", use_tls)),
            timeout=int(cfg_data.get("timeout", timeout)),
        )
        self._verify_config()

    def _verify_config(self) -> None:
        if not self.config.sender_email or not self.config.sender_password:
            logger.warning(
                "SMTP credentials are not fully configured. "
                "Set sender_email/sender_password via config or env."
            )

    def _send_email(self, recipient_email: str, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.sender_email
            msg["To"] = recipient_email
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(
                self.config.smtp_server,
                self.config.smtp_port,
                timeout=self.config.timeout,
            ) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.send_message(msg)
            logger.info("Email sent to %s", recipient_email)
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check email/password.")
            return False
        except Exception as exc:
            logger.error("Failed to send email: %s", exc)
            return False

    def send_pipeline_completion(
        self,
        recipient_emails: List[str] | str,
        run_id: str,
        stage: str,
        status: str = "SUCCESS",
        mlflow_uri: str = "http://localhost:5000",
        error_msg: Optional[str] = None,
    ) -> bool:
        recipients = _normalize_recipients(recipient_emails)
        if not recipients:
            logger.warning("No recipients supplied for pipeline completion email.")
            return False

        template_key = "pipeline_error" if status.upper() == "FAILED" else "pipeline_complete"
        html_body = self.TEMPLATES[template_key].format(
            run_id=run_id,
            stage=stage,
            status=status,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mlflow_uri=mlflow_uri,
            error_msg=error_msg or "Pipeline stage completed successfully.",
        )

        success = True
        for email in recipients:
            ok = self._send_email(
                recipient_email=email,
                subject=f"SafeSteer-IN Pipeline {status}: {stage}",
                html_body=html_body,
            )
            success = success and ok
        return success

    def send_evaluation_complete(
        self,
        recipient_emails: List[str] | str,
        model_key: str,
        run_id: str,
        mlflow_uri: str = "http://localhost:5000",
    ) -> bool:
        recipients = _normalize_recipients(recipient_emails)
        if not recipients:
            logger.warning("No recipients supplied for evaluation email.")
            return False

        html_body = self.TEMPLATES["evaluation_complete"].format(
            model_key=model_key,
            run_id=run_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mlflow_uri=mlflow_uri,
        )

        success = True
        for email in recipients:
            ok = self._send_email(
                recipient_email=email,
                subject="SafeSteer-IN Evaluation Complete",
                html_body=html_body,
            )
            success = success and ok
        return success


def _cli() -> int:
    parser = argparse.ArgumentParser(description="SafeSteer SMTP notifier")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "smtp_config.json"),
        help="Path to SMTP config JSON",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a test pipeline completion email",
    )
    parser.add_argument(
        "--recipient",
        action="append",
        default=[],
        help="Optional explicit recipient email. Can be provided multiple times.",
    )
    args = parser.parse_args()

    config_data = load_json_config(Path(args.config))
    notifier = SMTPNotifier(config_path=args.config)

    if not args.test:
        print("SMTP notifier ready. Use --test to send a test email.")
        return 0

    recipients = _normalize_recipients(args.recipient)
    if not recipients:
        recipients = _normalize_recipients(config_data.get("recipients"))
    if not recipients and notifier.config.sender_email:
        recipients = [notifier.config.sender_email]
    if not recipients:
        print("No recipients configured. Set recipients in smtp_config.json or pass --recipient.")
        return 1

    ok = notifier.send_pipeline_completion(
        recipient_emails=recipients,
        run_id="test_run_001",
        stage="smtp_test",
        status="SUCCESS",
        mlflow_uri=config_data.get("mlflow_uri", "file:./mlruns"),
    )
    print(f"Test email sent: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_cli())
