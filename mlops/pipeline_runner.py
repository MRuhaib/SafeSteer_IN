#!/usr/bin/env python
"""
SafeSteer-IN MLOps Pipeline Runner
==================================
Runs the offline artifact pipeline stages in sequence using the stage scripts
under src/scripts/. Supports selective stages, per-stage arguments, and optional
SMTP notifications.

Examples:
    python mlops/pipeline_runner.py
    python mlops/pipeline_runner.py --stages 01_build_dataset 02_extract_vectors
    python mlops/pipeline_runner.py --stage-args "02_extract_vectors:--model sarvam-1 --no-probe"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_SCRIPTS_DIR = PROJECT_ROOT / "src" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


STAGES = {
    "01_build_dataset": "01_build_dataset.py",
    "02_extract_vectors": "02_extract_vectors.py",
    "03_train_classifier": "03_train_classifier.py",
    "04_calibrate_alpha": "04_calibrate_alpha.py",
    "05_evaluate": "05_evaluate.py",
    "06_launch_demo": "06_launch_demo.py",
}

DEFAULT_STAGE_ORDER = [
    "01_build_dataset",
    "02_extract_vectors",
    "03_train_classifier",
    "04_calibrate_alpha",
    "05_evaluate",
]


@dataclass
class NotifyConfig:
    enabled: bool
    recipients: List[str]
    mlflow_uri: str


def _parse_stage_args(stage_args: List[str]) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for item in stage_args:
        if ":" not in item:
            raise ValueError(
                "Stage args must be in the form 'stage:--arg value --flag'"
            )
        stage, arg_str = item.split(":", 1)
        stage = stage.strip()
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}' in stage args")
        parsed[stage] = shlex.split(arg_str.strip()) if arg_str.strip() else []
    return parsed


def _resolve_stage_list(stages: List[str]) -> List[str]:
    if not stages:
        return list(DEFAULT_STAGE_ORDER)
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid: {list(STAGES)}")
    return stages


def _get_notify_config(raw_recipients: str | None) -> NotifyConfig:
    recipients: List[str] = []
    if raw_recipients:
        recipients = [r.strip() for r in raw_recipients.split(",") if r.strip()]
    enabled = len(recipients) > 0
    mlflow_uri = (
        os.getenv("MLFLOW_UI_URI")
        or os.getenv("MLFLOW_TRACKING_URI")
        or "http://localhost:5000"
    )
    return NotifyConfig(enabled=enabled, recipients=recipients, mlflow_uri=mlflow_uri)


def _send_notification(
    notify: NotifyConfig,
    run_id: str,
    stage: str,
    status: str,
    error_msg: str | None = None,
) -> None:
    if not notify.enabled:
        return
    try:
        from notifications.smtp_handler import SMTPNotifier

        notifier = SMTPNotifier()
        notifier.send_pipeline_completion(
            recipient_emails=notify.recipients,
            run_id=run_id,
            stage=stage,
            status=status,
            mlflow_uri=notify.mlflow_uri,
            error_msg=error_msg,
        )
    except Exception:
        return


def _run_stage(
    stage: str,
    python_bin: str,
    extra_args: List[str],
    dry_run: bool,
) -> int:
    script_path = SRC_SCRIPTS_DIR / STAGES[stage]
    if not script_path.exists():
        raise FileNotFoundError(f"Stage script not found: {script_path}")
    cmd = [python_bin, str(script_path)] + extra_args
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SafeSteer-IN pipeline stages")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (default: 01-05).",
    )
    parser.add_argument(
        "--stage-args",
        nargs="*",
        default=[],
        help="Per-stage args: stage:--arg value --flag",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--notify",
        type=str,
        default=os.getenv("PIPELINE_NOTIFY_EMAILS", ""),
        help="Comma-separated emails for SMTP alerts (or set PIPELINE_NOTIFY_EMAILS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue remaining stages even if one fails",
    )
    args = parser.parse_args()

    stage_list = _resolve_stage_list(args.stages or [])
    stage_args = _parse_stage_args(args.stage_args)
    notify = _get_notify_config(args.notify)

    pipeline_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("SafeSteer-IN - MLOps Pipeline Runner")
    print("=" * 72)
    print(f"Stages: {stage_list}")
    print(f"Python: {args.python}")
    if notify.enabled:
        print(f"Notifications: {', '.join(notify.recipients)}")
    print()

    for stage in stage_list:
        print(f"==> Running {stage}")
        extra = stage_args.get(stage, [])
        try:
            code = _run_stage(stage, args.python, extra, args.dry_run)
        except Exception as exc:
            _send_notification(
                notify,
                pipeline_run_id,
                stage,
                status="FAILED",
                error_msg=str(exc),
            )
            print(f"[ERROR] {stage} failed: {exc}")
            if not args.continue_on_fail:
                return 1
            continue

        if code != 0:
            _send_notification(
                notify,
                pipeline_run_id,
                stage,
                status="FAILED",
                error_msg=f"Exit code {code}",
            )
            print(f"[ERROR] {stage} exited with code {code}")
            if not args.continue_on_fail:
                return code
        else:
            _send_notification(
                notify,
                pipeline_run_id,
                stage,
                status="SUCCESS",
                error_msg=None,
            )
            print(f"[OK] {stage} completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
