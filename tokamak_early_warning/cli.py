"""Minimal project CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_REPORTS_DIR, DEFAULT_SEED, utc_timestamp


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def run_reproduce(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    reports_root = repo_root / args.reports_dir
    run_dir = reports_root / utc_timestamp()
    train_out = run_dir / "train"
    eval_out = run_dir / "evaluation"

    train_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_root)
        _run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo_root)

    if not args.skip_fetch_data:
        _run(["bash", "scripts/fetch_data.sh"], cwd=repo_root)

    _run(
        [
            sys.executable,
            "scripts/train_tcn.py",
            "--seed",
            str(args.seed),
            "--output-dir",
            str(train_out),
        ]
    , cwd=repo_root)

    _run(
        [
            sys.executable,
            "scripts/evaluate_alarm_policy.py",
            "--model",
            "both",
            "--seed",
            str(args.seed),
            "--output-dir",
            str(eval_out),
            "--save-figures",
        ]
    , cwd=repo_root)

    summary_path = run_dir / "run_summary.json"
    summary = {
        "timestamp": run_dir.name,
        "seed": args.seed,
        "train_output": str(train_out),
        "evaluation_output": str(eval_out),
        "metrics_files": {
            "train": str(train_out / "tcn_results.json"),
            "evaluation": str(eval_out / "alarm_evaluation_results.json"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Reproducible run completed at: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokamak_early_warning",
        description="CLI for tokamak early warning research workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Run deterministic train + evaluation pipeline.",
    )
    reproduce_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    reproduce_parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    reproduce_parser.add_argument("--skip-install", action="store_true")
    reproduce_parser.add_argument("--skip-fetch-data", action="store_true")
    reproduce_parser.set_defaults(func=run_reproduce)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
