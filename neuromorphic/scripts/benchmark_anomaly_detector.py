"""
Cross-run benchmark anomaly detector (issue #299).

No historical-mean tracking of any BenchmarkSuite metric existed before this:
each run's summary() was evaluated in isolation, so a regression that no
human happened to look at was functionally invisible. This script runs
BenchmarkSuite on a small CI-sized network, compares the two dashboard-
visible metrics (concept_separability, binding_accuracy -- see the issue's
staged minimal scope) against their rolling historical distribution, and
prints a flag for any metric more than `--threshold` std devs from that
history.

Unlike benchmark_ci_gate.py / learning_evidence_ci_gate.py, this never fails
the build (exit code is always 0): a statistical outlier can be a genuine
improvement as easily as a regression, so "flag it in CI output" (the
issue's own words) means surface it for a human to look at, not block merge
on it. Use benchmark_ci_gate.py / learning_evidence_ci_gate.py for hard
regression gates.

History persistence: neuromorphic/benchmarks/metric_history.json is a
committed, rolling JSON log (window-capped, see
BenchmarkSuite.DEFAULT_HISTORY_WINDOW). CI runs compare against it but do
NOT write back to it by default -- GitHub Actions runners are ephemeral and
this repo has no auto-commit-back step, so an unreviewed auto-append would
either be silently lost every run or require new write-back infrastructure
outside this issue's scope. --update-history appends the current run and
prints the updated JSON (mirroring the other two gates' --update-baseline)
for a maintainer to review and commit, keeping the same human-in-the-loop
model as ci_performance_baseline.json and learning_evidence_baseline.json.

Usage:
    cd neuromorphic && uv run python scripts/benchmark_anomaly_detector.py
    cd neuromorphic && uv run python scripts/benchmark_anomaly_detector.py \
        --update-history > /tmp/updated.json
        # review /tmp/updated.json, then copy it over
        # benchmarks/metric_history.json and commit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_NEURO_DIR = _SCRIPT_DIR.parent
_DEFAULT_HISTORY = _NEURO_DIR / "benchmarks" / "metric_history.json"

if str(_NEURO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_NEURO_DIR / "src"))

# Small CI network, matching ci_performance_baseline.json's profile but with
# a real (non-zero) concept layer -- concept_separability is meaningless
# (always the degenerate error shape) without one.
_DEFAULT_ENV: dict[str, str] = {
    "NEURO_BRAINSTEM_N": "50",
    "NEURO_REFLEX_N": "30",
    "NEURO_SENSORY_N": "200",
    "NEURO_MOTOR_N": "100",
    "NEURO_CEREBELLUM_N": "50",
    "NEURO_ASSOCIATION_N": "150",
    "NEURO_PREDICTIVE_N": "80",
    "NEURO_WORKING_MEM_N": "40",
    "NEURO_FEATURE_N": "0",
    "NEURO_CONCEPT_N": "100",
    "NEURO_DG_N": "0",
    "NEURO_META_N": "0",
    "NEURO_WORKSPACE_N": "0",
    "NEURO_DENDRITES": "0",
    "ENGRAM_SKIP_MUJOCO_LOOP": "1",
}


def apply_ci_env(env_overrides: dict[str, str]) -> None:
    for key, value in env_overrides.items():
        os.environ[key] = str(value)


def run_benchmark_suite(bench_args: dict[str, Any]) -> dict[str, Any]:
    """Build the CI-sized network and run BenchmarkSuite in-process."""
    from neuromorphic.benchmarks import BenchmarkSuite
    from neuromorphic.config import NeuromorphicConfig
    from neuromorphic.network import NeuromorphicNetwork

    config = NeuromorphicConfig.from_env()
    config.concept_layer.k_winners = int(bench_args.get("concept_k_winners", 10))
    # Fixed weight-init seed (issue #322's two-seeds distinction) keeps this
    # deterministic across CI runs, same fix applied in learning_evidence_ci_gate.py.
    network = NeuromorphicNetwork(config, seed=int(bench_args.get("network_seed", 1234)))
    suite = BenchmarkSuite(network)
    return suite.run_all(
        n_patterns=int(bench_args.get("n_patterns", 4)),
        training_reps=int(bench_args.get("training_reps", 3)),
        steps_per_pattern=int(bench_args.get("steps_per_pattern", 6)),
        seed=int(bench_args.get("seed", 42)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-run benchmark anomaly detector")
    parser.add_argument(
        "--history",
        type=Path,
        default=_DEFAULT_HISTORY,
        help="Rolling history JSON log (default: benchmarks/metric_history.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Std-dev threshold for flagging (default: DEFAULT_ANOMALY_THRESHOLD, 2.0)",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=None,
        help="Minimum prior runs required before comparing (default: DEFAULT_MIN_HISTORY, 3)",
    )
    parser.add_argument(
        "--update-history",
        action="store_true",
        help="Append this run's metrics to the rolling window and print the updated "
        "history JSON to stdout (does not write to disk)",
    )
    args = parser.parse_args()

    apply_ci_env(_DEFAULT_ENV)
    results = run_benchmark_suite({})

    from neuromorphic.benchmarks import (
        DEFAULT_ANOMALY_THRESHOLD,
        DEFAULT_MIN_HISTORY,
        detect_metric_anomalies,
        extract_tracked_metrics,
        format_anomaly_report,
        load_metric_history,
    )

    threshold = args.threshold if args.threshold is not None else DEFAULT_ANOMALY_THRESHOLD
    min_history = args.min_history if args.min_history is not None else DEFAULT_MIN_HISTORY

    metrics = extract_tracked_metrics(results)
    history = load_metric_history(args.history)

    if args.update_history:
        from neuromorphic.benchmarks import DEFAULT_HISTORY_WINDOW

        updated = dict(history)
        for name, value in metrics.items():
            updated.setdefault(name, []).append(value)
            updated[name] = updated[name][-DEFAULT_HISTORY_WINDOW:]
        print(
            json.dumps(
                {
                    "description": (
                        "Rolling per-metric history for cross-run anomaly detection "
                        "(issue #299)."
                    ),
                    "window_size": DEFAULT_HISTORY_WINDOW,
                    "history": updated,
                },
                indent=2,
            )
        )
        return 0

    anomalies = detect_metric_anomalies(
        metrics, history, threshold=threshold, min_history=min_history
    )
    print("=== Benchmark Anomaly Report ===")
    print(format_anomaly_report(anomalies))
    if any(a.flagged for a in anomalies):
        print(
            "\nOne or more metrics fell outside their historical range. This is "
            "informational, not a build failure -- review whether it's a "
            "regression or a genuine improvement."
        )
    # Always exit 0 -- see module docstring: this flags, it does not gate.
    return 0


if __name__ == "__main__":
    sys.exit(main())
