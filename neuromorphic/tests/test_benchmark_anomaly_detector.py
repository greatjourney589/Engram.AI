"""Tests for scripts/benchmark_anomaly_detector.py (issue #299)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_NEURO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import benchmark_anomaly_detector  # noqa: E402


def test_committed_history_is_valid_json():
    path = _NEURO_DIR / "benchmarks" / "metric_history.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "history" in data
    assert "window_size" in data
    for metric, values in data["history"].items():
        assert isinstance(values, list)
        assert len(values) > 0, f"{metric} history is empty"


def test_fresh_run_metrics_are_deterministic():
    """The CI network profile pins both the pattern seed and the network's own
    weight-init seed (issue #322's two-seeds distinction), so repeated runs
    must produce byte-identical metrics -- a flaky anomaly detector would be
    worse than none (real signal buried in noise)."""
    benchmark_anomaly_detector.apply_ci_env(benchmark_anomaly_detector._DEFAULT_ENV)

    from neuromorphic.benchmarks import extract_tracked_metrics

    first = extract_tracked_metrics(benchmark_anomaly_detector.run_benchmark_suite({}))
    second = extract_tracked_metrics(benchmark_anomaly_detector.run_benchmark_suite({}))
    assert first == second


def test_fresh_run_does_not_flag_against_committed_history():
    """Ground-truth check: a fresh run through the real gate pipeline must not
    be flagged against the committed history it was seeded from. If this
    fails, either there's a genuine drift, or the committed history has
    drifted from what the code now produces and needs refreshing via
    `python scripts/benchmark_anomaly_detector.py --update-history`.
    """
    from neuromorphic.benchmarks import (
        detect_metric_anomalies,
        extract_tracked_metrics,
        load_metric_history,
    )

    benchmark_anomaly_detector.apply_ci_env(benchmark_anomaly_detector._DEFAULT_ENV)
    results = benchmark_anomaly_detector.run_benchmark_suite({})
    metrics = extract_tracked_metrics(results)
    history = load_metric_history(_NEURO_DIR / "benchmarks" / "metric_history.json")

    anomalies = detect_metric_anomalies(metrics, history)
    flagged = [a for a in anomalies if a.flagged]
    assert flagged == [], (
        "Fresh run was flagged against the committed metric history.\n"
        "If this is a genuine deviation, investigate before dismissing it.\n"
        "If the committed history has drifted, refresh it with:\n"
        "  cd neuromorphic && python scripts/benchmark_anomaly_detector.py --update-history\n"
        f"Flagged: {flagged}"
    )


def test_main_exits_zero_even_when_history_would_flag(tmp_path, monkeypatch, capsys):
    """main() never fails the build -- see module docstring: this flags, it
    does not gate. Point it at a history file engineered to guarantee a flag
    and confirm the exit code is still 0."""
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps({"window_size": 30, "history": {"concept_separability": [0.0, 0.0, 0.0, 0.0]}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_anomaly_detector.py", "--history", str(history_path)],
    )
    exit_code = benchmark_anomaly_detector.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Benchmark Anomaly Report" in out


def test_update_history_flag_does_not_write_to_disk(tmp_path, monkeypatch, capsys):
    history_path = tmp_path / "history.json"
    original = json.dumps({"window_size": 30, "history": {}})
    history_path.write_text(original)

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_anomaly_detector.py", "--history", str(history_path), "--update-history"],
    )
    exit_code = benchmark_anomaly_detector.main()
    assert exit_code == 0
    assert history_path.read_text(encoding="utf-8") == original  # unchanged on disk

    printed = json.loads(capsys.readouterr().out)
    assert "concept_separability" in printed["history"]
    assert "binding_accuracy" in printed["history"]


@pytest.mark.parametrize("threshold_arg", ["--threshold", "--min-history"])
def test_cli_accepts_numeric_overrides(threshold_arg, tmp_path, monkeypatch, capsys):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps({"window_size": 30, "history": {}}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_anomaly_detector.py", "--history", str(history_path), threshold_arg, "1"],
    )
    assert benchmark_anomaly_detector.main() == 0
