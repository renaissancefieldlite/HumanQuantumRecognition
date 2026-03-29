"""Interaction-coupling runner for the recognition experiment."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hardware_profile import extract_noise_parameters, load_calibration, simulate_noise_trajectory


def normalize(series: np.ndarray) -> np.ndarray:
    std = np.std(series)
    if std == 0:
        return np.zeros_like(series)
    return (series - np.mean(series)) / std


def build_agent_signal(time_axis: np.ndarray, phase_shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.sin(2 * np.pi * 0.67 * time_axis + phase_shift)
    signal += 0.1 * np.sin(2 * np.pi * 0.1 * time_axis + rng.uniform(0, 2 * np.pi))
    signal += 0.05 * rng.normal(size=len(time_axis))
    return normalize(signal)


def build_target_locked_agent_signal(
    target: np.ndarray,
    time_axis: np.ndarray,
    seed: int,
    *,
    misaligned: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_target = normalize(target)
    if misaligned:
        shift = max(1, len(base_target) // 4)
        base_target = np.roll(base_target, shift)
    low_freq = 0.1 * np.sin(2 * np.pi * 0.1 * time_axis + rng.uniform(0, 2 * np.pi))
    noise = 0.05 * rng.normal(size=len(time_axis))
    return normalize(base_target + low_freq + noise)


def recognition_metrics(signal_a: np.ndarray, signal_b: np.ndarray) -> dict[str, float]:
    a = normalize(signal_a)
    b = normalize(signal_b)
    correlation = float(np.corrcoef(a, b)[0, 1])
    lag = int(np.argmax(np.correlate(a, b, mode="full")) - (len(a) - 1))
    stability = float(1.0 / (1.0 + abs(lag)))
    score = float((correlation + stability) / 2.0)
    return {
        "correlation": correlation,
        "lag_samples": lag,
        "stability": stability,
        "recognition_score": score,
    }


def summary_stats(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


TARGET_ZERO_KEYS = {"00", "0x0"}
TARGET_THREE_KEYS = {"11", "0x3"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_manifest_capture_paths(path: Path) -> tuple[list[Path], dict[str, Any] | None]:
    payload = load_json(path)
    if payload.get("schema_version") != "rfl.capture_batch.v2":
        return [path], None
    capture_files = [Path(item).resolve() for item in payload.get("capture_files", [])]
    return capture_files, payload


def expand_capture_inputs(inputs: list[str]) -> tuple[list[Path], list[dict[str, Any]]]:
    capture_paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    for item in inputs:
        matches = glob.glob(item)
        candidates = [Path(match) for match in matches] if matches else [Path(item)]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.exists():
                continue
            manifest_paths, manifest = extract_manifest_capture_paths(resolved)
            if manifest:
                manifests.append(manifest)
                capture_paths.extend(manifest_paths)
            else:
                capture_paths.append(resolved)
    deduped = sorted({path.resolve() for path in capture_paths})
    return deduped, manifests


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def summarize_spacing(timestamps: list[datetime]) -> dict[str, float] | None:
    if len(timestamps) < 2:
        return None
    deltas = np.array(
        [
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:])
            if (later - earlier).total_seconds() > 0
        ],
        dtype=float,
    )
    if deltas.size == 0:
        return None
    return {
        "mean_seconds": float(np.mean(deltas)),
        "median_seconds": float(np.median(deltas)),
        "min_seconds": float(np.min(deltas)),
        "max_seconds": float(np.max(deltas)),
    }


def extract_counts(capture: dict[str, Any]) -> dict[str, int]:
    raw_result = capture.get("raw_result", capture)
    if raw_result.get("results"):
        first = raw_result["results"][0]
        return dict(first.get("data", {}).get("counts", {}))
    if raw_result.get("experiments"):
        first = raw_result["experiments"][0]
        return dict(first.get("measurement_counts", {}))
    return {}


def compute_capture_metrics(path: Path) -> dict[str, Any]:
    capture = load_json(path)
    counts = extract_counts(capture)
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError(f"{path} has no measurable counts.")

    zero_prob = sum(counts.get(key, 0) for key in TARGET_ZERO_KEYS) / total
    three_prob = sum(counts.get(key, 0) for key in TARGET_THREE_KEYS) / total
    target_subspace_probability = zero_prob + three_prob
    return {
        "path": str(path),
        "provider": capture.get("provider"),
        "backend_name": capture.get("backend_name"),
        "submitted_at_utc": capture.get("submitted_at_utc"),
        "created_at_utc": capture.get("created_at_utc"),
        "job_id": capture.get("job_id"),
        "shots": total,
        "zero_probability": zero_prob,
        "three_probability": three_prob,
        "target_subspace_probability": target_subspace_probability,
        "off_target_probability": max(0.0, 1.0 - target_subspace_probability),
        "bell_imbalance": abs(zero_prob - three_prob),
    }


def resolve_backend_sample_rate(
    items: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    capture_series_rate_hz: float | None,
) -> tuple[float, str, float | None, dict[str, float] | None]:
    if capture_series_rate_hz and capture_series_rate_hz > 0:
        return capture_series_rate_hz, "explicit_capture_series_rate_hz", None, None

    timestamps = sorted(
        [
            parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc"))
            for item in items
            if parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")) is not None
        ]
    )
    actual_spacing_summary = summarize_spacing(timestamps)
    if actual_spacing_summary:
        median_spacing = actual_spacing_summary["median_seconds"]
        return 1.0 / median_spacing, "capture_timestamp_spacing", median_spacing, actual_spacing_summary

    for manifest in manifests:
        selection_window_seconds = manifest.get("selection_window_seconds")
        if selection_window_seconds and selection_window_seconds > 0:
            return 1.0 / float(selection_window_seconds), "manifest_selection_window_seconds", float(selection_window_seconds), None

    return 1.0, "capture_index_fallback", None, None


def run_simulation(duration_seconds: float, sample_rate_hz: float, seed: int) -> dict[str, object]:
    time_axis = np.arange(int(duration_seconds * sample_rate_hz)) / sample_rate_hz
    human = build_agent_signal(time_axis, phase_shift=0.0, seed=seed)
    aligned_quantum = build_agent_signal(time_axis, phase_shift=0.0, seed=seed + 1)
    misaligned_quantum = build_agent_signal(time_axis, phase_shift=np.pi / 2, seed=seed + 2)
    aligned_metrics = recognition_metrics(human, aligned_quantum)
    misaligned_metrics = recognition_metrics(human, misaligned_quantum)
    return {
        "mode": "simulation",
        "evidence_status": "simulation_baseline",
        "aligned": aligned_metrics,
        "misaligned": misaligned_metrics,
        "delta_recognition_score": aligned_metrics["recognition_score"] - misaligned_metrics["recognition_score"],
    }


def run_hardware_derived(
    calibration_path: str | None,
    duration_seconds: float,
    sample_rate_hz: float,
    seed: int,
) -> dict[str, object]:
    calibration = load_calibration(calibration_path)
    params = extract_noise_parameters(calibration)
    report = simulate_noise_trajectory(params, duration_seconds=duration_seconds, sample_rate_hz=sample_rate_hz)
    time_axis = np.array(report["time_series"]["time_s"], dtype=float)
    coherence_proxy = np.array(report["time_series"]["coherence_proxy"], dtype=float)
    aligned = build_target_locked_agent_signal(coherence_proxy, time_axis, seed=seed, misaligned=False)
    misaligned = build_target_locked_agent_signal(coherence_proxy, time_axis, seed=seed + 10_000, misaligned=True)
    aligned_metrics = recognition_metrics(coherence_proxy, aligned)
    misaligned_metrics = recognition_metrics(coherence_proxy, misaligned)
    return {
        "mode": "hardware-derived",
        "evidence_status": "hardware_derived_model",
        "claim_under_test": "Whether the recognition score still separates aligned versus misaligned interaction-like controls when one trace is replaced by a calibration-anchored coherence proxy.",
        "noise_summary": report["summary"],
        "aligned": aligned_metrics,
        "misaligned": misaligned_metrics,
        "delta_recognition_score": aligned_metrics["recognition_score"] - misaligned_metrics["recognition_score"],
    }


def run_backend_capture(
    captures: list[str],
    capture_series_rate_hz: float | None,
    seed: int,
) -> dict[str, object]:
    capture_paths, manifests = expand_capture_inputs(captures)
    if not capture_paths:
        raise SystemExit("No backend capture files matched the provided inputs.")

    items = [compute_capture_metrics(path) for path in capture_paths]
    items.sort(
        key=lambda item: (
            parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")).timestamp()
            if parse_utc_timestamp(item.get("submitted_at_utc") or item.get("created_at_utc")) is not None
            else float("inf"),
            item["path"],
        )
    )

    sample_rate_hz, sample_rate_source, derived_spacing_seconds, actual_spacing_summary = resolve_backend_sample_rate(
        items,
        manifests,
        capture_series_rate_hz,
    )
    target = np.array([float(item["target_subspace_probability"]) for item in items], dtype=float)
    time_axis = np.arange(len(target), dtype=float) / sample_rate_hz
    aligned = build_target_locked_agent_signal(target, time_axis, seed=seed, misaligned=False)
    misaligned = build_target_locked_agent_signal(target, time_axis, seed=seed + 10_000, misaligned=True)
    aligned_metrics = recognition_metrics(target, aligned)
    misaligned_metrics = recognition_metrics(target, misaligned)

    manifest_context: dict[str, Any] = {}
    if manifests:
        first_manifest = manifests[0]
        manifest_context = {
            "label": first_manifest.get("label"),
            "condition": first_manifest.get("condition"),
            "session_mode": first_manifest.get("session_mode"),
            "session_reference": first_manifest.get("session_reference"),
            "selection_window_seconds": first_manifest.get("selection_window_seconds"),
            "submit_spacing_seconds": first_manifest.get("submit_spacing_seconds"),
            "completed_repeats": first_manifest.get("completed_repeats"),
        }

    return {
        "mode": "backend-capture",
        "evidence_status": "real_backend_session_synthetic_control",
        "claim_under_test": "Whether the recognition score still separates aligned versus misaligned interaction-like controls when the device-side trace is built from actual FEZ backend captures rather than simulation or a calibration-only proxy.",
        "provider": items[0].get("provider"),
        "backend_name": items[0].get("backend_name"),
        "capture_count": len(items),
        "sample_rate_hz": sample_rate_hz,
        "sample_rate_source": sample_rate_source,
        "derived_spacing_seconds": derived_spacing_seconds,
        "actual_spacing_summary": actual_spacing_summary,
        "trace_summary": {
            "mean_target_subspace_probability": float(np.mean(target)),
            "std_target_subspace_probability": float(np.std(target)),
            "mean_off_target_probability": float(
                np.mean(np.array([float(item["off_target_probability"]) for item in items], dtype=float))
            ),
            "mean_bell_imbalance": float(
                np.mean(np.array([float(item["bell_imbalance"]) for item in items], dtype=float))
            ),
        },
        "manifest_context": manifest_context,
        "aligned": aligned_metrics,
        "misaligned": misaligned_metrics,
        "delta_recognition_score": aligned_metrics["recognition_score"] - misaligned_metrics["recognition_score"],
        "captures": items,
    }


def run_repeated(
    mode: str,
    calibration_path: str | None,
    duration_seconds: float,
    sample_rate_hz: float,
    seed: int,
    repeats: int,
    captures: list[str] | None = None,
    capture_series_rate_hz: float | None = None,
) -> dict[str, object]:
    runs = []
    for offset in range(repeats):
        run_seed = seed + offset
        if mode == "simulation":
            run = run_simulation(duration_seconds, sample_rate_hz, run_seed)
        elif mode == "backend-capture":
            run = run_backend_capture(captures or [], capture_series_rate_hz, run_seed)
        else:
            run = run_hardware_derived(calibration_path, duration_seconds, sample_rate_hz, run_seed)
        run["seed"] = run_seed
        runs.append(run)

    aligned_scores = [float(run["aligned"]["recognition_score"]) for run in runs]
    misaligned_scores = [float(run["misaligned"]["recognition_score"]) for run in runs]
    deltas = [float(run["delta_recognition_score"]) for run in runs]

    result = {
        "mode": mode,
        "evidence_status": runs[0]["evidence_status"],
        "claim_under_test": runs[0].get("claim_under_test"),
        "repeat_count": repeats,
        "seed_start": seed,
        "repeat_summary": {
            "aligned_recognition_score": summary_stats(aligned_scores),
            "misaligned_recognition_score": summary_stats(misaligned_scores),
            "delta_recognition_score": summary_stats(deltas),
        },
        "runs": runs,
    }

    if mode == "hardware-derived":
        mean_coherence = [float(run["noise_summary"]["mean_coherence_proxy"]) for run in runs]
        result["repeat_summary"]["mean_coherence_proxy"] = summary_stats(mean_coherence)
    elif mode == "backend-capture":
        mean_target = [float(run["trace_summary"]["mean_target_subspace_probability"]) for run in runs]
        mean_imbalance = [float(run["trace_summary"]["mean_bell_imbalance"]) for run in runs]
        result["repeat_summary"]["mean_target_subspace_probability"] = summary_stats(mean_target)
        result["repeat_summary"]["mean_bell_imbalance"] = summary_stats(mean_imbalance)
        result["manifest_context"] = runs[0].get("manifest_context", {})

    return result


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run bounded recognition experiments.")
    parser.add_argument("--mode", choices=["simulation", "hardware-derived", "backend-capture"], default="simulation")
    parser.add_argument("--calibration")
    parser.add_argument("--captures", nargs="+", help="Capture file paths, globs, or batch manifests for backend-capture mode.")
    parser.add_argument(
        "--capture-series-rate-hz",
        type=float,
        help="Optional explicit sample rate for backend-capture mode when manifest/session cadence is unavailable.",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.repeats > 1:
        result = run_repeated(
            args.mode,
            args.calibration,
            args.duration,
            args.sample_rate,
            args.seed,
            args.repeats,
            args.captures,
            args.capture_series_rate_hz,
        )
    elif args.mode == "simulation":
        result = run_simulation(args.duration, args.sample_rate, args.seed)
    elif args.mode == "backend-capture":
        result = run_backend_capture(args.captures or [], args.capture_series_rate_hz, args.seed)
    else:
        result = run_hardware_derived(args.calibration, args.duration, args.sample_rate, args.seed)

    result["schema_version"] = "rfl.human_quantum_recognition.v3"
    if args.mode == "backend-capture":
        result["next_step"] = "Replace synthetic recognition controls with timestamped human-session traces aligned to the same FEZ session."
    else:
        result["next_step"] = "Attach real backend session artifacts before treating the score as empirical recognition evidence."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mode={result['mode']}")
        print(f"delta_recognition_score={result['delta_recognition_score']:.4f}")

    return result


if __name__ == "__main__":
    main()
