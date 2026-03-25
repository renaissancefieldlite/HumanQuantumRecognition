"""Interaction-coupling runner for the recognition experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def run_simulation(duration_seconds: float, sample_rate_hz: float) -> dict[str, object]:
    time_axis = np.arange(int(duration_seconds * sample_rate_hz)) / sample_rate_hz
    human = build_agent_signal(time_axis, phase_shift=0.0, seed=67)
    aligned_quantum = build_agent_signal(time_axis, phase_shift=0.0, seed=68)
    misaligned_quantum = build_agent_signal(time_axis, phase_shift=np.pi / 2, seed=69)
    aligned_metrics = recognition_metrics(human, aligned_quantum)
    misaligned_metrics = recognition_metrics(human, misaligned_quantum)
    return {
        "mode": "simulation",
        "evidence_status": "simulation_baseline",
        "aligned": aligned_metrics,
        "misaligned": misaligned_metrics,
        "delta_recognition_score": aligned_metrics["recognition_score"] - misaligned_metrics["recognition_score"],
    }


def run_hardware_derived(calibration_path: str | None, duration_seconds: float, sample_rate_hz: float) -> dict[str, object]:
    calibration = load_calibration(calibration_path)
    params = extract_noise_parameters(calibration)
    report = simulate_noise_trajectory(params, duration_seconds=duration_seconds, sample_rate_hz=sample_rate_hz)
    time_axis = np.array(report["time_series"]["time_s"], dtype=float)
    human = build_agent_signal(time_axis, phase_shift=0.0, seed=67)
    coherence_proxy = np.array(report["time_series"]["coherence_proxy"], dtype=float)
    misaligned = build_agent_signal(time_axis, phase_shift=np.pi / 2, seed=69)
    aligned_metrics = recognition_metrics(human, coherence_proxy)
    misaligned_metrics = recognition_metrics(human, misaligned)
    return {
        "mode": "hardware-derived",
        "evidence_status": "hardware_derived_model",
        "noise_summary": report["summary"],
        "aligned": aligned_metrics,
        "misaligned": misaligned_metrics,
        "delta_recognition_score": aligned_metrics["recognition_score"] - misaligned_metrics["recognition_score"],
    }


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run bounded recognition experiments.")
    parser.add_argument("--mode", choices=["simulation", "hardware-derived"], default="simulation")
    parser.add_argument("--calibration")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.mode == "simulation":
        result = run_simulation(args.duration, args.sample_rate)
    else:
        result = run_hardware_derived(args.calibration, args.duration, args.sample_rate)

    result["schema_version"] = "rfl.human_quantum_recognition.v2"
    result["next_step"] = "Attach real session artifacts before treating the score as empirical recognition evidence."

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
