"""Generic calibration-anchored hardware profile utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_calibration(path: str | None) -> dict[str, Any]:
    if not path:
        return default_calibration()
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def default_calibration() -> dict[str, Any]:
    return {
        "provider": "generic_hardware_snapshot",
        "backend_name": "example_device",
        "qubits": [
            {
                "id": 0,
                "t1_us": 135.2,
                "t2_us": 82.1,
                "readout_error": 0.021,
                "single_qubit_gate_error": 0.0011,
                "frequency_ghz": 4.97,
                "anharmonicity_ghz": -0.23,
            },
            {
                "id": 1,
                "t1_us": 118.7,
                "t2_us": 76.4,
                "readout_error": 0.028,
                "single_qubit_gate_error": 0.0014,
                "frequency_ghz": 5.08,
                "anharmonicity_ghz": -0.21,
            },
            {
                "id": 2,
                "t1_us": 141.5,
                "t2_us": 91.0,
                "readout_error": 0.019,
                "single_qubit_gate_error": 0.0009,
                "frequency_ghz": 5.14,
                "anharmonicity_ghz": -0.24,
            },
        ],
        "couplers": [
            {"pair": [0, 1], "cross_talk": 0.012},
            {"pair": [1, 2], "cross_talk": 0.017},
        ],
    }


def extract_noise_parameters(calibration: dict[str, Any]) -> dict[str, float]:
    qubits = calibration.get("qubits", [])
    couplers = calibration.get("couplers", [])

    def mean_or(default: float, values: list[float]) -> float:
        return float(np.mean(values)) if values else default

    params = {
        "mean_t1_us": mean_or(120.0, [float(q.get("t1_us", 0.0)) for q in qubits if q.get("t1_us") is not None]),
        "mean_t2_us": mean_or(90.0, [float(q.get("t2_us", 0.0)) for q in qubits if q.get("t2_us") is not None]),
        "mean_readout_error": mean_or(0.02, [float(q.get("readout_error", 0.0)) for q in qubits if q.get("readout_error") is not None]),
        "mean_gate_error": mean_or(0.001, [float(q.get("single_qubit_gate_error", 0.0)) for q in qubits if q.get("single_qubit_gate_error") is not None]),
        "mean_frequency_ghz": mean_or(5.0, [float(q.get("frequency_ghz", 0.0)) for q in qubits if q.get("frequency_ghz") is not None]),
        "mean_anharmonicity_ghz": mean_or(0.2, [abs(float(q.get("anharmonicity_ghz", 0.0))) for q in qubits if q.get("anharmonicity_ghz") is not None]),
        "mean_cross_talk": mean_or(0.01, [float(c.get("cross_talk", 0.0)) for c in couplers if c.get("cross_talk") is not None]),
    }
    params["derived_phase_drift_rate"] = float(1.0 / max(params["mean_t2_us"], 1e-9))
    params["derived_amplitude_decay_rate"] = float(1.0 / max(params["mean_t1_us"], 1e-9))
    params["derived_leakage_rate"] = float(
        params["mean_gate_error"] * (1.0 / max(params["mean_anharmonicity_ghz"], 1e-9)) * 0.01
    )
    return params


def simulate_noise_trajectory(
    params: dict[str, float],
    duration_seconds: float = 60.0,
    sample_rate_hz: float = 50.0,
    seed: int = 67,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_count = int(duration_seconds * sample_rate_hz)
    time_axis = np.arange(sample_count) / sample_rate_hz

    t1_s = max(params["mean_t1_us"] * 1e-6, 1e-3)
    t2_s = max(params["mean_t2_us"] * 1e-6, 1e-3)

    amplitude_decay = np.exp(-time_axis / t1_s)
    phase_decay = np.exp(-time_axis / t2_s)
    drift_noise = rng.normal(0.0, 0.0005, sample_count).cumsum()
    frequency_drift = params["mean_frequency_ghz"] + drift_noise

    depolarization = rng.binomial(1, min(params["mean_gate_error"] * 10.0, 0.25), sample_count)
    readout_flip = rng.binomial(1, min(params["mean_readout_error"], 0.5), sample_count)
    leakage = rng.binomial(1, min(params["derived_leakage_rate"] * 100.0, 0.2), sample_count)
    neighbor_bleed = rng.binomial(1, min(params["mean_cross_talk"] * 5.0, 0.2), sample_count)

    coherence_proxy = amplitude_decay * phase_decay
    coherence_proxy *= (1.0 - 0.15 * depolarization)
    coherence_proxy *= (1.0 - 0.10 * leakage)
    coherence_proxy *= (1.0 - 0.10 * neighbor_bleed)

    return {
        "schema_version": "generic.hardware_noise_profile.v1",
        "parameters": params,
        "summary": {
            "mean_coherence_proxy": float(np.mean(coherence_proxy)),
            "min_coherence_proxy": float(np.min(coherence_proxy)),
            "max_coherence_proxy": float(np.max(coherence_proxy)),
            "drift_span_ghz": float(np.max(frequency_drift) - np.min(frequency_drift)),
            "depolarization_events": int(np.sum(depolarization)),
            "readout_flip_events": int(np.sum(readout_flip)),
            "leakage_events": int(np.sum(leakage)),
            "neighbor_bleed_events": int(np.sum(neighbor_bleed)),
        },
        "time_series": {
            "time_s": time_axis.tolist(),
            "coherence_proxy": coherence_proxy.tolist(),
            "frequency_drift_ghz": frequency_drift.tolist(),
            "depolarization_events": depolarization.tolist(),
            "readout_flip_events": readout_flip.tolist(),
            "leakage_events": leakage.tolist(),
            "neighbor_bleed_events": neighbor_bleed.tolist(),
        },
    }
