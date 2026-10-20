from typing import Dict, Any, List
import numpy as np
import pandas as pd


def generate_fraud_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic financial fraud detection dataset with class imbalance."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * 0.035)  # 3.5% fraud rate
    n_legit = n_samples - n_fraud

    # Legit transactions
    amount_legit = rng.exponential(scale=65.0, size=n_legit) + 2.0
    hour_legit = rng.integers(6, 23, size=n_legit)
    dist_legit = rng.gamma(shape=2.0, scale=3.0, size=n_legit)
    v1_legit = rng.normal(0.0, 1.0, size=n_legit)
    v2_legit = rng.normal(0.0, 1.0, size=n_legit)
    v3_legit = rng.normal(0.0, 1.0, size=n_legit)
    v4_legit = rng.normal(0.0, 1.0, size=n_legit)
    v5_legit = rng.normal(0.0, 1.0, size=n_legit)
    channel_legit = rng.choice(["POS", "ONLINE", "ATM", "MOBILE"], p=[0.4, 0.35, 0.15, 0.1], size=n_legit)
    device_legit = rng.choice(["iOS", "Android", "Web", "Windows"], p=[0.45, 0.35, 0.15, 0.05], size=n_legit)

    # Fraud transactions
    amount_fraud = rng.exponential(scale=380.0, size=n_fraud) + 25.0
    hour_fraud = rng.choice([1, 2, 3, 4, 23, 0], size=n_fraud)
    dist_fraud = rng.gamma(shape=5.0, scale=12.0, size=n_fraud)
    v1_fraud = rng.normal(-2.5, 1.5, size=n_fraud)
    v2_fraud = rng.normal(3.0, 1.2, size=n_fraud)
    v3_fraud = rng.normal(-1.8, 1.0, size=n_fraud)
    v4_fraud = rng.normal(2.1, 1.3, size=n_fraud)
    v5_fraud = rng.normal(-0.5, 1.0, size=n_fraud)
    channel_fraud = rng.choice(["POS", "ONLINE", "ATM", "MOBILE"], p=[0.1, 0.7, 0.15, 0.05], size=n_fraud)
    device_fraud = rng.choice(["iOS", "Android", "Web", "Windows"], p=[0.2, 0.3, 0.4, 0.1], size=n_fraud)

    # Concatenate
    amount = np.concatenate([amount_legit, amount_fraud])
    hour = np.concatenate([hour_legit, hour_fraud])
    dist = np.concatenate([dist_legit, dist_fraud])
    v1 = np.concatenate([v1_legit, v1_fraud])
    v2 = np.concatenate([v2_legit, v2_fraud])
    v3 = np.concatenate([v3_legit, v3_fraud])
    v4 = np.concatenate([v4_legit, v4_fraud])
    v5 = np.concatenate([v5_legit, v5_fraud])
    channel = np.concatenate([channel_legit, channel_fraud])
    device = np.concatenate([device_legit, device_fraud])
    target = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Shuffle
    indices = rng.permutation(n_samples)
    df = pd.DataFrame({
        "transaction_amount": np.round(amount[indices], 2),
        "transaction_hour": hour[indices],
        "merchant_distance_km": np.round(dist[indices], 2),
        "v1_signal": np.round(v1[indices], 4),
        "v2_signal": np.round(v2[indices], 4),
        "v3_signal": np.round(v3[indices], 4),
        "v4_signal": np.round(v4[indices], 4),
        "v5_signal": np.round(v5[indices], 4),
        "payment_channel": channel[indices],
        "device_platform": device[indices],
        "is_fraud": target[indices],
    })
    return df


def generate_iot_telemetry_dataset(n_samples: int = 4000, seed: int = 101) -> pd.DataFrame:
    """Generates synthetic edge IoT sensor dataset for predictive maintenance."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(68.0, 12.0, size=n_samples)
    vibration = rng.exponential(scale=1.5, size=n_samples) + 0.1
    pressure = rng.normal(101.3, 8.5, size=n_samples)
    rpm = rng.normal(2400.0, 250.0, size=n_samples)
    operating_hours = rng.uniform(50.0, 5000.0, size=n_samples)
    model_type = rng.choice(["Sensor-A10", "Sensor-B20", "Sensor-ProX"], size=n_samples)

    # Failure risk score (synthetic target)
    risk_logits = (
        0.05 * (temp - 70.0) +
        0.8 * (vibration - 1.5) +
        0.02 * (operating_hours / 1000.0) -
        0.001 * (rpm - 2400.0) +
        rng.normal(0, 0.5, size=n_samples)
    )
    failure = (risk_logits > 1.2).astype(int)

    df = pd.DataFrame({
        "temperature_c": np.round(temp, 1),
        "vibration_rms": np.round(vibration, 3),
        "pressure_kpa": np.round(pressure, 2),
        "rpm": np.round(rpm, 0),
        "operating_hours": np.round(operating_hours, 1),
        "hardware_tier": model_type,
        "maintenance_required": failure
    })
    return df


SAMPLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "fraud_detection_benchmark": {
        "name": "Financial Fraud Detection (Imbalanced)",
        "description": "5,000 credit card transaction events with 3.5% positive fraud rate and subtle multi-variate feature drift.",
        "target_column": "is_fraud",
        "task_type": "classification",
        "generator": generate_fraud_dataset
    },
    "edge_iot_predictive_maintenance": {
        "name": "Edge IoT Sensor Anomaly",
        "description": "4,000 industrial sensor telemetry readings for predictive maintenance under tight edge-compute constraints.",
        "target_column": "maintenance_required",
        "task_type": "classification",
        "generator": generate_iot_telemetry_dataset
    }
}


def list_sample_datasets() -> List[Dict[str, Any]]:
    return [
        {
            "id": k,
            "name": v["name"],
            "description": v["description"],
            "target_column": v["target_column"],
            "task_type": v["task_type"]
        }
        for k, v in SAMPLE_CATALOG.items()
    ]


def get_sample_dataset(sample_id: str) -> pd.DataFrame:
    if sample_id not in SAMPLE_CATALOG:
        raise ValueError(f"Unknown sample dataset '{sample_id}'. Available: {list(SAMPLE_CATALOG.keys())}")
    return SAMPLE_CATALOG[sample_id]["generator"]()
