"""Generate small synthetic inputs; no animal recordings are distributed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def create_demo(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recording = output_dir / "demo.multilfp.parquet"
    labels = output_dir / "demo.labels.csv"
    if recording.exists() or labels.exists():
        raise FileExistsError("Demo output already exists; choose a new directory")
    fs, epochs = 250, 60
    rng = np.random.default_rng(10)
    t = np.arange(fs * epochs) / fs
    signals = {
        "LFP_A": (80 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 8, len(t))).astype(
            "float32"
        ),
        "LFP_B": (30 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 4, len(t))).astype(
            "float32"
        ),
        "LFP_C": (50 * np.sin(2 * np.pi * 12 * t) + rng.normal(0, 6, len(t))).astype(
            "float32"
        ),
    }
    icemg = (0.4 + 0.3 * np.sin(np.arange(epochs * 2) / 9)).astype("float32")
    icemg[20] = np.nan
    signals["icemg"] = np.repeat(icemg, fs // 2)
    metadata = {
        "schema_version": 1,
        "sampling_rate_hz": fs,
        "master_clock": {
            "coordinate": "nidq_reference_sample",
            "rate_hz": 10000,
            "first_sample": 0,
            "sample_step": 10000 // fs,
        },
        "alignment": {
            "epoch_seconds": 1,
            "n_epochs": epochs,
            "n_rows": fs * epochs,
            "end_reference_sample_exclusive": epochs * 10000,
        },
        "channels": {
            name: {"kind": "lfp", "unit": "uV"} for name in signals if name != "icemg"
        },
        "provenance": {
            "synthetic": True,
            "description": "Artificial reference clock and signals",
        },
    }
    metadata["channels"]["icemg"] = {
        "kind": "icemg",
        "unit": "dimensionless",
        "source_rate_hz": 2,
        "hold_samples": fs // 2,
    }
    table = pa.table(signals).replace_schema_metadata(
        {b"accusleepy": json.dumps(metadata).encode("utf-8")}
    )
    pq.write_table(table, recording, compression="zstd")
    states = np.repeat(np.array([1, 2, 3, -1]), 15)
    pd.DataFrame({"brain_state": states}).to_csv(labels, index=False)
    return recording, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=Path, help="Directory for the synthetic Parquet and labels"
    )
    args = parser.parse_args()
    for path in create_demo(args.output_dir):
        print(path)
