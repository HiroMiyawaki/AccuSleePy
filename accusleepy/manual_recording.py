"""Display-only recordings, with explicit timing and unprocessed icEMG."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class LFPChannels(Mapping):
    """Load only requested Parquet columns; the viewer owns its display cache."""

    def __init__(self, names, filename=None, frame=None):
        self.names = names
        self.filename = filename
        self.frame = frame

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        if name not in self.names:
            raise KeyError(name)
        if self.frame is not None:
            return self.frame[name].to_numpy()
        return pq.read_table(self.filename, columns=[name]).column(0).to_numpy()


@dataclass
class ManualRecording:
    channels: Mapping
    emg: np.ndarray
    mode: str
    metadata: dict
    strict: bool
    sampling_rate: float
    epoch_length: float
    source_file: str = ""


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def load_manual_recording(filename, sampling_rate, epoch_length):
    """Read signal names and timing without treating auxiliary columns as labels."""
    metadata = {}
    if Path(filename).suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(filename)
        schema = parquet.schema_arrow
        names = schema.names
        raw = (schema.metadata or {}).get(b"accusleepy")
        if raw is not None:
            metadata = json.loads(raw)
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                raise ValueError("Unsupported accusleepy metadata schema")
        numeric = all(
            pa.types.is_integer(f.type) or pa.types.is_floating(f.type) for f in schema
        )
        channels = LFPChannels(
            [n for n in names if n not in ("emg", "icemg")], filename=filename
        )
        n_rows = parquet.metadata.num_rows
        frame = None
    elif Path(filename).suffix.lower() == ".csv":
        frame = pd.read_csv(filename)
        names = list(frame.columns)
        numeric = all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes)
        channels = LFPChannels(
            [n for n in names if n not in ("emg", "icemg")], frame=frame
        )
        n_rows = len(frame)
    else:
        raise ValueError("Recording must be CSV or Parquet")
    if len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("Signal column names must be nonempty and unique")
    if not numeric or not channels:
        raise ValueError(
            "All signal columns must be numeric; at least one LFP is required"
        )
    mode = "emg" if "emg" in names else "icemg"
    if mode not in names:
        raise ValueError("Recording requires an emg or icemg column")
    strict = bool(metadata) or mode == "icemg" or list(channels) != ["eeg"]
    sampling_rate = metadata.get("sampling_rate_hz", sampling_rate)
    alignment = metadata.get("alignment", {})
    epoch_length = alignment.get("epoch_seconds", epoch_length)
    _positive_number(sampling_rate, "sampling_rate_hz")
    _positive_number(epoch_length, "epoch_seconds")
    samples = sampling_rate * epoch_length
    if strict and not float(samples).is_integer():
        raise ValueError("Each epoch must contain an integer number of samples")
    if strict and (n_rows == 0 or n_rows % round(samples)):
        raise ValueError(
            "Recording length must be an exact number of epochs; no padding applied"
        )
    if metadata:
        clock = metadata.get("master_clock", {})
        for key in ("rate_hz", "first_sample", "sample_step"):
            if type(clock.get(key)) is not int:
                raise ValueError(f"master_clock.{key} must be an integer")
        if clock.get("coordinate") != "nidq_reference_sample":
            raise ValueError("Unsupported master clock coordinate")
        if clock["rate_hz"] <= 0 or clock["sample_step"] <= 0:
            raise ValueError("Master clock rate and step must be positive")
        if clock["rate_hz"] != sampling_rate * clock["sample_step"]:
            raise ValueError("Master clock step conflicts with sampling rate")
        if not float(epoch_length * clock["rate_hz"]).is_integer():
            raise ValueError("Epoch boundaries must be integer master samples")
        for key, expected in {
            "n_rows": n_rows,
            "n_epochs": n_rows // round(samples),
            "end_reference_sample_exclusive": clock["first_sample"]
            + n_rows * clock["sample_step"],
        }.items():
            if key in alignment and alignment[key] != expected:
                raise ValueError(f"alignment.{key} does not match recording")
        descriptions = metadata.get("channels", {})
        if set(descriptions) != set(names):
            raise ValueError("Channel metadata must describe every signal column")
        if descriptions[mode].get("kind") != mode:
            raise ValueError(f"{mode} metadata kind does not match its column")
        for name in channels:
            if descriptions[name].get("kind") != "lfp":
                raise ValueError(f"{name} is not an LFP channel")
    emg = (
        frame[mode].to_numpy()
        if frame is not None
        else pq.read_table(filename, columns=[mode]).column(0).to_numpy()
    )
    if np.isinf(emg).any():
        raise ValueError("EMG/icEMG contains infinity")
    if mode == "emg" and strict and not np.isfinite(emg).all():
        raise ValueError(
            "EMG contains missing values; filtered EMG requires finite samples"
        )
    return ManualRecording(
        channels,
        emg,
        mode,
        metadata,
        strict,
        sampling_rate,
        epoch_length,
        str(filename),
    )


def is_curated_label_file(filename):
    """Recognize normal and numbered curation outputs, including older nested names."""
    return re.search(r"\.curated(?:-\d+)?\.csv$", Path(filename).name) is not None


def curated_label_path(filename):
    """Choose a fresh output without overwriting the source or an older curation."""
    path = Path(filename)
    stem = path.stem
    candidate = path.with_name(stem + ".curated.csv")
    i = 2
    while any(
        candidate.with_suffix(suffix).exists()
        for suffix in (".csv", ".intervals.csv", ".metadata.json")
    ):
        candidate = path.with_name(f"{stem}.curated-{i}.csv")
        i += 1
    return str(candidate)


def save_curation_metadata(
    filename,
    labels,
    metadata,
    epoch_length,
    source_label_file,
    brain_state_codes=None,
    initial_labels=None,
    source_recording_file=None,
):
    """Write exact integer intervals separately from the compatible brain_state CSV."""
    path = Path(filename)
    session_input = str(source_label_file)
    change_basis = "session_start"
    if Path(source_label_file).resolve() == path.resolve():
        # Updating a curation must not replace its original provenance with a
        # self-reference. Recompute cumulative changes from the original CSV.
        try:
            previous = json.loads(
                path.with_suffix(".metadata.json").read_text(encoding="utf-8")
            )
            original = previous.get("source_label_file")
            if original and Path(original).resolve() != path.resolve():
                source_label_file = original
                baseline = pd.read_csv(original, usecols=["brain_state"])[
                    "brain_state"
                ].to_numpy()
                if len(baseline) == len(labels):
                    initial_labels = baseline
                    change_basis = "source_label_file"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    else:
        change_basis = "source_label_file"
    clock = metadata.get("master_clock")
    provenance = {
        "source_label_file": str(source_label_file),
        "session_input_label_file": session_input,
        "changed_epoch_indices_basis": change_basis,
        "source_recording_file": source_recording_file,
        "epoch_seconds": epoch_length,
        "source_recording_metadata": metadata,
        "brain_state_codes": brain_state_codes
        or {"REM": 1, "Wake": 2, "NREM": 3, "undefined": -1},
        "original_flags": "Preserved in source files; not modified by manual labels",
    }
    if initial_labels is not None:
        provenance["changed_epoch_indices"] = np.flatnonzero(
            labels != initial_labels
        ).tolist()
    if clock:
        step = round(epoch_length * clock["rate_hz"])
        idx = np.arange(len(labels), dtype=np.int64)
        start = clock["first_sample"] + step * idx
        pd.DataFrame(
            {
                "epoch_index": idx,
                "start_reference_sample": start,
                "end_reference_sample_exclusive": start + step,
                "brain_state": labels,
            }
        ).to_csv(path.with_suffix(".intervals.csv"), index=False)
        provenance["master_clock"] = clock
    path.with_suffix(".metadata.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
