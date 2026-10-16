# Multi-LFP manual-curation input contract (version 1)

This is a display/manual-editing format. Automated classification/training still
expects the legacy `eeg` and `emg` format. See the
[operation guide](../accusleepy/gui/text/manual_scoring_guide.md) for GUI behavior.

## Signal table

Use Parquet with one numeric column per signal, one row per synchronized sample.
Column names must be nonempty and unique. The example below uses arbitrary names:

| Column | Meaning |
|---|---|
| `LFP_A`, `LFP_B`, … | One or more synchronized LFP signals |
| `icemg` | Direct icEMG values held until the next source update |

All columns except reserved `emg`/`icemg` are LFP candidates, in file column order.
The first two are shown by default; one LFP produces one display slot. `emg`
selects legacy bandpass/RMS/log processing and takes precedence when both reserved
columns exist. Without `emg`, `icemg` selects unfiltered display. At least one
LFP and one EMG-family column are required.

Use float32 for compact recordings. Store LFPs in a consistent physical unit
(e.g. already converted to µV); do not apply a conversion twice. Units and source
information are metadata. Infinity is rejected. LFP NaNs are currently rejected
rather than repaired. icEMG NaNs are kept as gaps; one NaN in an epoch makes its
upper overview mean NaN. The lower icEMG trace uses original values, not RMS,
logarithms, centering, interpolation or normalization.

For a 2 Hz icEMG source in a 1,250 Hz table, hold each source value for 625 rows,
including NaNs. The period must reflect the actual source/table sampling rates.
The exporter is responsible for synchronization and selecting the intended time
interval. The viewer does not re-align, stretch, resample or pad the new format.

Do not add timestamps, row numbers, pandas index, labels, confidence or validity
flags as table columns: they would be interpreted as LFPs. Use metadata or separate
files. Write pandas tables with `index=False`. Preserve original labels, flags and
source data independently; do not manufacture confidence from signal features.

## Parquet metadata

Store a UTF-8 JSON object under Arrow schema metadata key `b"accusleepy"`.
A complete synthetic example (250 Hz table, one-second epochs, 60 seconds) is:

```json
{
  "schema_version": 1,
  "sampling_rate_hz": 250,
  "master_clock": {
    "coordinate": "nidq_reference_sample",
    "rate_hz": 10000,
    "first_sample": 0,
    "sample_step": 40
  },
  "alignment": {
    "epoch_seconds": 1,
    "n_rows": 15000,
    "n_epochs": 60,
    "end_reference_sample_exclusive": 600000
  },
  "channels": {
    "LFP_A": {"kind": "lfp", "unit": "uV"},
    "LFP_B": {"kind": "lfp", "unit": "uV"},
    "icemg": {
      "kind": "icemg",
      "unit": "dimensionless",
      "source_rate_hz": 2,
      "hold_samples": 125
    }
  }
}
```

Replace every example value with actual acquisition/export metadata. No real
session is implied by this example's reference clock. Currently the supported
coordinate identifier is `nidq_reference_sample`. Other coordinate systems need
an explicit future schema/adapter; do not relabel a different clock as NIDQ.

`schema_version`, `sampling_rate_hz`, `master_clock` and descriptions of every
column in `channels` are required when this metadata is supplied. Each description
must have the correct `kind` (`lfp`, `emg` or `icemg`). Unit and source fields carry
provenance; no unit conversion is performed by the viewer.
`alignment.epoch_seconds` supplies the epoch duration. If omitted, the GUI's
selected duration is used. Optional row/epoch/end counts are checked when present.
Other provenance fields can be retained in the JSON. Windows source paths can be
provenance strings; the viewer reads the selected local Parquet, not those paths.

Without metadata, numeric CSV/Parquet signals can still be displayed using the
GUI sampling rate and epoch length, but no exact master-clock interval sidecar
can be generated. Legacy `eeg`/`emg` recordings without this metadata retain their
original preprocessing and save behavior.

## Integer time coordinates and labels

For output row `k`, `master_sample = first_sample + sample_step * k`.
The clock rate and step are positive integers and must agree with the table rate.
Each epoch must contain an integer number of rows and integer master samples.
For zero-based epoch `i`, let `D = epoch_seconds * rate_hz`:

- start: `first_sample + D*i`;
- end (exclusive): `first_sample + D*(i+1)`;
- center (for comparison to center-indexed scores): `first_sample + D*(i+0.5)`.

Drawing/zoom/decimation do not change those label intervals. A label CSV must
have exactly one `brain_state` per epoch. The default configured codes are REM=1,
Wake=2, NREM=3, undefined=-1. Codes are checked against the current brain-state
configuration; no source-state conversion is guessed. An optional
`confidence_score` must match the label count. Keep invalid/uncertain flags in
their original files; editing a label does not erase those source flags.

New-format saves create a compatible curated CSV, a provenance JSON and, with
master metadata, an integer `.intervals.csv`. Starting from an original label CSV
creates a fresh curated output; reopening a curated CSV updates that same output
and sidecars. The provenance records the basis of changed-epoch indices. Copy
all three files when transferring manual scores.

## Exporter checks

Verify row order/count, column types/names, full or chunked source comparisons,
master endpoints, NaN counts and held-value boundaries. Re-read the JSON metadata
from the actual Parquet, and record the output checksum. Never silently truncate,
fill or shift an input that does not match. The synthetic generator in
`examples/create_demo_recording.py` provides a small working example.
