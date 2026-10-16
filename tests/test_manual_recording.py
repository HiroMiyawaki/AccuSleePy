"""New input timing must never be inferred by padding or resampling."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from accusleepy.manual_recording import (
    curated_label_path,
    is_curated_label_file,
    load_manual_recording,
    save_curation_metadata,
)
from accusleepy.signal_processing import create_spectrogram


def test_modes_column_order_and_lazy_lfp(tmp_path):
    p = tmp_path / "signals.parquet"
    pq.write_table(
        pa.table(
            {
                "PFC": [1.0, 2.0],
                "CA1": [3.0, 4.0],
                "icemg": [0.2, 0.3],
                "emg": [10.0, 11.0],
            }
        ),
        p,
    )
    view = load_manual_recording(p, 2, 1)
    assert view.mode == "emg"
    assert list(view.channels) == ["PFC", "CA1"]
    np.testing.assert_array_equal(view.emg, [10, 11])
    np.testing.assert_array_equal(view.channels["CA1"], [3, 4])


def test_icemg_nan_and_no_padding(tmp_path):
    p = tmp_path / "signals.parquet"
    pq.write_table(
        pa.table({"eeg": [1.0, 2.0, 3.0], "icemg": [0.2, float("nan"), 0.4]}), p
    )
    with pytest.raises(ValueError, match="exact number of epochs"):
        load_manual_recording(p, 2, 1)
    view = load_manual_recording(p, 3, 1)
    assert view.mode == "icemg" and view.strict
    assert np.isnan(view.emg[1])
    assert view.emg[0] == 0.2


@pytest.mark.parametrize(
    "columns",
    [
        {"CA1": [1.0, 2.0]},
        {"icemg": [0.2, 0.3]},
        {"CA1": ["x", "y"], "icemg": [0.2, 0.3]},
    ],
)
def test_invalid_signals_rejected(tmp_path, columns):
    p = tmp_path / "signals.parquet"
    pq.write_table(pa.table(columns), p)
    with pytest.raises(ValueError):
        load_manual_recording(p, 2, 1)


def test_metadata_conflict_rejected(tmp_path):
    p = tmp_path / "signals.parquet"
    metadata = {
        "schema_version": 1,
        "sampling_rate_hz": 1250,
        "master_clock": {
            "coordinate": "nidq_reference_sample",
            "rate_hz": 10000,
            "first_sample": 123,
            "sample_step": 9,
        },
    }
    table = pa.table({"eeg": np.zeros(1250), "icemg": np.zeros(1250)})
    pq.write_table(
        table.replace_schema_metadata({b"accusleepy": json.dumps(metadata).encode()}), p
    )
    with pytest.raises(ValueError, match="conflicts"):
        load_manual_recording(p, 1, 1)


def test_integer_interval_export_and_fresh_path(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text("brain_state\n1\n2\n3\n")
    output = curated_label_path(source)
    assert output != str(source)
    metadata = {
        "master_clock": {
            "coordinate": "nidq_reference_sample",
            "rate_hz": 10000,
            "first_sample": 123,
            "sample_step": 8,
        }
    }
    save_curation_metadata(output, np.array([1, 3, -1]), metadata, 1, source)
    import pandas as pd

    intervals = pd.read_csv(str(output).replace(".csv", ".intervals.csv"))
    assert intervals.start_reference_sample.tolist() == [123, 10123, 20123]
    assert intervals.end_reference_sample_exclusive.tolist() == [10123, 20123, 30123]
    assert source.read_text() == "brain_state\n1\n2\n3\n"


def test_chunked_spectrum_preserves_windows_and_values():
    signal = np.random.default_rng(1).normal(size=128 * 19)
    full, f = create_spectrogram(signal, 128, 1)
    chunked, cf = create_spectrogram(signal, 128, 1, chunk_epochs=7)
    np.testing.assert_array_equal(cf, f)
    np.testing.assert_allclose(chunked, full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("labels.csv", False),
        ("labels.curated.csv", True),
        ("labels.curated-2.csv", True),
        ("labels.curated.curated.csv", True),
        ("labels.curated.intervals.csv", False),
    ],
)
def test_curated_output_recognition(name, expected):
    assert is_curated_label_file(name) is expected
