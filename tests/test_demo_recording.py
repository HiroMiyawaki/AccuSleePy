"""The documented demo CLI must work outside the source directory."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from accusleepy.manual_recording import load_manual_recording


def test_demo_cli_and_input_contract(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "examples" / "create_demo_recording.py"
    )
    output = tmp_path / "synthetic"
    subprocess.run([sys.executable, str(script), str(output)], cwd=tmp_path, check=True)
    p = output / "demo.multilfp.parquet"
    f = pq.ParquetFile(p)
    meta = json.loads(f.schema_arrow.metadata[b"accusleepy"])
    view = load_manual_recording(p, 0, 1)
    assert view.sampling_rate == 250 and view.epoch_length == 1
    assert list(view.channels) == ["LFP_A", "LFP_B", "LFP_C"]
    assert f.metadata.num_rows == 15000
    assert meta["provenance"]["synthetic"]
    assert view.mode == "icemg"
    assert np.isnan(view.emg).sum() == 125
    assert pd.read_csv(output / "demo.labels.csv").shape == (60, 1)
    before = p.read_bytes()
    again = subprocess.run(
        [sys.executable, str(script), str(output)], cwd=tmp_path, capture_output=True
    )
    assert again.returncode != 0
    assert p.read_bytes() == before
