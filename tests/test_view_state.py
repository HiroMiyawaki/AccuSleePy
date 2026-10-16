"""Resume preferences cannot be carried over to a replaced recording."""

from accusleepy.view_state import load_view_state, save_view_state, state_path


def test_dataset_identity_and_corrupt_preferences(tmp_path):
    p = tmp_path / "recording.parquet"
    p.write_bytes(b"original")
    save_view_state(p, {"epoch": 12})
    assert load_view_state(p)["epoch"] == 12
    other = tmp_path / "other.parquet"
    other.write_bytes(b"original")
    assert load_view_state(other) is None
    p.write_bytes(b"replaced")
    assert load_view_state(p) is None
    state_path(p).write_text("not JSON")
    assert load_view_state(p) is None
