"""Curation actions must survive channel changes and unprocessed icEMG."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from accusleepy.gui.main import AccuSleepWindow
from accusleepy.gui.manual_scoring import ManualScoringWindow
from accusleepy.manual_recording import load_manual_recording


@pytest.fixture
def multilfp_files(tmp_path):
    fs, n = 128, 24
    t = np.arange(fs * n) / fs
    icemg = np.repeat(np.linspace(0.1, 0.8, n * 2, dtype=np.float32), fs // 2)
    icemg[fs // 2 : fs] = np.nan
    table = pa.table(
        {
            "CA1": (10 * np.sin(t * 2 * np.pi * 8)).astype("float32"),
            "PFC": (100 * np.sin(t * 2 * np.pi * 3)).astype("float32"),
            "other": (30 * np.sin(t * 2 * np.pi * 12)).astype("float32"),
            "icemg": icemg,
        }
    )
    meta = {
        "schema_version": 1,
        "sampling_rate_hz": fs,
        "master_clock": {
            "coordinate": "nidq_reference_sample",
            "rate_hz": 12800,
            "first_sample": 123,
            "sample_step": 100,
        },
        "alignment": {"epoch_seconds": 1},
        "channels": {
            name: {"kind": "icemg" if name == "icemg" else "lfp"}
            for name in table.column_names
        },
    }
    p = tmp_path / "multilfp.parquet"
    pq.write_table(
        table.replace_schema_metadata({b"accusleepy": json.dumps(meta).encode()}), p
    )
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"brain_state": np.ones(n, dtype=int)}).to_csv(labels, index=False)
    return p, labels


@pytest.fixture
def multiwindow(qtbot, multilfp_files, sample_emg_filter):
    p, labels = multilfp_files
    view = load_manual_recording(p, 0, 5)
    w = ManualScoringWindow(
        view.channels["CA1"],
        view.emg,
        str(labels),
        np.ones(24, dtype=int),
        None,
        128,
        1,
        sample_emg_filter,
        recording=view,
    )
    qtbot.addWidget(
        w,
        before_close_func=lambda widget: setattr(
            widget, "last_saved_labels", widget.labels.copy()
        ),
    )
    return w


def test_stacked_plots_and_raw_icemg(multiwindow):
    w = multiwindow
    assert w.selected_channels == ["CA1", "PFC"]
    assert len(w.ui.upperfigure.channel_selectors) == 2
    ua, ub = w.ui.upperfigure.spectrogram_axes
    la, lb = w.ui.lowerfigure.lfp_axes
    assert ua.get_position().y0 > ub.get_position().y0
    assert la.get_position().y0 > lb.get_position().y0
    assert ua.get_xlim() == ub.get_xlim()
    assert la.get_xlim() == lb.get_xlim()
    assert np.isnan(w.upper_emg[0])
    np.testing.assert_allclose(
        w.upper_emg[1:], w.recording.emg.reshape(-1, 128).mean(1)[1:]
    )
    x, y = w.ui.lowerfigure.emg_line.get_data()
    np.testing.assert_array_equal(y, w.recording.emg[x])
    assert np.isnan(y).any()
    assert w.ui.lowerfigure.emg_line.get_drawstyle() == "steps-post"
    assert ua.images[0].get_clim() != ub.images[0].get_clim()
    assert not np.array_equal(ua.images[0].get_array(), ub.images[0].get_array())
    assert w.ui.upperfigure.canvas.axes[4].get_ylim() == (0, 1)
    np.testing.assert_array_equal(
        ua.images[0].get_array(), w.channel_cache["CA1"][1][:151]
    )
    np.testing.assert_array_equal(
        ub.images[0].get_array(), w.channel_cache["PFC"][1][:151]
    )


def test_selection_navigation_edit_undo_and_save(multiwindow, multilfp_files):
    w = multiwindow
    source = multilfp_files[1]
    original = source.read_bytes()
    w.ui.autoscroll.setChecked(False)
    # Jump on the second spectrum, preserving its half-epoch convention.
    w.click_to_jump(
        SimpleNamespace(xdata=10.5, inaxes=w.ui.upperfigure.spectrogram_axes[1])
    )
    assert w.epoch == 10
    w.modify_current_epoch_label(3)
    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in w.ui.lowerfigure.lfp_axes]
    upper_range = (w.upper_left_epoch, w.upper_right_epoch)
    # Actual dropdown signal, not just a direct method call.
    w.ui.upperfigure.channel_selectors[1].setCurrentText("other")
    assert w.selected_channels == ["CA1", "other"]
    assert (w.upper_left_epoch, w.upper_right_epoch) == upper_range
    assert w.epoch == 10 and w.labels[10] == 3 and w.history_index == 1
    assert limits == [
        (ax.get_xlim(), ax.get_ylim()) for ax in w.ui.lowerfigure.lfp_axes
    ]
    w.undo()
    assert w.labels[10] == 1
    w.redo()
    assert w.labels[10] == 3
    w.save()
    assert source.read_bytes() == original
    assert pd.read_csv(w.label_file).brain_state.iloc[10] == 3
    intervals = pd.read_csv(w.label_file.replace(".csv", ".intervals.csv"))
    assert intervals.start_reference_sample.iloc[10] == 123 + 12800 * 10
    assert intervals.end_reference_sample_exclusive.iloc[-1] == 123 + 12800 * 24


def test_second_lfp_mouse_zoom_drag_reset_and_rebuild(multiwindow):
    w = multiwindow
    ax = w.ui.lowerfigure.lfp_axes[1]
    w.scroll_zoom("lower", SimpleNamespace(button="up", inaxes=ax))
    factor = w.eeg_signal_scale_factor
    assert factor > 1
    w.lower_plot_click(
        SimpleNamespace(
            button=1, name="button_press_event", inaxes=ax, ydata=0.0, dblclick=False
        )
    )
    w.lower_plot_move(SimpleNamespace(button=1, inaxes=ax, ydata=0.2))
    assert w.eeg_signal_offset == 0.2
    w.lower_plot_click(
        SimpleNamespace(
            button=1, name="button_press_event", inaxes=ax, ydata=0.2, dblclick=True
        )
    )
    assert w.eeg_signal_offset == 0 and w.eeg_signal_scale_factor == 1
    w.update_epochs_shown("plus")
    assert len(w.ui.lowerfigure.lfp_axes) == 2
    w.lower_plot_click(SimpleNamespace(button=3, name="button_press_event"))
    assert w.ui.lowerfigure.extra_eeg_epoch_plot.get_visible()
    w.shift_epoch("right")
    assert w.ui.lowerfigure.extra_eeg_line.get_xdata().size > 0
    w.ui.upperfigure.channel_selectors[0].setCurrentText("PFC")
    w.ui.upperfigure.channel_selectors[0].setCurrentText("CA1")
    assert w.lfp_adjustments["CA1"] == [1, 0]


def test_main_uses_metadata_and_rejects_label_length(
    qtbot, monkeypatch, multilfp_files
):
    p, labels = multilfp_files
    main = AccuSleepWindow()
    qtbot.addWidget(main)
    main.recording_manager.current.recording_file = str(p)
    main.recording_manager.current.label_file = str(labels)
    captured = []
    monkeypatch.setattr(ManualScoringWindow, "exec", lambda self: captured.append(self))
    main.manual_scoring()
    assert captured[0].sampling_rate == 128 and captured[0].epoch_length == 1
    assert captured[0].label_file != str(labels)
    captured[0].close()
    pd.DataFrame({"brain_state": np.ones(23, dtype=int)}).to_csv(labels, index=False)
    main.manual_scoring()
    assert len(captured) == 1
    assert main.ui.manual_scoring_status.text() == "invalid label file"
    main.close()


def test_keyboard_roi_and_duplicate_selection(multiwindow, qtbot):
    from PySide6.QtCore import Qt

    w = multiwindow
    w.ui.autoscroll.setChecked(False)
    w.activateWindow()
    w.setFocus()
    qtbot.wait(10)
    qtbot.keyClick(w, Qt.Key.Key_3)
    assert w.labels[0] == 3
    qtbot.keyClick(w, Qt.Key.Key_Right)
    assert w.epoch == 1
    w.enter_label_roi_mode(2)
    w.roi_callback(SimpleNamespace(xdata=2), SimpleNamespace(xdata=4))
    assert np.all(w.labels[2:5] == 2)
    w.undo()
    assert np.all(w.labels[2:5] == 1)
    w.exit_label_roi_mode()
    w.ui.upperfigure.channel_selectors[1].setCurrentText("CA1")
    assert w.selected_channels == ["CA1", "CA1"]
    np.testing.assert_array_equal(
        w.ui.lowerfigure.eeg_line.get_ydata(),
        w.ui.lowerfigure.extra_eeg_line.get_ydata(),
    )


def test_channel_controls_below_time_zoom_and_independent_buttons(multiwindow, qtbot):
    from PySide6.QtCore import QPoint, Qt

    w = multiwindow
    w.ui.upperfigure.canvas.draw()
    qtbot.wait(10)
    for combo in w.ui.upperfigure.channel_selectors:
        assert combo.parent() is w
        assert (
            combo.mapTo(w, QPoint(0, 0)).y()
            > w.ui.xzoomreset.mapTo(w, QPoint(0, w.ui.xzoomreset.height())).y()
        )
    assert w.ui.upperfigure.canvas.figure.subplotpars.right == 0.95
    assert len(w.lfp_buttons) == 2
    w.active_lfp_slot = 1
    qtbot.mouseClick(w.lfp_buttons[0]["zoom_in"], Qt.MouseButton.LeftButton)
    assert w.lfp_adjustments["CA1"][0] > 1
    assert w.lfp_adjustments["PFC"] == [1, 0]
    first_scale = w.lfp_adjustments["CA1"][0]
    qtbot.mouseClick(w.lfp_buttons[1]["zoom_in"], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(w.lfp_buttons[1]["pan_up"], Qt.MouseButton.LeftButton)
    assert w.lfp_adjustments["PFC"][0] > 1
    assert w.lfp_adjustments["PFC"][1] > 0
    assert w.lfp_adjustments["CA1"] == [first_scale, 0]
    w.ui.upperfigure.channel_selectors[1].setCurrentText("other")
    assert w.lfp_control_labels[1].text() == "Ch2: other"
    qtbot.mouseClick(w.lfp_buttons[1]["pan_down"], Qt.MouseButton.LeftButton)
    assert w.lfp_adjustments["other"][1] < 0
    assert w.lfp_adjustments["PFC"][1] > 0


def test_selected_epoch_guides_align_and_blit_without_trails(multiwindow):
    w = multiwindow
    upper = w.ui.upperfigure
    assert len(upper.epoch_lines) == 4
    assert {ax.yaxis.label.get_fontsize() for ax in w.ui.lowerfigure.lfp_axes} == {
        w.ui.lowerfigure.canvas.axes[1].yaxis.label.get_fontsize()
    }
    for epoch in (0, 3, 8):
        w.epoch = epoch
        w.update_upper_marker()
        upper.draw_upper_marker()
        pixels = np.asarray(upper.canvas.buffer_rgba()).copy()
        upper.canvas.draw()
        redrawn = np.asarray(upper.canvas.buffer_rgba())
        # Compare plot interiors: neighboring tick-label antialiasing can overlap
        # an axes edge and is drawn in a different order by full-figure rendering.
        height = pixels.shape[0]
        for ax in upper.canvas.axes:
            if not ax.get_visible():
                continue
            left, bottom, right, top = map(int, ax.bbox.extents)
            region = (
                slice(height - top + 3, height - bottom - 3),
                slice(left + 3, right - 3),
            )
            np.testing.assert_array_equal(pixels[region], redrawn[region])
        positions = [
            line.axes.transData.transform((line.get_xdata()[0], 0))[0]
            for line in upper.epoch_lines
        ]
        np.testing.assert_allclose(positions, positions[0], atol=1e-8)
        assert all(line.get_linewidth() <= 0.5 for line in upper.epoch_lines)
    w.zoom_x("in")
    w.ui.upperfigure.channel_selectors[1].setCurrentText("other")
    positions = [
        line.axes.transData.transform((line.get_xdata()[0], 0))[0]
        for line in upper.epoch_lines
    ]
    np.testing.assert_allclose(positions, positions[0], atol=1e-8)


def test_view_state_roundtrip_and_saved_labels(qtbot, monkeypatch, multilfp_files):
    p, labels = multilfp_files
    main = AccuSleepWindow()
    qtbot.addWidget(main)
    main.recording_manager.current.recording_file = str(p)
    main.recording_manager.current.label_file = str(labels)
    captured = []
    monkeypatch.setattr(ManualScoringWindow, "exec", lambda self: captured.append(self))
    main.manual_scoring()
    first = captured[-1]
    first.ui.autoscroll.setChecked(False)
    first.click_to_jump(SimpleNamespace(xdata=12, inaxes=None))
    first.zoom_x("in")
    first.update_epochs_shown("plus")
    first.ui.upperfigure.channel_selectors[1].setCurrentText("other")
    first.lfp_buttons[1]["zoom_in"].click()
    first.update_spectrogram_brightness("brighter")
    first.modify_current_epoch_label(3)
    first.save()
    saved_label = first.label_file
    expected = (
        first.epoch,
        first.upper_left_epoch,
        first.upper_right_epoch,
        first.lower_left_epoch,
        first.epochs_to_show,
    )
    expected_clim = first.ui.upperfigure.extra_spec_ref.get_clim()
    expected_scale = first.lfp_adjustments["other"].copy()
    first.close()
    # Selecting the original source again resumes the last saved curation.
    main.manual_scoring()
    second = captured[-1]
    assert second.source_label_file == saved_label
    assert second.label_file == saved_label
    assert second.labels[12] == 3
    assert (
        second.epoch,
        second.upper_left_epoch,
        second.upper_right_epoch,
        second.lower_left_epoch,
        second.epochs_to_show,
    ) == expected
    assert second.selected_channels == ["CA1", "other"]
    assert second.lfp_adjustments["other"] == expected_scale
    assert second.ui.upperfigure.extra_spec_ref.get_clim() == expected_clim
    assert not second.autoscroll_state
    assert pd.read_csv(labels).brain_state.iloc[12] == 1
    # A clean close records subsequent navigation even without another label save.
    second.shift_epoch("right")
    next_epoch = second.epoch
    second.modify_current_epoch_label(2)
    second.save()
    assert second.label_file == saved_label
    saved_metadata = json.loads(
        p.parent.joinpath(saved_label.replace(".csv", ".metadata.json")).read_text()
    )
    assert saved_metadata["source_label_file"] == str(labels)
    assert saved_metadata["changed_epoch_indices"] == [12, next_epoch]
    assert pd.read_csv(saved_label).brain_state.iloc[next_epoch] == 2
    assert not list(p.parent.glob("*.curated.curated*.csv"))
    second.close()
    main.manual_scoring()
    third = captured[-1]
    assert third.epoch == next_epoch and third.labels[12] == 3
    third.close()
    main.close()
