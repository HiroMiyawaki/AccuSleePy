"""AccuSleePy manual scoring GUI.

Icon sources:
    Arkinasi, https://www.flaticon.com/authors/arkinasi
    kendis lasman, https://www.flaticon.com/packs/ui-79
"""

import copy
import os
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import (
    QKeyCombination,
    QRect,
    Qt,
    QUrl,
)
from PySide6.QtGui import (
    QCloseEvent,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from accusleepy.constants import UNDEFINED_LABEL
from accusleepy.fileio import EMGFilter, load_config, save_labels
from accusleepy.gui.mplwidget import resample_x_ticks
from accusleepy.gui.viewer_window import Ui_ViewerWindow
from accusleepy.manual_recording import (
    curated_label_path,
    is_curated_label_file,
    save_curation_metadata,
)
from accusleepy.signal_processing import create_spectrogram, get_emg_power
from accusleepy.view_state import load_view_state, save_view_state

# colormap for displaying brain state labels
# the first entry represents the "undefined" state
# the other entries are the digits in "keyboard" order (1234567890)
LABEL_CMAP = np.concatenate(
    [np.array([[0, 0, 0, 0]]), plt.colormaps["tab10"](range(10))], axis=0
)
# relative path to user manual text file
USER_MANUAL_FILE = os.path.normpath(r"text/manual_scoring_guide.md")

# constants used by callback functions
# label formats
DISPLAY_FORMAT = "display"
DIGIT_FORMAT = "digit"
# offset changes
OFFSET_UP = "up"
OFFSET_DOWN = "down"
OFFSET_INCREMENTS = {OFFSET_UP: 0.02, OFFSET_DOWN: -0.02}
# changes to number of epochs
DIRECTION_PLUS = "plus"
DIRECTION_MINUS = "minus"
# changes to selected epoch
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
# zoom directions
ZOOM_IN = "in"
ZOOM_OUT = "out"
ZOOM_RESET = "reset"
SIGNAL_ZOOM_FACTORS = {ZOOM_IN: 1.08, ZOOM_OUT: 0.95}
# signal names
EEG_SIGNAL = "eeg"
EMG_SIGNAL = "emg"
# spectrogram color changes
BRIGHTER = "brighter"
DIMMER = "dimmer"
# next epoch target
DIFFERENT_STATE = "different"
UNDEFINED_STATE = "undefined"
# how far from the edge of the upper plot the marker should be
# before starting to scroll again - must be in (0, 0.5)
SCROLL_BOUNDARY = 0.35
# max number of sequential undo actions allowed
UNDO_LIMIT = 1000
# brightness scaling factors for the spectrogram
BRIGHTER_SCALE_FACTOR = 0.96
DIMMER_SCALE_FACTOR = 1.07
# zoom factor for upper plots - larger values = bigger changes
ZOOM_FACTOR = 0.1
# rate limit for zoom events triggered by scrolling
MAX_SCROLL_EVENTS_PER_SEC = 24
# tip to show when EEG/EMG plot moves a lot
RESET_TIP = "click plot 2x to reset"


@dataclass
class StateChange:
    """Information about an event when brain state labels were changed"""

    previous_labels: np.array  # old brain state labels
    new_labels: np.array  # new brain state labels
    epoch: int  # first epoch affected


class ManualScoringWindow(QDialog):
    """AccuSleePy manual scoring GUI"""

    def __init__(
        self,
        eeg: np.array,
        emg: np.array,
        label_file: str,
        labels: np.array,
        confidence_scores: np.array,
        sampling_rate: int | float,
        epoch_length: int | float,
        emg_filter: EMGFilter,
        recording=None,
        first_eeg=None,
        recording_file="",
    ):
        """Initialize the manual scoring window

        :param eeg: EEG signal
        :param emg: EMG signal
        :param label_file: filename for labels
        :param labels: brain state labels
        :param confidence_scores: confidence scores
        :param sampling_rate: sampling rate, in Hz
        :param epoch_length: epoch length, in seconds
        :param emg_filter: EMG filter parameters
        """
        super().__init__()

        self.recording = recording
        self.recording_file = recording_file or (
            recording.source_file if recording else ""
        )
        self.saved_view_state = load_view_state(self.recording_file)
        self.resume_label_source = label_file
        if self.saved_view_state and label_file in (
            self.saved_view_state.get("label_source"),
            self.saved_view_state.get("label_file"),
        ):
            self.resume_label_source = self.saved_view_state.get(
                "label_source", label_file
            )
        self.source_label_file = label_file
        self.curation_saved = is_curated_label_file(label_file)
        self.label_file = (
            curated_label_path(label_file)
            if recording and not self.curation_saved
            else label_file
        )
        self.emg_mode = recording.mode if recording else "emg"
        self.channel_cache = {}
        self.spectrum_cache = {}
        self.channel_clims = {}
        self.lfp_adjustments = {}
        self.active_lfp_slot = 0
        self.selected_channels = list(recording.channels)[:2] if recording else []
        self.eeg = eeg
        self.emg = emg
        self.labels = labels
        self.initial_labels = labels.copy()
        self.confidence_scores = confidence_scores
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.emg_filter = emg_filter

        self.n_epochs = len(self.labels)

        # initialize the UI
        self.ui = Ui_ViewerWindow()
        self.ui.setupUi(self)
        self.emg_control_label = QLabel(
            "icEMG" if self.emg_mode == "icemg" else "EMG", self
        )
        emg_index = next(
            i
            for i in range(self.ui.eeg_emg_buttons.count())
            if self.ui.eeg_emg_buttons.itemAt(i).layout() == self.ui.emgbuttons
        )
        self.ui.eeg_emg_buttons.insertWidget(emg_index, self.emg_control_label)
        self.setWindowTitle("AccuSleePy manual scoring window")

        config = load_config()
        self.brain_state_set = config.brain_state_set
        self.epochs_to_show = min(config.epochs_to_show, self.n_epochs)
        if self.epochs_to_show % 2 == 0 and self.epochs_to_show > 1:
            self.epochs_to_show -= 1
        self.autoscroll_state = config.autoscroll_state

        # find the set of y-axis locations of valid brain state labels
        self.label_display_options = convert_labels(
            np.array([b.digit for b in self.brain_state_set.brain_states]),
            style=DISPLAY_FORMAT,
        )
        self.smallest_display_label = np.min(self.label_display_options)

        self.ui.upperfigure.epoch_length = self.epoch_length
        self.ui.lowerfigure.epoch_length = self.epoch_length

        if self.recording:
            for name in self.selected_channels:
                self.lfp_adjustments[name] = [1.0, 0.0]
            self.eeg, spectrogram, spectrogram_frequencies = self._get_channel(
                self.selected_channels[0], first_eeg
            )
            if self.emg_mode == "icemg":
                samples = round(self.sampling_rate * self.epoch_length)
                self.upper_emg = self.emg.reshape(-1, samples).mean(axis=1)
            else:
                self.upper_emg = create_upper_emg_signal(
                    self.emg, self.sampling_rate, self.epoch_length, self.emg_filter
                )
                _, self.emg = transform_eeg_emg(np.zeros(1), self.emg)
        else:
            spectrogram, spectrogram_frequencies = create_spectrogram(
                self.eeg, self.sampling_rate, self.epoch_length
            )
            self.upper_emg = create_upper_emg_signal(
                self.emg, self.sampling_rate, self.epoch_length, self.emg_filter
            )
            self.eeg, self.emg = transform_eeg_emg(self.eeg, self.emg)
        # only a portion will be displayed in the lower plots
        # we'll store this so that it's easy to modify
        self.eeg_shown = None
        self.emg_shown = None

        # convert labels to "display" format and make an image to display them
        self.display_labels = convert_labels(self.labels, DISPLAY_FORMAT)
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )
        # same sort of thing for confidence scores
        self.confidence_img = create_confidence_img(self.confidence_scores)

        # history of changes to the brain state labels
        self.history = list()
        # index of the change "ahead" of the current state
        # i.e., which change will be applied by a "redo" action
        self.history_index = 0

        # set up both figures
        self.ui.upperfigure.setup_upper_figure(
            self.n_epochs,
            self.label_img,
            self.confidence_scores,
            self.confidence_img,
            spectrogram,
            spectrogram_frequencies,
            self.upper_emg,
            self.epochs_to_show,
            self.label_display_options,
            self.brain_state_set,
            self.roi_callback,
        )
        self.ui.lowerfigure.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            self.brain_state_set,
            self.label_display_options,
        )

        if self.recording:
            upper = self.ui.upperfigure
            if len(self.selected_channels) == 2:
                upper.add_second_spectrogram()
            for slot, name in enumerate(self.selected_channels):
                _, spec, _ = self._get_channel(name)
                upper.set_spectrogram(slot, spec, name)
            self._setup_channel_controls()
            self._configure_lower_channels()
            upper.canvas.axes[4].set_ylabel(
                self.emg_mode, rotation="horizontal", ha="right"
            )
            if self.emg_mode == "icemg":
                upper.canvas.axes[4].set_ylim(self.icemg_limits)
                upper.canvas.axes[4].set_yticks([0, 0.5, 1])
                upper.canvas.axes[4].tick_params(axis="y", labelsize=7)

        # initialize values that can be changed by user input
        self.epoch = 0
        self.upper_left_epoch = 0
        self.upper_right_epoch = self.n_epochs - 1
        self.lower_left_epoch = 0
        self.lower_right_epoch = self.epochs_to_show - 1
        self.eeg_signal_scale_factor = 1
        self.emg_signal_scale_factor = 1
        self.eeg_signal_offset = 0
        self.emg_signal_offset = 0
        self.roi_brain_state = 0
        self.label_roi_mode = False
        self.epoch_markers_shown = False

        # set autoscroll state and epochs to show, based on defaults
        self.ui.autoscroll.setChecked(self.autoscroll_state)
        self.ui.shownepochslabel.setText(str(self.epochs_to_show))

        # keep track of save state to warn user when they quit
        self.last_saved_labels = copy.deepcopy(self.labels)

        # no need to show the tip about the lower axes yet
        self.ui.reset_tip_label.setText("")

        # Populate both spectra before the first draw, including the second slot.
        if self.recording:
            self.adjust_upper_figure_x_limits()
        # populate the lower figure
        self.update_lower_figure()

        # user input: keyboard shortcuts
        keypress_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        keypress_right.activated.connect(partial(self.shift_epoch, DIRECTION_RIGHT))

        keypress_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        keypress_left.activated.connect(partial(self.shift_epoch, DIRECTION_LEFT))

        keypress_zoom_in_x = list()
        for zoom_key in [Qt.Key.Key_Plus, Qt.Key.Key_Equal]:
            keypress_zoom_in_x.append(QShortcut(QKeySequence(zoom_key), self))
            keypress_zoom_in_x[-1].activated.connect(partial(self.zoom_x, ZOOM_IN))

        keypress_zoom_out_x = QShortcut(QKeySequence(Qt.Key.Key_Minus), self)
        keypress_zoom_out_x.activated.connect(partial(self.zoom_x, ZOOM_OUT))

        keypress_quit = QShortcut(
            QKeySequence(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_W)),
            self,
        )
        keypress_quit.activated.connect(self.close)

        keypress_save = QShortcut(
            QKeySequence(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_S)),
            self,
        )
        keypress_save.activated.connect(self.save)

        keypress_modify_label = list()
        keypress_roi = list()
        digit_key_label_pairs = [
            (Qt.Key[f"Key_{brain_state.digit}"], brain_state.digit)
            for brain_state in self.brain_state_set.brain_states
        ] + [(Qt.Key.Key_Backspace, UNDEFINED_LABEL)]

        for digit_key, digit_label in digit_key_label_pairs:
            keypress_modify_label.append(
                QShortcut(
                    QKeySequence(digit_key),
                    self,
                )
            )
            keypress_modify_label[-1].activated.connect(
                partial(self.modify_current_epoch_label, digit_label)
            )
            keypress_roi.append(
                QShortcut(
                    QKeySequence(
                        QKeyCombination(
                            Qt.Modifier.SHIFT,
                            digit_key,
                        )
                    ),
                    self,
                )
            )
            keypress_roi[-1].activated.connect(
                partial(self.enter_label_roi_mode, digit_label)
            )

        keypress_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        keypress_esc.activated.connect(self.exit_label_roi_mode)

        keypress_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        keypress_space.activated.connect(
            partial(self.jump_to_next_state, DIRECTION_RIGHT, DIFFERENT_STATE)
        )
        keypress_shift_right = QShortcut(
            QKeySequence(
                QKeyCombination(
                    Qt.Modifier.SHIFT,
                    Qt.Key.Key_Right,
                )
            ),
            self,
        )
        keypress_shift_right.activated.connect(
            partial(self.jump_to_next_state, DIRECTION_RIGHT, DIFFERENT_STATE)
        )
        keypress_shift_left = QShortcut(
            QKeySequence(
                QKeyCombination(
                    Qt.Modifier.SHIFT,
                    Qt.Key.Key_Left,
                )
            ),
            self,
        )
        keypress_shift_left.activated.connect(
            partial(self.jump_to_next_state, DIRECTION_LEFT, DIFFERENT_STATE)
        )
        keypress_ctrl_right = QShortcut(
            QKeySequence(
                QKeyCombination(
                    Qt.Modifier.CTRL,
                    Qt.Key.Key_Right,
                )
            ),
            self,
        )
        keypress_ctrl_right.activated.connect(
            partial(self.jump_to_next_state, DIRECTION_RIGHT, UNDEFINED_STATE)
        )
        keypress_ctrl_left = QShortcut(
            QKeySequence(
                QKeyCombination(
                    Qt.Modifier.CTRL,
                    Qt.Key.Key_Left,
                )
            ),
            self,
        )
        keypress_ctrl_left.activated.connect(
            partial(self.jump_to_next_state, DIRECTION_LEFT, UNDEFINED_STATE)
        )

        keypress_undo = QShortcut(
            QKeySequence(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Z)),
            self,
        )
        keypress_undo.activated.connect(self.undo)
        keypress_redo = QShortcut(
            QKeySequence(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Y)),
            self,
        )
        keypress_redo.activated.connect(self.redo)

        # user input: mouse events
        self.ui.upperfigure.canvas.mpl_connect("button_press_event", self.click_to_jump)
        self.ui.upperfigure.canvas.mpl_connect(
            "scroll_event", partial(self.scroll_zoom, "upper")
        )
        self.ui.lowerfigure.canvas.mpl_connect(
            "scroll_event", partial(self.scroll_zoom, "lower")
        )
        self.ui.lowerfigure.canvas.mpl_connect(
            "button_press_event", self.lower_plot_click
        )
        self.ui.lowerfigure.canvas.mpl_connect(
            "button_release_event", self.lower_plot_click
        )
        self.ui.lowerfigure.canvas.mpl_connect(
            "motion_notify_event", self.lower_plot_move
        )
        self.now_zooming = False  # impose timeout on zoom events
        self.drag_axes = None  # axes experiencing a drag event
        self.last_y = None  # most recent mouse y position

        # user input: buttons
        self.ui.savebutton.clicked.connect(self.save)
        self.ui.xzoomin.clicked.connect(partial(self.zoom_x, ZOOM_IN))
        self.ui.xzoomout.clicked.connect(partial(self.zoom_x, ZOOM_OUT))
        self.ui.xzoomreset.clicked.connect(partial(self.zoom_x, ZOOM_RESET))
        self.ui.autoscroll.stateChanged.connect(self.update_autoscroll_state)
        self.ui.eegzoomin.clicked.connect(
            partial(self._lfp_button_action, 0, "zoom", ZOOM_IN)
        )
        self.ui.eegzoomout.clicked.connect(
            partial(self._lfp_button_action, 0, "zoom", ZOOM_OUT)
        )
        self.ui.emgzoomin.clicked.connect(
            partial(self.update_signal_zoom, EMG_SIGNAL, ZOOM_IN)
        )
        self.ui.emgzoomout.clicked.connect(
            partial(self.update_signal_zoom, EMG_SIGNAL, ZOOM_OUT)
        )
        self.ui.eegshiftup.clicked.connect(
            partial(self._lfp_button_action, 0, "offset", OFFSET_UP)
        )
        self.ui.eegshiftdown.clicked.connect(
            partial(self._lfp_button_action, 0, "offset", OFFSET_DOWN)
        )
        self.ui.emgshiftup.clicked.connect(
            partial(self.update_signal_offset, EMG_SIGNAL, OFFSET_UP)
        )
        self.ui.emgshiftdown.clicked.connect(
            partial(self.update_signal_offset, EMG_SIGNAL, OFFSET_DOWN)
        )
        self.ui.shownepochsplus.clicked.connect(
            partial(self.update_epochs_shown, DIRECTION_PLUS)
        )
        self.ui.shownepochsminus.clicked.connect(
            partial(self.update_epochs_shown, DIRECTION_MINUS)
        )
        self.ui.specbrighter.clicked.connect(
            partial(self.update_spectrogram_brightness, BRIGHTER)
        )
        self.ui.specdimmer.clicked.connect(
            partial(self.update_spectrogram_brightness, DIMMER)
        )
        self.ui.helpbutton.clicked.connect(self.show_user_manual)

        self._restore_view_state()
        self.show()

    def _save_view_state(self):
        upper = self.ui.upperfigure
        refs = [upper.spec_ref]
        if upper.extra_spec_ref is not None:
            refs.append(upper.extra_spec_ref)
        save_view_state(
            self.recording_file,
            {
                "sampling_rate": self.sampling_rate,
                "epoch_length": self.epoch_length,
                "n_epochs": self.n_epochs,
                "epoch": self.epoch,
                "upper_range": [self.upper_left_epoch, self.upper_right_epoch],
                "lower_left": self.lower_left_epoch,
                "epochs_to_show": self.epochs_to_show,
                "channels": self.selected_channels,
                "active_lfp_slot": self.active_lfp_slot,
                "lfp_adjustments": self.lfp_adjustments,
                "spectrum_clims": [list(ref.get_clim()) for ref in refs],
                "eeg_scale": self.eeg_signal_scale_factor,
                "eeg_offset": self.eeg_signal_offset,
                "emg_scale": self.emg_signal_scale_factor,
                "emg_offset": self.emg_signal_offset,
                "autoscroll": self.autoscroll_state,
                "epoch_markers": self.epoch_markers_shown,
                "label_source": self.resume_label_source,
                "label_file": self.label_file
                if self.curation_saved
                else self.source_label_file,
            },
        )

    def _restore_view_state(self):
        state = self.saved_view_state
        if not state or any(
            state.get(k) != v
            for k, v in (
                ("sampling_rate", self.sampling_rate),
                ("epoch_length", self.epoch_length),
                ("n_epochs", self.n_epochs),
            )
        ):
            return
        try:
            epoch = state["epoch"]
            left, right = state["upper_range"]
            shown, lower = state["epochs_to_show"], state["lower_left"]
            if not all(type(x) is int for x in (epoch, left, right, shown, lower)):
                return
            if not (
                0 <= left <= epoch <= right < self.n_epochs
                and 1 <= shown <= self.n_epochs
                and 0 <= lower <= epoch < lower + shown <= self.n_epochs
            ):
                return
            if self.recording:
                for slot, name in enumerate(state.get("channels", [])):
                    if (
                        slot < len(self.selected_channels)
                        and name in self.recording.channels
                    ):
                        self.ui.upperfigure.channel_selectors[slot].setCurrentText(name)
                for name, pair in state.get("lfp_adjustments", {}).items():
                    if name in self.recording.channels and self._valid_adjustment(pair):
                        self.lfp_adjustments[name] = list(pair)
            self.epoch = epoch
            self.epochs_to_show = shown
            self.lower_left_epoch, self.lower_right_epoch = lower, lower + shown - 1
            self.upper_left_epoch, self.upper_right_epoch = left, right
            self.ui.shownepochslabel.setText(str(shown))
            self.ui.lowerfigure.canvas.figure.clf()
            self.ui.lowerfigure.setup_lower_figure(
                self.label_img,
                self.sampling_rate,
                shown,
                self.brain_state_set,
                self.label_display_options,
            )
            if self.recording:
                self._configure_lower_channels()
            elif self._valid_adjustment(
                [state.get("eeg_scale"), state.get("eeg_offset")]
            ):
                self.eeg_signal_scale_factor = state["eeg_scale"]
                self.eeg_signal_offset = state["eeg_offset"]
            if self._valid_adjustment(
                [state.get("emg_scale"), state.get("emg_offset")]
            ):
                self.emg_signal_scale_factor = state["emg_scale"]
                self.emg_signal_offset = state["emg_offset"]
            refs = [self.ui.upperfigure.spec_ref, self.ui.upperfigure.extra_spec_ref]
            for ref, clim in zip(refs, state.get("spectrum_clims", []), strict=False):
                if (
                    ref is not None
                    and len(clim) == 2
                    and np.isfinite(clim).all()
                    and clim[0] < clim[1]
                ):
                    ref.set_clim(*clim)
            active = state.get("active_lfp_slot", 0)
            if type(active) is int and 0 <= active < max(
                1, len(self.selected_channels)
            ):
                self.active_lfp_slot = active
            self.epoch_markers_shown = bool(state.get("epoch_markers", False))
            self.ui.autoscroll.setChecked(
                bool(state.get("autoscroll", self.autoscroll_state))
            )
            self.update_figures()
        except (KeyError, TypeError, ValueError, IndexError):
            # Malformed preferences must never prevent access to a recording.
            return

    @staticmethod
    def _valid_adjustment(pair):
        try:
            return len(pair) == 2 and bool(np.isfinite(pair).all()) and pair[0] > 0
        except (TypeError, ValueError):
            return False

    def add_to_history(self, state_change: StateChange) -> None:
        """Add an event to the history of changes to brain state labels

        This allows the user to undo / redo changes to the brain state
        labels by navigating backwards or forwards through a list of
        changes that have been made. If one or more changes are undone,
        and then a new change is performed, it will not be possible to
        redo the changes that were undone. At most UNDO_LIMIT changes
        are stored.

        :param state_change: description of the change to the labels
        """
        # if history is empty
        if len(self.history) == 0:
            self.history.append(state_change)
            self.history_index = 1
            return
        # if we are not at the end of the history
        if self.history_index < len(self.history):
            # remove events after the most recent one
            self.history = self.history[: self.history_index]
            self.history.append(state_change)
            self.history_index += 1
        else:
            self.history.append(state_change)
            # if this would make the history list too long
            if self.history_index == UNDO_LIMIT:
                # remove the oldest entry
                self.history = self.history[1:]
            else:
                self.history_index += 1

    def force_modify_labels(self, epoch: int, new_labels: np.array) -> None:
        """Change brain state labels for an undo/redo action"""
        # make the change
        self.labels[epoch : epoch + len(new_labels)] = new_labels
        self.display_labels = convert_labels(
            self.labels,
            style=DISPLAY_FORMAT,
        )
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )

    def redo(self) -> None:
        """Redo the last change to brain state labels that was undone"""
        # if there are no events to redo
        if self.history_index == len(self.history):
            return
        # make the change
        state_change = self.history[self.history_index]
        self.force_modify_labels(
            epoch=state_change.epoch, new_labels=state_change.new_labels
        )
        # update history index
        self.history_index += 1
        # shift the cursor
        simulated_click = SimpleNamespace(
            **{"xdata": state_change.epoch, "inaxes": None}
        )
        self.click_to_jump(simulated_click)

    def undo(self) -> None:
        """Undo the last change to the brain state labels"""
        # if there are no events to undo
        if self.history_index == 0:
            return
        # make the change
        state_change = self.history[self.history_index - 1]
        self.force_modify_labels(
            epoch=state_change.epoch, new_labels=state_change.previous_labels
        )
        # update history index
        self.history_index -= 1
        # shift the cursor
        simulated_click = SimpleNamespace(
            **{"xdata": state_change.epoch, "inaxes": None}
        )
        self.click_to_jump(simulated_click)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Check if there are unsaved changes before closing"""
        if not np.array_equal(self.labels, self.last_saved_labels):
            result = QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Really quit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
                return
        self._save_view_state()
        event.accept()

    def show_user_manual(self) -> None:
        """Show a popup window with the user manual"""
        self.popup = QWidget()
        self.popup_vlayout = QVBoxLayout(self.popup)
        self.guide_textbox = QTextBrowser(self.popup)
        self.popup_vlayout.addWidget(self.guide_textbox)

        url = QUrl.fromLocalFile(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), USER_MANUAL_FILE)
        )
        self.guide_textbox.setSource(url)
        self.guide_textbox.setOpenLinks(False)

        self.popup.setGeometry(QRect(100, 100, 830, 600))
        self.popup.show()

    def jump_to_next_state(self, direction: str, target: str) -> None:
        """Jump to epoch based on a target brain state

        This allows the user to jump to the next epoch in a given direction
        (left or right) that has a given state (undefined, or different from
        the current epoch). It's useful for reviewing state transitions or
        locating unlabeled epochs.

        :param direction: left or right
        :param target: different or undefined
        """
        # create a simulated click so we can reuse click_to_jump
        simulated_click = SimpleNamespace(**{"xdata": self.epoch, "inaxes": None})
        if direction == DIRECTION_RIGHT:
            if target == DIFFERENT_STATE:
                matches = np.where(
                    self.labels[self.epoch + 1 :] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[self.epoch + 1 :] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = int(matches[0]) + 1 + self.epoch
        else:
            if target == DIFFERENT_STATE:
                matches = np.where(
                    self.labels[: self.epoch] != self.labels[self.epoch]
                )[0]
            else:
                matches = np.where(self.labels[: self.epoch] == UNDEFINED_LABEL)[0]
            if matches.size > 0:
                simulated_click.xdata = int(matches[-1])
        self.click_to_jump(simulated_click)

    def roi_callback(self, eclick, erelease) -> None:
        """Callback for ROI labeling widget

        This is called by the RectangleSelector widget when the user finishes
        drawing an ROI. It sets a range of epochs to the desired brain state.
        The function signature is required to have this format.
        """
        # get range of epochs affected
        epoch_start = int(np.ceil(eclick.xdata))
        epoch_end = int(np.floor(erelease.xdata)) + 1

        previous_labels = copy.deepcopy(self.labels[epoch_start:epoch_end])
        new_labels = np.ones(epoch_end - epoch_start).astype(int) * self.roi_brain_state

        # if something changed, track the change
        if not np.array_equal(previous_labels, new_labels):
            self.add_to_history(
                StateChange(
                    previous_labels=previous_labels,
                    new_labels=new_labels,
                    epoch=epoch_start,
                )
            )
        # make the change
        self.labels[epoch_start:epoch_end] = self.roi_brain_state
        self.display_labels = convert_labels(
            self.labels,
            style=DISPLAY_FORMAT,
        )
        self.label_img = create_label_img(
            self.display_labels, self.label_display_options
        )
        # update the plots
        self.update_figures()
        self.exit_label_roi_mode()

    def exit_label_roi_mode(self) -> None:
        """Restore the normal GUI state after an ROI is drawn"""
        self.ui.upperfigure.roi.set_active(False)
        self.ui.upperfigure.roi.set_visible(False)
        self.ui.upperfigure.roi.update()
        self.label_roi_mode = False
        self.ui.upperfigure.editing_patch.set_visible(False)
        # re-enable the click-to-jump preview cursor
        self.ui.upperfigure.cursor.visible = True

    def enter_label_roi_mode(self, brain_state: int) -> None:
        """Enter ROI drawing mode

        In this mode, a user can draw an ROI on the upper brain state label
        image to set a range of epochs to a new brain state.

        :param brain_state: new brain state to set
        """
        self.label_roi_mode = True
        self.roi_brain_state = brain_state
        # clicking won't jump to a new epoch in this mode, so hide the
        # click-to-jump preview cursor
        self.ui.upperfigure.cursor.visible = False
        self.ui.upperfigure.cursor.hide()
        self.ui.upperfigure.roi_patch.set(
            facecolor=LABEL_CMAP[
                convert_labels(np.array([brain_state]), DISPLAY_FORMAT)
            ]
        )
        self.ui.upperfigure.editing_patch.set_visible(True)
        self.ui.upperfigure.canvas.draw()
        self.ui.upperfigure.roi.set_active(True)

    @property
    def eeg_signal_scale_factor(self):
        if self.recording:
            return self.lfp_adjustments[self.selected_channels[self.active_lfp_slot]][0]
        return self._eeg_scale

    @eeg_signal_scale_factor.setter
    def eeg_signal_scale_factor(self, value):
        if self.recording:
            self.lfp_adjustments[self.selected_channels[self.active_lfp_slot]][0] = (
                value
            )
        else:
            self._eeg_scale = value

    @property
    def eeg_signal_offset(self):
        if self.recording:
            return self.lfp_adjustments[self.selected_channels[self.active_lfp_slot]][1]
        return self._eeg_offset

    @eeg_signal_offset.setter
    def eeg_signal_offset(self, value):
        if self.recording:
            self.lfp_adjustments[self.selected_channels[self.active_lfp_slot]][1] = (
                value
            )
        else:
            self._eeg_offset = value

    def _lfp_button_action(self, slot, action, direction, checked=False):
        """Each button group always targets its own displayed LFP slot."""
        if self.recording:
            self.active_lfp_slot = slot
        if action == "zoom":
            self.update_signal_zoom(EEG_SIGNAL, direction)
        else:
            self.update_signal_offset(EEG_SIGNAL, direction)

    def _setup_channel_controls(self):
        selectors = QVBoxLayout()
        for slot, name in enumerate(self.selected_channels):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Ch{slot + 1}", self))
            combo = QComboBox(self)
            combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            combo.addItems(list(self.recording.channels))
            combo.setCurrentText(name)
            combo.setToolTip(f"LFP displayed in slot {slot + 1}")
            combo.currentTextChanged.connect(partial(self.select_channel, slot))
            row.addWidget(combo)
            selectors.addLayout(row)
            self.ui.upperfigure.channel_selectors.append(combo)
        layout = self.ui.zoom_and_brightness
        zoom_index = next(
            i
            for i in range(layout.count())
            if layout.itemAt(i).layout() == self.ui.zoom_buttons
        )
        layout.insertLayout(zoom_index + 1, selectors)
        self.channel_selector_layout = selectors

        self.lfp_control_labels = []
        first = {
            "zoom_in": self.ui.eegzoomin,
            "zoom_out": self.ui.eegzoomout,
            "pan_up": self.ui.eegshiftup,
            "pan_down": self.ui.eegshiftdown,
        }
        self.lfp_buttons = [first]
        title = QLabel(f"Ch1: {self.selected_channels[0]}", self)
        self.lfp_control_labels.append(title)
        self.ui.eeg_emg_buttons.insertWidget(0, title)
        if len(self.selected_channels) < 2:
            return
        group = QVBoxLayout()
        group.setSpacing(2)
        title = QLabel(f"Ch2: {self.selected_channels[1]}", self)
        self.lfp_control_labels.append(title)
        group.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(self.ui.eegbuttons.horizontalSpacing())
        grid.setVerticalSpacing(self.ui.eegbuttons.verticalSpacing())
        buttons = {}
        for key, template, row, col, action, direction in (
            ("zoom_in", self.ui.eegzoomin, 0, 0, "zoom", ZOOM_IN),
            ("zoom_out", self.ui.eegzoomout, 1, 0, "zoom", ZOOM_OUT),
            ("pan_up", self.ui.eegshiftup, 0, 1, "offset", OFFSET_UP),
            ("pan_down", self.ui.eegshiftdown, 1, 1, "offset", OFFSET_DOWN),
        ):
            button = QPushButton(self)
            button.setObjectName(f"lfp2_{key}")
            button.setIcon(template.icon())
            button.setIconSize(template.iconSize())
            button.setSizePolicy(template.sizePolicy())
            button.setStyleSheet(template.styleSheet())
            button.setAutoRepeat(template.autoRepeat())
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(f"Ch2: {key.replace('_', ' ')}")
            button.clicked.connect(
                partial(self._lfp_button_action, 1, action, direction)
            )
            grid.addWidget(button, row, col)
            buttons[key] = button
        group.addLayout(grid)
        self.ui.eeg_emg_buttons.insertLayout(2, group)
        self.lfp_buttons.append(buttons)

    def _get_channel(self, name, raw=None):
        if name not in self.channel_cache:
            if raw is None:
                raw = self.recording.channels[name]
            if not np.isfinite(raw).all():
                raise ValueError(f"LFP {name} contains missing or infinite values")
            if name not in self.spectrum_cache:
                self.spectrum_cache[name] = create_spectrogram(
                    raw, self.sampling_rate, self.epoch_length, chunk_epochs=512
                )
            spec, f = self.spectrum_cache[name]
            if spec.shape[1] != self.n_epochs:
                raise ValueError("Spectrogram length does not match labels")
            # One fixed auto scale per channel, independent of the visible time range.
            center = np.mean(raw)
            sample = raw[:: max(1, len(raw) // 1_000_000)]
            scale = np.percentile(np.abs(sample - center), 98)
            normalized = (raw - center) / (scale * 2.2 if scale > 0 else 1)
            self.channel_cache[name] = (normalized, spec, f)
            self.lfp_adjustments.setdefault(name, [1.0, 0.0])
        return self.channel_cache[name]

    def select_channel(self, slot, name):
        """Replace only one display slot; preserve labels, history and time range."""
        previous = self.selected_channels[slot]
        try:
            data, spec, _ = self._get_channel(name)
        except Exception as exc:
            selector = self.ui.upperfigure.channel_selectors[slot]
            selector.blockSignals(True)
            selector.setCurrentText(previous)
            selector.blockSignals(False)
            QMessageBox.warning(self, "Cannot display channel", str(exc))
            return
        refs = [self.ui.upperfigure.spec_ref, self.ui.upperfigure.extra_spec_ref]
        self.channel_clims[previous] = refs[slot].get_clim()
        self.selected_channels[slot] = name
        self.lfp_control_labels[slot].setText(f"Ch{slot + 1}: {name}")
        for cached_name in list(self.channel_cache):
            if cached_name not in self.selected_channels:
                del self.channel_cache[cached_name]
        self.active_lfp_slot = slot
        if slot == 0:
            self.eeg = data
        self.ui.upperfigure.set_spectrogram(
            slot, spec, name, self.channel_clims.get(name)
        )
        self.ui.lowerfigure.lfp_axes[slot].set_ylabel(
            name, rotation="horizontal", ha="right", fontsize=8
        )
        self.adjust_upper_figure_x_limits()
        self.ui.upperfigure.canvas.draw()
        self.update_lower_figure()

    def _configure_lower_channels(self):
        lower = self.ui.lowerfigure
        lower.canvas.axes[0].set_ylabel(
            self.selected_channels[0], rotation="horizontal", ha="right", fontsize=8
        )
        if len(self.selected_channels) == 2:
            lower.add_second_lfp(self.selected_channels[1])
        if self.emg_mode == "icemg":
            ax = lower.canvas.axes[1]
            finite = self.emg[np.isfinite(self.emg)]
            lo, hi = (
                (min(0, float(finite.min())), max(1, float(finite.max())))
                if len(finite)
                else (0, 1)
            )
            ax.set_ylim(lo, hi)
            self.icemg_limits = (lo, hi)
            ax.set_yticks([lo, hi])
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("icEMG", rotation="horizontal", ha="right", fontsize=8)
            lower.emg_line.set_drawstyle("steps-post")
            # Keep the existing epoch brackets inside the new physical-value axis.
            for line in lower.bottom_marker:
                line.set_ydata(lo + (np.asarray(line.get_ydata()) + 1) * (hi - lo) / 2)

    def save(self) -> None:
        """Save brain state labels to file"""
        try:
            if self.recording and not self.curation_saved:
                # Another viewer may have saved since this window was opened.
                self.label_file = curated_label_path(self.source_label_file)
                self.setWindowTitle(f"AccuSleePy viewer: {self.label_file}")
            save_labels(self.labels, self.label_file)
            if self.recording:
                save_curation_metadata(
                    self.label_file,
                    self.labels,
                    self.recording.metadata,
                    self.epoch_length,
                    self.source_label_file,
                    {b.name: b.digit for b in self.brain_state_set.brain_states}
                    | {"undefined": UNDEFINED_LABEL},
                    self.initial_labels,
                    self.recording.source_file,
                )
                self.curation_saved = True
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save labels", str(exc))
            return
        self.last_saved_labels = copy.deepcopy(self.labels)
        self._save_view_state()

    def update_spectrogram_brightness(self, direction: str) -> None:
        """Modify spectrogram color range based on button press

        :param direction: brighter or dimmer
        """
        ref = self.ui.upperfigure.spec_ref
        if self.recording and self.active_lfp_slot == 1:
            ref = self.ui.upperfigure.extra_spec_ref
        vmin, vmax = ref.get_clim()
        if direction == BRIGHTER:
            ref.set(clim=(vmin, vmax * BRIGHTER_SCALE_FACTOR))
        else:
            ref.set(clim=(vmin, vmax * DIMMER_SCALE_FACTOR))
        self.ui.upperfigure.canvas.draw()

    def update_epochs_shown(self, direction: str) -> None:
        """Change the number of epochs shown based on button press

        The user can change the number of epochs shown in the lower figure
        via button presses. This requires extensive changes to both figures.
        The number of epochs can only change in increments of 2 and should
        always be an odd number >= 3.

        :param direction: plus or minus
        """
        # if we are near the beginning or end of the recording, we need
        # to change the epoch range differently.
        if direction == DIRECTION_PLUS:
            if self.epochs_to_show + 2 > self.n_epochs:
                return
            self.epochs_to_show += 2
            if self.lower_left_epoch == 0:
                self.lower_right_epoch += 2
            elif self.lower_right_epoch == self.n_epochs - 1:
                self.lower_left_epoch -= 2
            else:
                self.lower_left_epoch -= 1
                self.lower_right_epoch += 1
        else:
            if self.epochs_to_show > 3:
                self.epochs_to_show -= 2
                if self.lower_left_epoch == 0:
                    self.lower_right_epoch -= 2
                elif self.lower_right_epoch == self.n_epochs - 1:
                    self.lower_left_epoch += 2
                else:
                    self.lower_left_epoch += 1
                    self.lower_right_epoch -= 1

        self.ui.shownepochslabel.setText(str(self.epochs_to_show))

        # rebuild lower figure from scratch
        self.ui.lowerfigure.canvas.figure.clf()
        self.ui.lowerfigure.setup_lower_figure(
            self.label_img,
            self.sampling_rate,
            self.epochs_to_show,
            self.brain_state_set,
            self.label_display_options,
        )
        if self.recording:
            self._configure_lower_channels()
        self.update_figures()
        # upper plot x limits might need to change
        self.zoom_x(direction=None)

    def _refresh_eeg_line(self) -> None:
        """Recompute and update the displayed EEG line."""
        if self.recording:
            lower = self.ui.lowerfigure
            first = round(
                self.lower_left_epoch * self.sampling_rate * self.epoch_length
            )
            last = round(
                (self.lower_right_epoch + 1) * self.sampling_rate * self.epoch_length
            )
            lines = [lower.eeg_line, lower.extra_eeg_line]
            for slot, name in enumerate(self.selected_channels):
                values = self.channel_cache[name][0][first:last]
                scale, offset = self.lfp_adjustments[name]
                x, y = decimate_for_display(
                    values * scale + offset, round(lower.lfp_axes[slot].bbox.width)
                )
                lines[slot].set_data(x, y)
            return
        adjusted = (
            self.eeg_shown * self.eeg_signal_scale_factor + self.eeg_signal_offset
        )
        x, y = decimate_for_display(
            adjusted, round(self.ui.lowerfigure.canvas.axes[0].bbox.width)
        )
        self.ui.lowerfigure.eeg_line.set_data(x, y)

    def _refresh_emg_line(self) -> None:
        """Recompute and update the displayed EMG line."""
        if self.emg_mode == "icemg":
            # Select every transition, including NaN boundaries, without averaging.
            a = self.emg_shown
            changes = (
                np.flatnonzero(
                    (a[1:] != a[:-1]) & ~(np.isnan(a[1:]) & np.isnan(a[:-1]))
                )
                + 1
            )
            indices = np.unique(np.r_[0, changes, len(a) - 1])
            self.ui.lowerfigure.emg_line.set_data(indices, a[indices])
            # Manual zoom/pan operates on the viewport, not on the stored values.
            scale = self.emg_signal_scale_factor
            base_lo, base_hi = self.icemg_limits
            center = (base_hi + base_lo) / 2 - self.emg_signal_offset
            half = (base_hi - base_lo) / (2 * scale)
            self.ui.lowerfigure.canvas.axes[1].set_ylim(center - half, center + half)
            return
        adjusted = (
            self.emg_shown * self.emg_signal_scale_factor + self.emg_signal_offset
        )
        x, y = decimate_for_display(
            adjusted, round(self.ui.lowerfigure.canvas.axes[1].bbox.width)
        )
        self.ui.lowerfigure.emg_line.set_data(x, y)

    def update_signal_offset(self, signal: str, direction: str) -> None:
        """Shift EEG or EMG up or down

        :param signal: eeg or emg
        :param direction: up or down
        """
        if signal == EEG_SIGNAL:
            self.eeg_signal_offset += OFFSET_INCREMENTS[direction]
            self._refresh_eeg_line()
        else:
            self.emg_signal_offset += OFFSET_INCREMENTS[direction]
            self._refresh_emg_line()
        self.ui.lowerfigure.canvas.draw()

    def update_signal_zoom(self, signal: str, direction: str) -> None:
        """Zoom EEG or EMG y-axis

        :param signal: eeg or emg
        :param direction: in or out
        """
        if signal == EEG_SIGNAL:
            self.eeg_signal_scale_factor *= SIGNAL_ZOOM_FACTORS[direction]
        else:
            self.emg_signal_scale_factor *= SIGNAL_ZOOM_FACTORS[direction]
        self.update_lower_figure()

    def update_autoscroll_state(self, checked) -> None:
        """Toggle autoscroll behavior

        If autoscroll is enabled, setting the brain state of the current epoch
        via a keypress will advance to the next epoch.

        :param checked: state of the checkbox
        """
        self.autoscroll_state = checked

    def adjust_upper_figure_x_limits(self) -> None:
        """Update the x-axis limits and image data of the upper figure subplots"""
        # hand the images only the epochs that are about to be shown. this
        # comes first because set_extent can otherwise autoscale the limits
        # we are about to set
        self.ui.upperfigure.set_upper_visible_range(
            self.upper_left_epoch, self.upper_right_epoch, self.label_img
        )
        for i in [0, 1, 2, 4]:
            self.ui.upperfigure.canvas.axes[i].set_xlim(
                (self.upper_left_epoch - 0.5, self.upper_right_epoch + 0.5)
            )
        for ax in self.ui.upperfigure.spectrogram_axes:
            ax.set_xlim((self.upper_left_epoch, self.upper_right_epoch + 1))

    def zoom_x(self, direction: str | None) -> None:
        """Change upper figure x-axis zoom level

        :param direction: in, out, reset, or None
        """
        self.upper_left_epoch, self.upper_right_epoch = find_new_x_limits(
            direction=direction,
            left_epoch=self.upper_left_epoch,
            right_epoch=self.upper_right_epoch,
            min_n_shown=self.epochs_to_show,
            total_epochs=self.n_epochs,
            selected_epoch=self.epoch,
        )
        self.adjust_upper_figure_x_limits()
        self.ui.upperfigure.canvas.draw()

    def modify_current_epoch_label(self, digit: int) -> None:
        """Change the current epoch's brain state label

        :param digit: new brain state label in "digit" format
        """
        previous_label = self.labels[self.epoch]
        self.labels[self.epoch] = digit
        # if something changed, track the change
        if previous_label != digit:
            self.add_to_history(
                StateChange(
                    previous_labels=np.array([previous_label]),
                    new_labels=np.array([digit]),
                    epoch=self.epoch,
                )
            )

        # make the change
        display_label = convert_labels(
            np.array([digit]),
            style=DISPLAY_FORMAT,
        )[0]
        self.display_labels[self.epoch] = display_label
        # update the label image
        if display_label == 0:
            self.label_img[:, self.epoch] = np.array([0, 0, 0, 1])
        else:
            self.label_img[:, self.epoch, :] = 1
            self.label_img[
                display_label - self.smallest_display_label, self.epoch, :
            ] = LABEL_CMAP[display_label]
        # autoscroll, if that is enabled
        if self.autoscroll_state and self.epoch < self.n_epochs - 1:
            # the label image just changed, so the upper figure needs a real
            # redraw even if its x limits stay put
            self.shift_epoch(  # this calls update_figures()
                DIRECTION_RIGHT, force_upper_redraw=True
            )
        else:
            self.update_figures()

    def shift_epoch(self, direction: str, force_upper_redraw: bool = False) -> None:
        """Set the current epoch one step forward or backward

        When the user presses the left or right arrow key, the previous
        or next epoch will be selected. There are a variety of edge cases
        that need to be handled separately for the upper and lower figures.

        :param direction: left or right
        :param force_upper_redraw: whether the upper figure must be redrawn
            in full even if its x limits do not move, because something
            other than the epoch marker changed
        """
        # whether the upper figure's x limits moved, which rules out
        # redrawing only the epoch marker
        upper_shifted = False
        shift_amount = {DIRECTION_LEFT: -1, DIRECTION_RIGHT: 1}[direction]
        # prevent movement outside the data range
        if not (0 <= (self.epoch + shift_amount) < self.n_epochs):
            return

        # shift to new epoch
        self.epoch = self.epoch + shift_amount

        # update upper plot if needed
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        if (
            (
                self.epoch
                > self.upper_left_epoch + (1 - SCROLL_BOUNDARY) * upper_epochs_shown
                or self.epoch + (self.epochs_to_show - 1) / 2 > self.upper_right_epoch
            )
            and self.upper_right_epoch < (self.n_epochs - 1)
            and direction == DIRECTION_RIGHT
        ):
            self.upper_left_epoch += 1
            self.upper_right_epoch += 1
            self.adjust_upper_figure_x_limits()
            upper_shifted = True
        elif (
            (
                self.epoch
                < self.upper_left_epoch + SCROLL_BOUNDARY * upper_epochs_shown
                or self.epoch - (self.epochs_to_show - 1) / 2 < self.upper_left_epoch
            )
            and self.upper_left_epoch > 0
            and direction == DIRECTION_LEFT
        ):
            self.upper_left_epoch -= 1
            self.upper_right_epoch -= 1
            self.adjust_upper_figure_x_limits()
            upper_shifted = True

        # update parts of lower plot
        old_window_center = round((self.epochs_to_show - 1) / 2) + self.lower_left_epoch
        # change the window bounds if needed
        if self.epoch < old_window_center and self.lower_left_epoch > 0:
            self.lower_left_epoch -= 1
            self.lower_right_epoch -= 1
        elif (
            self.epoch > old_window_center
            and self.lower_right_epoch < self.n_epochs - 1
        ):
            self.lower_left_epoch += 1
            self.lower_right_epoch += 1

        self.update_figures(upper_marker_only=not (upper_shifted or force_upper_redraw))

    def update_upper_marker(self) -> None:
        """Update location of the upper figure's epoch marker"""
        epoch_padding = round((self.epochs_to_show - 1) / 2)
        if self.epoch - epoch_padding < 0:
            left_edge = 0
            right_edge = self.epochs_to_show - 1
        elif self.epoch + epoch_padding > self.n_epochs - 1:
            right_edge = self.n_epochs - 1
            left_edge = self.n_epochs - self.epochs_to_show
        else:
            left_edge = self.epoch - epoch_padding
            right_edge = self.epoch + epoch_padding
        self.ui.upperfigure.upper_marker[0].set_xdata(
            [
                left_edge - 0.5,
                right_edge + 0.5,
            ]
        )
        self.ui.upperfigure.upper_marker[1].set_xdata([self.epoch])
        self.ui.upperfigure.set_selected_epoch(self.epoch)

    def update_lower_epoch_marker(self) -> None:
        """Update location of the lower figure's epoch marker"""
        marker_left = (
            (self.epoch - self.lower_left_epoch)
            * self.epoch_length
            * self.sampling_rate
        )
        marker_right = (
            (1 + self.epoch - self.lower_left_epoch)
            * self.epoch_length
            * self.sampling_rate
        )
        self.ui.lowerfigure.top_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerfigure.top_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerfigure.top_marker[2].set_xdata([marker_right, marker_right])
        self.ui.lowerfigure.bottom_marker[0].set_xdata([marker_left, marker_left])
        self.ui.lowerfigure.bottom_marker[1].set_xdata([marker_left, marker_right])
        self.ui.lowerfigure.bottom_marker[2].set_xdata([marker_right, marker_right])
        for line, primary in zip(
            self.ui.lowerfigure.extra_top_marker,
            self.ui.lowerfigure.top_marker,
            strict=False,
        ):
            line.set_xdata(primary.get_xdata())

    def update_figures(self, upper_marker_only: bool = False) -> None:
        """Update and redraw both figures

        :param upper_marker_only: whether the epoch marker's position is the
            only thing that changed in the upper figure. If so, that figure
            is blitted from a cached background instead of redrawn, which
            skips resampling the spectrogram. Defaults to False so that any
            new caller redraws in full rather than risking a stale figure.
        """
        # upper figure
        self.update_upper_marker()
        if upper_marker_only:
            self.ui.upperfigure.draw_upper_marker()
        else:
            # this re-slices the images and syncs the x limits. it isn't
            # always needed, but it's not too expensive
            self.adjust_upper_figure_x_limits()
            self.ui.upperfigure.canvas.draw()
        # lower figure
        self.update_lower_figure()

    def update_lower_figure(self) -> None:
        """Update and redraw the lower figure"""
        # get subset of signals to plot
        first_sample = round(
            self.lower_left_epoch * self.sampling_rate * self.epoch_length
        )
        last_sample = round(
            (self.lower_right_epoch + 1) * self.sampling_rate * self.epoch_length
        )
        # store the (unmodified) eeg/emg segments to display
        self.eeg_shown = self.eeg[first_sample:last_sample]
        self.emg_shown = self.emg[first_sample:last_sample]

        self.update_lower_epoch_marker()

        # replot eeg and emg
        self._refresh_eeg_line()
        self._refresh_emg_line()

        # replot brain state
        self.ui.lowerfigure.label_img_ref.set(
            data=self.label_img[
                :, self.lower_left_epoch : (self.lower_right_epoch + 1), :
            ]
        )
        # update timestamps
        x_ticks = resample_x_ticks(
            np.arange(self.lower_left_epoch, self.lower_right_epoch + 1)
        )
        self.ui.lowerfigure.canvas.axes[1].set_xticklabels(
            [
                f"{int(x // 3600):02d}:{int(x // 60) % 60:02d}:{x % 60:05.2f}"
                for x in x_ticks * self.epoch_length
            ]
        )

        self.ui.lowerfigure.eeg_epoch_plot.set(visible=self.epoch_markers_shown)
        self.ui.lowerfigure.emg_epoch_plot.set(visible=self.epoch_markers_shown)
        if self.ui.lowerfigure.extra_eeg_epoch_plot is not None:
            self.ui.lowerfigure.extra_eeg_epoch_plot.set(
                visible=self.epoch_markers_shown
            )

        self.ui.lowerfigure.canvas.draw()

    def click_to_jump(self, event) -> None:
        """Jump to a new epoch when the user clicks on the upper figure

        This is the callback for mouse clicks on the upper figure. Clicking on
        any of the subplots will jump to the nearest epoch.

        :param event: a MouseEvent containing the click data
        """
        # make sure click location is valid
        # and we are not in label ROI mode
        if event.xdata is None or self.label_roi_mode:
            return

        # get click location
        x = event.xdata
        # if it's a real click, and not one we simulated
        if event.inaxes in self.ui.upperfigure.spectrogram_axes:
            if self.recording:
                self.active_lfp_slot = self.ui.upperfigure.spectrogram_axes.index(
                    event.inaxes
                )
            x -= 0.5

        # get the "zoom level" so we can preserve that
        upper_epochs_shown = self.upper_right_epoch - self.upper_left_epoch + 1
        upper_epoch_padding = round((upper_epochs_shown - 1) / 2)
        # update epoch
        self.epoch = round(np.clip(x, 0, self.n_epochs - 1))
        # update upper figure x-axis limits
        if self.epoch - upper_epoch_padding < 0:
            self.upper_left_epoch = 0
            self.upper_right_epoch = upper_epochs_shown - 1
        elif self.epoch + upper_epoch_padding > self.n_epochs - 1:
            self.upper_right_epoch = self.n_epochs - 1
            self.upper_left_epoch = self.n_epochs - upper_epochs_shown
        else:
            self.upper_left_epoch = self.epoch - upper_epoch_padding
            self.upper_right_epoch = self.epoch + upper_epoch_padding
        self.adjust_upper_figure_x_limits()

        # update lower figure x-axis range
        lower_epoch_padding = round((self.epochs_to_show - 1) / 2)
        if self.epoch - lower_epoch_padding < 0:
            self.lower_left_epoch = 0
            self.lower_right_epoch = self.epochs_to_show - 1
        elif self.epoch + lower_epoch_padding > self.n_epochs - 1:
            self.lower_right_epoch = self.n_epochs - 1
            self.lower_left_epoch = self.n_epochs - self.epochs_to_show
        else:
            self.lower_left_epoch = self.epoch - lower_epoch_padding
            self.lower_right_epoch = self.epoch + lower_epoch_padding

        self.update_figures()

    def scroll_zoom(self, canvas, event) -> None:
        """Zoom on mouse scroll events"""
        if self.now_zooming:
            return

        self.now_zooming = True
        start_time = time.time()

        direction = ZOOM_IN if event.button == "up" else ZOOM_OUT

        if canvas == "upper":
            self.zoom_x(direction=direction)
        elif event.inaxes in self.ui.lowerfigure.lfp_axes:
            if self.recording:
                self.active_lfp_slot = self.ui.lowerfigure.lfp_axes.index(event.inaxes)
            self.update_signal_zoom(EEG_SIGNAL, direction)
        elif event.inaxes == self.ui.lowerfigure.canvas.axes[1]:
            self.update_signal_zoom(EMG_SIGNAL, direction)

        end_time = time.time()
        time_elapsed = end_time - start_time
        if time_elapsed < 1 / MAX_SCROLL_EVENTS_PER_SEC:
            time.sleep(1 / MAX_SCROLL_EVENTS_PER_SEC - time_elapsed)
        self.now_zooming = False

    def lower_plot_click(self, event):
        """Triggered when the lower plots are clicked"""
        # right click swaps epoch marker state
        if event.button == 3 and event.name == "button_press_event":
            self.epoch_markers_shown = not self.epoch_markers_shown
            self.ui.lowerfigure.eeg_epoch_plot.set(visible=self.epoch_markers_shown)
            self.ui.lowerfigure.emg_epoch_plot.set(visible=self.epoch_markers_shown)
            if self.ui.lowerfigure.extra_eeg_epoch_plot is not None:
                self.ui.lowerfigure.extra_eeg_epoch_plot.set(
                    visible=self.epoch_markers_shown
                )
            self.ui.lowerfigure.canvas.draw()
            return

        # only process LMB clicks on the EEG/EMG axes
        if event.button != 1 or (
            event.inaxes
            not in [*self.ui.lowerfigure.lfp_axes, self.ui.lowerfigure.canvas.axes[1]]
            and event.name == "button_press_event"
        ):
            return

        if event.name == "button_press_event":
            # button press sets the current axes, y location
            self.drag_axes = event.inaxes
            if self.recording and event.inaxes in self.ui.lowerfigure.lfp_axes:
                self.active_lfp_slot = self.ui.lowerfigure.lfp_axes.index(event.inaxes)
            self.last_y = event.ydata
            self.ui.reset_tip_label.setText("")
        else:
            # display tip if y position is extreme
            if max([abs(self.eeg_signal_offset), abs(self.emg_signal_offset)]) > 0.8:
                self.ui.reset_tip_label.setText(RESET_TIP)

            # button release clears current axes, y location
            self.drag_axes = None
            self.last_y = None

        # double click resets plot to original conditions
        if event.dblclick:
            if event.inaxes in self.ui.lowerfigure.lfp_axes:
                self.eeg_signal_offset = 0
                self.eeg_signal_scale_factor = 1
                self._refresh_eeg_line()
            else:
                self.emg_signal_offset = 0
                self.emg_signal_scale_factor = 1
                self._refresh_emg_line()
            self.ui.lowerfigure.canvas.draw()
            self.ui.reset_tip_label.setText("")

    def lower_plot_move(self, event):
        """Triggered when the mouse moves in the lower plots"""
        # only process move events where LMB is down
        # and the axes match the most recent click
        if event.button != 1 or event.inaxes != self.drag_axes:
            return

        # get movement distance on the y-axis
        y_diff = event.ydata - self.last_y
        # update mouse's most recent y position
        self.last_y = event.ydata

        # update y offset for the selected plot
        if event.inaxes in self.ui.lowerfigure.lfp_axes:
            self.eeg_signal_offset += y_diff
            self._refresh_eeg_line()
        else:
            self.emg_signal_offset += y_diff
            self._refresh_emg_line()
        self.ui.lowerfigure.canvas.draw()


def convert_labels(labels: np.array, style: str) -> np.array:
    """Convert labels between "display" and "digit" formats

    It's useful to represent brain state labels in two ways:
    Digit format: this is how labels are represented in files. It matches the digit
        attribute of the BrainState class as well as the number pressed on the
        keyboard to set an epoch to that brain state.
    Display format: the y-axis value associated with a brain state when brain state
        labels are displayed as an image. This is also the index of the brain state
        in the colormap. Undefined epochs are mapped to 0, and digits are mapped to
        the numbers 1-10 in keyboard order (1234567890).

    :param labels: brain state labels
    :param style: target format for the output
    :return: formatted labels
    """
    if style == DISPLAY_FORMAT:
        # convert 0 to 10, undefined to 0
        labels = [i if i != 0 else 10 for i in labels]
        return np.array([i if i != UNDEFINED_LABEL else 0 for i in labels])
    elif style == DIGIT_FORMAT:
        # convert 0 to undefined, 10 to 0
        labels = [i if i != 0 else UNDEFINED_LABEL for i in labels]
        return np.array([i if i != 10 else 0 for i in labels])
    else:
        raise ValueError(f"style must be '{DISPLAY_FORMAT}' or '{DIGIT_FORMAT}'")


def create_label_img(labels: np.array, label_display_options: np.array) -> np.array:
    """Create an image to display brain state labels

    :param labels: brain state labels, in "display" format
    :param label_display_options: y-axis locations of valid brain state labels
    :return: brain state label image
    """
    # While there can be up to 10 valid brain states, it's possible that not all of them
    # are in use. We don't need to display brain states below and above the range of
    # valid brain states, since those rows would always be empty.
    smallest_display_label = np.min(label_display_options)
    # "background" of the image is white
    label_img = np.ones(
        [
            (np.max(label_display_options) - smallest_display_label + 1),
            len(labels),
            4,
        ]
    )
    for i, label in enumerate(labels):
        if label > 0:
            label_img[label - smallest_display_label, i, :] = LABEL_CMAP[label]
        else:
            # label is undefined
            label_img[:, i] = np.array([0, 0, 0, 1])
    return label_img


def create_confidence_img(confidence_scores: np.array) -> np.array:
    """Create an image to display confidence scores

    :param confidence_scores: confidence scores
    :return: confidence score image
    """
    if confidence_scores is None:
        return None

    confidence_img = np.ones([1, len(confidence_scores), 3])
    for i, c in enumerate(confidence_scores):
        confidence_img[0, i, 1:] = c
    return confidence_img


def create_upper_emg_signal(
    emg: np.array,
    sampling_rate: int | float,
    epoch_length: int | float,
    emg_filter: EMGFilter,
) -> np.array:
    """Calculate RMS of EMG for each epoch and apply a ceiling

    :param emg: EMG signal
    :param sampling_rate: sampling rate, in Hz
    :param epoch_length: epoch length, in seconds
    :param emg_filter: EMG filter parameters
    :return: processed EMG signal
    """
    emg_rms = get_emg_power(
        emg,
        sampling_rate,
        epoch_length,
        emg_filter,
    )
    return np.clip(
        emg_rms, np.percentile(emg_rms, 0.1), np.mean(emg_rms) + np.std(emg_rms) * 2.5
    )


def transform_eeg_emg(eeg: np.array, emg: np.array) -> (np.array, np.array):
    """Center and scale the EEG and EMG signals

    A heuristic approach to fitting the EEG and EMG signals in the plot.

    :param eeg: EEG signal
    :param emg: EMG signal
    :return: centered and scaled signals
    """
    eeg = eeg - np.mean(eeg)
    emg = emg - np.mean(emg)
    # avoid dividing by 0 if a signal is flat (e.g., no EMG was recorded)
    eeg_scale = np.percentile(np.abs(eeg), 98)
    if eeg_scale > 0:
        eeg = eeg / eeg_scale / 2.2
    emg_scale = np.percentile(np.abs(emg), 98)
    if emg_scale > 0:
        emg = emg / emg_scale / 2.2
    return eeg, emg


# minimum samples per pixel column before decimating a signal is worth it.
# measured: at 2 samples per column decimating is slightly slower, and it
# starts paying off from about 3
MIN_SAMPLES_PER_COLUMN = 4


def decimate_for_display(signal: np.array, n_columns: int) -> tuple:
    """Reduce a signal to the smallest and largest value in each pixel column

    Drawing many more points than the plot has pixels costs time without
    changing what the user sees. Keeping both extremes of every column
    preserves the envelope of the trace at display resolution, so the
    result is visually the same as plotting every sample.

    Signals that are not oversampled enough are returned unchanged. Below
    roughly three samples per column the vertical segments this produces
    cost more to rasterize than the samples it removes, so recordings at
    low sampling rates are left alone.

    :param signal: signal to be displayed
    :param n_columns: width of the plot, in pixels
    :return: x and y coordinates to plot
    """
    n_samples = len(signal)
    if n_columns < 1 or n_samples < MIN_SAMPLES_PER_COLUMN * n_columns:
        return np.arange(n_samples), signal
    # trim the remainder so the signal splits evenly into columns. this
    # discards less than one column's worth, which is under a pixel wide
    samples_per_column = n_samples // n_columns
    columns = signal[: samples_per_column * n_columns].reshape(
        n_columns, samples_per_column
    )
    y = np.empty(n_columns * 2)
    y[0::2] = columns.min(axis=1)
    y[1::2] = columns.max(axis=1)
    # both extremes of a column are drawn at the same x, so each column
    # becomes one vertical segment spanning that column's range
    x = np.repeat(np.linspace(0, n_samples - 1, n_columns), 2)
    return x, y


def find_new_x_limits(
    direction: str | None,
    left_epoch: int,
    right_epoch: int,
    total_epochs: int,
    min_n_shown: int,
    selected_epoch: int,
) -> (int, int):
    """Calculate new plot x limits to allow zooming

    :param direction: in, out, reset, or None
    :param left_epoch: index of current leftmost epoch
    :param right_epoch: index of current rightmost epoch
    :param total_epochs: total number of epochs in the recording
    :param min_n_shown: minimum number of epochs to display
    :param selected_epoch: currently selected epoch
    """
    # number of epochs currently displayed in the upper plots
    current_n_shown = right_epoch - left_epoch + 1
    if direction == ZOOM_IN:
        # can't display fewer than the number of epochs in the lower plot
        new_n_shown = max([min_n_shown, round(current_n_shown * (1 - ZOOM_FACTOR))])
    elif direction == ZOOM_OUT:
        # can't display more than the total number of epochs
        new_n_shown = min([total_epochs, round(current_n_shown / (1 - ZOOM_FACTOR))])
    elif direction == ZOOM_RESET:
        left_epoch = 0
        right_epoch = total_epochs - 1
        return left_epoch, right_epoch
    else:  # just recalculating if min_n_shown has changed
        new_n_shown = int(
            np.clip(current_n_shown, a_min=min_n_shown, a_max=total_epochs)
        )

    # count epochs to show on either side of the selected epoch
    epochs_on_left_side = int(np.ceil((new_n_shown - 1) / 2))
    epochs_on_right_side = new_n_shown - epochs_on_left_side - 1
    if selected_epoch - epochs_on_left_side < 0:
        # can't go further left than 0
        left_epoch = 0
        right_epoch = new_n_shown - 1
    elif selected_epoch + epochs_on_right_side >= total_epochs:
        # can't go further right than the total number of epochs
        left_epoch = total_epochs - new_n_shown
        right_epoch = total_epochs - 1
    else:
        left_epoch = selected_epoch - epochs_on_left_side
        right_epoch = selected_epoch + epochs_on_right_side

    return left_epoch, right_epoch
