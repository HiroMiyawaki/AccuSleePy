# Manual scoring interface
- Upper three plots: click to jump to an epoch, scroll to zoom
- Lower EEG & EMG plots:
  - click + drag to pan, scroll to zoom, double-click to reset view
  - right click to toggle epoch boundary markers
- Ctrl + S: save labels to file
- Right / left arrow: move one epoch forward / backward in time
- Numbers 0-9: set current epoch to this brain state
- Backspace: set current epoch to the undefined state
- Shift + (number 0-9, or backspace):
    Press this combination, then click and drag on the upper
    brain state label plot to set selected epochs
    to this brain state. Press Escape to cancel.
- Ctrl + Z: undo last change to brain state labels
- Ctrl + Y: redo last change to brain state labels
- Plus (+): zoom in (upper panel x-axis)
- Minus (-): zoom out (upper panel x-axis)
- Shift + right / left arrow: jump to the next / previous epoch with a different brain state
    than the current epoch
- Space bar: same as shift + right arrow
- Ctrl + right / left arrow: jump to the next / preceding undefined epoch
- Ctrl + W: quit

![ ](../images/viewer_window_annotated.png)

## Multiple LFP channels and icEMG

For manual curation, CSV/Parquet signal columns other than `emg` and `icemg`
are available as LFP channels. The first two are shown in vertically stacked
spectrograms and waveforms. Use the Ch1/Ch2 dropdowns below the time zoom buttons in the right-hand
control panel to choose the channels. With one LFP, only one spectrum and waveform are shown.
All panels use the same epoch selection and label history; existing keyboard
shortcuts, range labeling, navigation, undo and redo continue to work.

Each channel has its own automatic color range and waveform scale. Waveform
scales stay fixed while moving through time. The Ch1 and Ch2 amplitude button groups each control their own displayed
LFP, independent of which panel was last clicked. Scrolling and dragging on
either waveform adjusts that channel, and double-click resets its adjustment. Spectrogram
brightness controls apply to the last selected/clicked LFP spectrum.
Channel changes preserve time, zoom and edits. Switching back restores the
channel's waveform adjustments; spectra are cached for the viewer session.

An `emg` column selects the original EMG processing. If it is absent and an
`icemg` column is present, icEMG is displayed without filtering, RMS, logarithms,
centering or amplitude normalization. Its upper trace is the arithmetic mean
per epoch (any NaN makes that epoch NaN); its lower trace shows the stored
values with steps and gaps for NaNs. Lower icEMG zoom/pan changes the viewport,
not the values. If both columns exist, `emg` takes precedence.

Version 1 `accusleepy` Parquet metadata supplies the sampling rate and, when
present in `alignment`, the epoch duration. New-format recordings must have
exact sample/epoch/label counts: they are not resampled, padded or truncated.
Only signal columns belong in these files; timestamps, flags and labels must
not appear as additional signal columns. LFPs with NaN/infinity are rejected
with an error instead of being silently repaired.

When starting from original labels, Ctrl+S writes a fresh `*.curated.csv` (or a
numbered name if needed), leaving the original labels intact. When starting
from `*.curated.csv`, `*.curated-N.csv` or an older nested curation filename,
Ctrl+S updates that same CSV and its sidecars. The window title shows the output
path. Subsequent saves in the same window also update that output. A matching
`*.metadata.json` records input provenance and changed epoch indices. When
master-clock metadata is present, `*.intervals.csv` also records integer start
and exclusive-end reference samples for every epoch. Original invalid/uncertain
flags remain in the source files. Legacy `eeg`/`emg` files without new metadata
retain their previous save behavior. New-format support is for manual curation;
automatic scoring and training retain their existing input format.

Thin red vertical guides in the upper hypnogram, spectra and EMG/icEMG trace
mark the same selected epoch as the red diamond. They follow navigation and
remain aligned when zooming or switching LFP channels.

## Resume a recording

Saving labels or closing the viewer also saves its display preferences in the
application configuration directory (`viewer-state`, separate from the data).
Opening the same recording restores the selected epoch, upper time range,
lower time range/window length, LFP selections, amplitude adjustments,
spectrogram brightness, autoscroll and epoch-marker visibility.

If you select the original label file again, the viewer resumes its last saved
curation CSV when available; the main window reports the label path being used.
Explicitly choosing a different label file uses that file instead. Unsaved label
changes and undo history are not restored. Save your labels with Ctrl+S before
quitting if you want to retain edits; the existing unsaved-changes warning remains.
Display preferences are recorded on save and normal close, not continuously.

Preferences are specific to the recording's absolute path, size and modification
time, and are applied only with matching sample rate, epoch duration and epoch
count. Replacing/moving a recording or changing its timing starts a fresh view.
