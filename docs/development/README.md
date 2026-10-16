# Portable development and validation

The multi-LFP/icEMG additions are currently on `dev/m2-environment` in this fork.
Use a checkout containing that branch's changes; installing the upstream PyPI
release alone does not install these features. Publication and a new PyPI release
are separate tasks.

## Install from a checkout

Use a native Python supported by `pyproject.toml` (Python 3.12 is the tested
baseline). On Apple Silicon, use an arm64 Python. The original dependency bounds
and Poetry lock are retained. No CUDA device is required for manual curation.

On macOS/Linux, from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
python -m accusleepy
```

On Windows (PowerShell), from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m accusleepy
```

Activation is optional: use `.venv/bin/python` on macOS/Linux or
`.venv\Scripts\python.exe` on Windows when it is inconvenient.
For a reproducible development environment using the committed lock:

```sh
poetry env use python
poetry install --no-interaction
poetry run pre-commit install
poetry run python -m accusleepy
```

Do not lower dependency requirements to work around an installation failure.
First check Python/CPU architecture, available wheels and the actual resolver
error. A failure on an untested platform should be reported with those details.

## Synthetic smoke test

From an installed source checkout:

```sh
python examples/create_demo_recording.py demo-data
python -m accusleepy
```

In Windows without activation, replace `python` with `.venv\Scripts\python.exe`.
The generator refuses to overwrite existing outputs. Use a new output directory
for another run. The generated files and all real data stay out of Git.

Select the generated Parquet and label CSV. Check the stacked LFP panels, Ch1/Ch2
selectors, independent amplitude buttons, raw icEMG steps/gaps, arrow navigation,
number-key scoring, range scoring, undo/redo and Ctrl+S. Close/reopen the same
recording and confirm that time range, selected epoch, channels and saved labels
resume. Reopening the curated CSV must update that same CSV when saved.

See [the input contract](../input-format.md) and the
[full operation guide](../../accusleepy/gui/text/manual_scoring_guide.md).

## Automated checks

With Poetry:

```sh
poetry run ruff check
poetry run ruff format --check
poetry run pytest
poetry build
```

For headless tests, set `QT_QPA_PLATFORM=offscreen`. Do not use that setting for
normal desktop operation. Tests isolate resume preferences in temporary folders;
no private recordings or user configuration are required. CI is configured for the original
Linux Python 3.11–3.14 matrix and adds Python 3.12 on Windows and macOS. It also
generates synthetic inputs and builds wheel/source distributions.

Headless CI verifies logic and Qt lifecycle, not the native desktop appearance.
Native macOS was tested locally. Windows/Linux native GUI behavior still needs
confirmation on those machines; adding a CI job is not evidence it has passed.

## Publication boundaries

Include application code, synthetic tests, the generator, generic documentation,
CI, `pyproject.toml`, `poetry.lock` and the existing license. Keep the version at
0.13.0 until a release/versioning decision is made; local wheel builds are for
validation and must not be confused with a new upstream release.

`docs/local/` is ignored and holds machine/session-specific investigation notes,
exporter handoff and environment snapshots. `.venv`, caches and generated
`demo-data/` are also ignored. No real Parquet/LFP/NPZ files, curated labels,
viewer preferences or personal absolute paths belong in a commit or distribution.
Avoid `git add -f` on these ignored paths. The repository's default ignore rules
are not a general protection against arbitrary data files elsewhere: review
`git status` and staged filenames before publication.

Runtime settings are located with `platformdirs`, not inside the repository.
Resume preferences are local to a recording path, file identity and epoch timing.
They are not portable session state: copy the curated CSV and its metadata/interval
sidecars when moving manual scores to another computer.

Dynamic channel controls are built in `ManualScoringWindow`, outside generated
Qt Designer code. Keep that separation when updating `.ui` files.
