"""Per-recording viewer preferences, separate from experimental data."""

import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)


def state_path(filename):
    identity = str(Path(filename).resolve())
    key = hashlib.sha256(identity.encode()).hexdigest()
    return Path(user_config_dir("accusleepy")) / "viewer-state" / f"{key}.json"


def identity(filename):
    p = Path(filename).resolve()
    stat = p.stat()
    return {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_view_state(filename):
    if not filename:
        return None
    try:
        data = json.loads(state_path(filename).read_text(encoding="utf-8"))
        if data.get("version") == 1 and data.get("recording") == identity(filename):
            return data
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def save_view_state(filename, data):
    if not filename:
        return
    temporary = None
    try:
        destination = state_path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = dict(data, version=1, recording=identity(filename))
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as f:
            temporary = f.name
            json.dump(data, f, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, destination)
    except (OSError, ValueError):
        logger.warning("Could not save viewer settings", exc_info=True)
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove temporary viewer settings", exc_info=True
                )
