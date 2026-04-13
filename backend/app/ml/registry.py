"""Model artefact registry — save and load versioned sklearn pipelines."""

from datetime import datetime
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import settings


def _models_dir() -> Path:
    """Return the models directory, creating it if it does not exist."""
    path = Path(settings.MODEL_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_version_string() -> str:
    """Return a unique version string based on the model name prefix and UTC timestamp."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{settings.MODEL_NAME_PREFIX}_{ts}"


def save_model(pipeline: Pipeline, version: str | None = None) -> str:
    """Serialise *pipeline* to disk and return the version string.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline to persist.
    version:
        Optional explicit version string. A timestamp-based string is
        generated automatically when ``None`` is passed.
    """
    if version is None:
        version = build_version_string()
    filepath = _models_dir() / f"{version}.joblib"
    joblib.dump(pipeline, filepath)
    return version


def load_model(version: str | None = None) -> tuple[Pipeline, str]:
    """Load a trained pipeline from disk.

    Parameters
    ----------
    version:
        Specific version to load. When ``None``, the latest artefact is
        loaded automatically.

    Returns
    -------
    tuple[Pipeline, str]
        The loaded pipeline and its version string.

    Raises
    ------
    FileNotFoundError
        If the requested (or any) model artefact cannot be found.
    """
    if version is not None:
        filepath = _models_dir() / f"{version}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        return joblib.load(filepath), version
    version_str = get_latest_version()
    filepath = _models_dir() / f"{version_str}.joblib"
    return joblib.load(filepath), version_str


def get_latest_version() -> str:
    """Return the version string of the most recently saved model artefact.

    Raises
    ------
    FileNotFoundError
        If no ``.joblib`` files exist in the models directory.
    """
    models_path = _models_dir()
    files = sorted(models_path.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(
            f"No trained models found in {models_path}. "
            "Call POST /api/v1/train-model first."
        )
    return files[-1].stem