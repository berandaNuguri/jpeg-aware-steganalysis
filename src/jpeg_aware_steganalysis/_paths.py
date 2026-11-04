"""Project-relative locations with private external storage overrides."""
import json
import os
from pathlib import Path, PurePosixPath

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"


def project_root():
    # Prefer this source checkout over another clone used as the working directory.
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/jpeg_aware_steganalysis").is_dir():
            return candidate
    raise RuntimeError("The source checkout could not be located")


def project_path(value="."):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def data_path(*parts):
    return project_path(os.environ.get("RESEARCH_DATA_ROOT", "src/data")).joinpath(*parts)


def model_path(*parts):
    return project_path(os.environ.get("RESEARCH_MODEL_ROOT", "checkpoints")).joinpath(*parts)


def path_candidates(value, base_dir=None):
    """Resolve current inputs or explicitly configured historical path prefixes.

    RESEARCH_PATH_MAP is a private JSON object mapping old prefixes to current
    dataset roots. No machine-specific aliases are built into the source.
    """
    value = os.path.expandvars(os.path.expanduser(str(value))).strip()
    raw = Path(value)
    candidates = []
    if base_dir is not None and not raw.is_absolute():
        candidates.append(project_path(base_dir) / raw)
    candidates.append(project_path(raw))
    mappings = json.loads(os.environ.get("RESEARCH_PATH_MAP", "{}"))
    if not isinstance(mappings, dict):
        raise ValueError("RESEARCH_PATH_MAP must be a JSON object of path prefixes")
    portable = PurePosixPath(value.replace("\\", "/"))
    for source, target in mappings.items():
        try:
            relative = portable.relative_to(PurePosixPath(str(source).replace("\\", "/")))
        except ValueError:
            continue
        candidates.append(project_path(target).joinpath(*relative.parts))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def model_name_or_path(value):
    """Keep Hugging Face model IDs; anchor explicit local paths to this checkout."""
    value = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(value)
    if path.is_absolute() or value.startswith(("./", "../", "checkpoints/", "src/", "hf_cache/")) or (project_root() / path).exists():
        if value.startswith("checkpoints/"):
            return str(model_path(value[len("checkpoints/"):]))
        return str(project_path(path))
    return value
