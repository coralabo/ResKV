# SPDX-License-Identifier: Apache-2.0
"""UTF-8 JSON files, input-preserving result paths and source revision."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def validate_output_paths(inputs, outputs):
    """Reject outputs that are directories, inputs, or each other."""
    inputs = {Path(path).resolve() for path in inputs}
    selected = set()
    for output in map(Path, outputs):
        if output.is_dir():
            raise ValueError(f"output must be a file: {output}")
        resolved = output.resolve()
        if resolved in inputs:
            raise ValueError(f"output would overwrite input: {output}")
        if resolved in selected:
            raise ValueError(f"output paths must be distinct: {output}")
        selected.add(resolved)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: JSONL rows must be objects")
            yield row


def write_json(path, value):
    """Replace a complete JSON report atomically; reject nonfinite numbers."""
    text = json.dumps(value, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return text


def git_commit():
    """Commit of this checkout, suffixed with ``-dirty`` for uncommitted changes."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def source_sha256():
    """Identify runtime sources, including uncommitted changes and installed wheels."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("reskv", "evaluation", "benchmarks"):
        for path in sorted((root / package).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".json"):
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()
