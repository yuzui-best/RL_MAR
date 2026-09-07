from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenManifest:
    schema_version: str
    phase: str
    files: Mapping[str, str]
    models: Sequence[str]
    settings: Mapping[str, object]

    @classmethod
    def create(cls, *, phase: str, files: Mapping[str, Path], models: Sequence[str], settings: Mapping[str, object]):
        if phase not in {"train", "evolution_dev", "selection", "test"}:
            raise ValueError(f"unsupported phase: {phase}")
        resolved = {name: Path(path).resolve() for name, path in files.items()}
        if phase != "test" and any("functional94" in str(path).lower() for path in resolved.values()):
            raise ValueError("Functional94 is test-only and cannot be frozen into a training manifest")
        missing = [str(path) for path in resolved.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return cls("rlmar-frozen-manifest-2.0", phase, {name: file_sha256(path) for name, path in resolved.items()}, tuple(models), dict(settings))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def verify(self, files: Mapping[str, Path]) -> None:
        expected_names = set(self.files)
        if set(files) != expected_names:
            raise ValueError("manifest file set changed")
        changed = [name for name, path in files.items() if file_sha256(Path(path)) != self.files[name]]
        if changed:
            raise ValueError(f"frozen files changed: {', '.join(sorted(changed))}")
