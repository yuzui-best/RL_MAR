from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


DOMAINS = (
    "SequentialLogicControl",
    "ProcessControl",
    "MotionAndSortingControl",
    "DataHandlingAndCommunication",
    "SafetyAndMonitoring",
)
REWARD_SPLITS = frozenset({"train", "evolution_dev", "selection"})
FORBIDDEN_TRAINING_NAMES = frozenset({"functional94", "plcdataset.json"})
GENERATOR_VERSION = "rewardplc-1.0.0"


def stable_sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", value.lower()))


def token_fingerprint(value: str) -> str:
    return stable_sha256(" ".join(normalized_tokens(value)))


@dataclass(frozen=True)
class Provenance:
    source: str
    license: str
    commit: str
    independently_authored: bool = True


@dataclass(frozen=True)
class ScanCase:
    case_id: str
    kind: str
    initial_state: Mapping[str, Any]
    scans: tuple[Mapping[str, Any], ...]
    expected: tuple[Mapping[str, Any], ...]
    required_activations: tuple[str, ...]


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    family_id: str
    domain: str
    difficulty: str
    split: str
    specification: str
    pou_interface: Mapping[str, Mapping[str, str]]
    reference_st: str
    cases: tuple[ScanCase, ...]
    provenance: Provenance
    generator_version: str
    seed: int
    verifier_versions: Mapping[str, str]
    reference_sha256: str = field(init=False)
    interface_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_sha256", token_fingerprint(self.reference_st))
        object.__setattr__(self, "interface_sha256", stable_sha256(canonical_json(self.pou_interface)))

    def validate(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain: {self.domain}")
        if self.split not in REWARD_SPLITS:
            raise ValueError(f"invalid RewardPLC split: {self.split}")
        if len(self.cases) < 12:
            raise ValueError(f"{self.task_id} has fewer than 12 hidden cases")
        if not self.reference_st.strip() or not self.specification.strip():
            raise ValueError(f"{self.task_id} has an empty specification or reference")
        if not self.provenance.license or not self.provenance.commit:
            raise ValueError(f"{self.task_id} lacks provenance")
        for case in self.cases:
            if not case.scans or len(case.scans) != len(case.expected):
                raise ValueError(f"{self.task_id}/{case.case_id} has an invalid oracle")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_rewardplc_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    if any(name in lowered or name in resolved.name.lower() for name in FORBIDDEN_TRAINING_NAMES):
        raise PermissionError(f"training/evolution cannot access test corpus: {resolved}")
    return resolved


def load_reward_records(path: str | Path, *, allowed_splits: Iterable[str] = REWARD_SPLITS) -> list[TaskRecord]:
    source = assert_rewardplc_path(path)
    allowed = frozenset(allowed_splits)
    if not allowed or not allowed <= REWARD_SPLITS:
        raise ValueError("allowed_splits must be a non-empty RewardPLC split subset")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["tasks"] if isinstance(payload, dict) else payload
    records: list[TaskRecord] = []
    for row in rows:
        provenance = Provenance(**row.pop("provenance"))
        cases = tuple(ScanCase(**case) for case in row.pop("cases"))
        stored_reference_hash = row.pop("reference_sha256", None)
        stored_interface_hash = row.pop("interface_sha256", None)
        record = TaskRecord(**row, provenance=provenance, cases=cases)
        record.validate()
        if record.split not in allowed:
            continue
        if stored_reference_hash and stored_reference_hash != record.reference_sha256:
            raise ValueError(f"reference hash mismatch for {record.task_id}")
        if stored_interface_hash and stored_interface_hash != record.interface_sha256:
            raise ValueError(f"interface hash mismatch for {record.task_id}")
        records.append(record)
    return records


def family_split(family_index: int) -> str:
    if not 0 <= family_index < 30:
        raise ValueError("family_index must be in [0, 29]")
    if family_index < 24:
        return "train"
    if family_index < 27:
        return "evolution_dev"
    return "selection"


def verify_family_disjoint(records: Sequence[TaskRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        previous = seen.setdefault(record.family_id, record.split)
        if previous != record.split:
            raise ValueError(f"family leakage: {record.family_id} occurs in {previous} and {record.split}")


def record_fingerprints(record: TaskRecord) -> dict[str, str]:
    trace = [
        {"initial": case.initial_state, "scans": case.scans, "expected": case.expected}
        for case in record.cases
    ]
    return {
        "specification": token_fingerprint(record.specification),
        "reference": record.reference_sha256,
        "interface": record.interface_sha256,
        "traces": stable_sha256(canonical_json(trace)),
    }


def find_exact_leakage(left: Sequence[TaskRecord], right: Sequence[TaskRecord]) -> list[dict[str, str]]:
    right_index: dict[tuple[str, str], str] = {}
    for record in right:
        for kind, digest in record_fingerprints(record).items():
            right_index[(kind, digest)] = record.task_id
    leaks: list[dict[str, str]] = []
    for record in left:
        for kind, digest in record_fingerprints(record).items():
            other = right_index.get((kind, digest))
            if other:
                leaks.append({"kind": kind, "left": record.task_id, "right": other})
    return leaks


def write_dataset(records: Sequence[TaskRecord], destination: str | Path) -> dict[str, Any]:
    verify_family_disjoint(records)
    for record in records:
        record.validate()
    path = assert_rewardplc_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: sum(record.split == split for record in records) for split in sorted(REWARD_SPLITS)}
    payload = {
        "name": "RewardPLC-750",
        "generator_version": GENERATOR_VERSION,
        "counts": counts,
        "tasks": [record.to_dict() for record in records],
    }
    body = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    path.write_text(body + "\n", encoding="utf-8")
    manifest = {
        "dataset": path.name,
        "sha256": stable_sha256(body + "\n"),
        "counts": counts,
        "families": len({record.family_id for record in records}),
        "tasks": len(records),
        "generator_version": GENERATOR_VERSION,
    }
    manifest_path = path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for split in sorted(REWARD_SPLITS):
        split_ids = [record.task_id for record in records if record.split == split]
        path.with_name(f"{split}.json").write_text(
            json.dumps({"split": split, "task_ids": split_ids}, indent=2) + "\n", encoding="utf-8"
        )
    return manifest
