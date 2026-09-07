from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping

from .dataset import DOMAINS, GENERATOR_VERSION, Provenance, ScanCase, TaskRecord, family_split


@dataclass(frozen=True)
class Template:
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    internal: Mapping[str, str]
    specification: str
    body: str


_TEMPLATES = {
    "SequentialLogicControl": Template(
        {"Start": "BOOL", "Stop": "BOOL", "Permit": "BOOL"}, {"Run": "BOOL"}, {"Latched": "BOOL"},
        "Latch a run command after Start while Permit is true; Stop or loss of Permit clears it.",
        "IF Stop OR NOT Permit THEN Latched := FALSE; ELSIF Start THEN Latched := TRUE; END_IF; Run := Latched;",
    ),
    "ProcessControl": Template(
        {"Level": "INT", "Enable": "BOOL"}, {"Fill": "BOOL", "HighAlarm": "BOOL"}, {"Demand": "BOOL"},
        "Maintain a level using hysteresis while enabled and raise an independent high alarm.",
        "IF NOT Enable THEN Demand := FALSE; ELSIF Level <= {low} THEN Demand := TRUE; ELSIF Level >= {high} THEN Demand := FALSE; END_IF; Fill := Demand; HighAlarm := Level >= {alarm};",
    ),
    "MotionAndSortingControl": Template(
        {"Enable": "BOOL", "AtTarget": "BOOL", "Jam": "BOOL"}, {"Motor": "BOOL", "Done": "BOOL"}, {"Moving": "BOOL"},
        "Drive toward a target while enabled; a jam stops motion and target arrival latches completion until disabled.",
        "IF NOT Enable THEN Moving := FALSE; Done := FALSE; ELSIF Jam THEN Moving := FALSE; ELSIF AtTarget THEN Moving := FALSE; Done := TRUE; ELSIF NOT Done THEN Moving := TRUE; END_IF; Motor := Moving;",
    ),
    "DataHandlingAndCommunication": Template(
        {"Valid": "BOOL", "Value": "INT", "Reset": "BOOL"}, {"Count": "INT", "OverLimit": "BOOL"}, {"Samples": "INT"},
        "Count valid samples, reset on command, and report whether the current value exceeds a configured limit.",
        "IF Reset THEN Samples := 0; ELSIF Valid THEN Samples := Samples + 1; END_IF; Count := Samples; OverLimit := Valid AND (Value > {limit});",
    ),
    "SafetyAndMonitoring": Template(
        {"EStop": "BOOL", "GuardOpen": "BOOL", "Reset": "BOOL", "ResetCode": "INT"}, {"SafeEnable": "BOOL", "Fault": "BOOL"}, {"FaultLatched": "BOOL"},
        "Latch a safety fault on emergency stop or open guard; clear only by Reset with the configured reset code when both hazards are absent.",
        "IF EStop OR GuardOpen THEN FaultLatched := TRUE; ELSIF Reset AND (ResetCode = {code}) THEN FaultLatched := FALSE; END_IF; Fault := FaultLatched; SafeEnable := NOT FaultLatched;",
    ),
}


def _interface(template: Template) -> dict[str, Mapping[str, str]]:
    return {"inputs": dict(template.inputs), "outputs": dict(template.outputs), "internal": dict(template.internal)}


def _st(domain: str, task_id: str, parameters: Mapping[str, int]) -> str:
    template = _TEMPLATES[domain]
    declaration = [f"FUNCTION_BLOCK FB_{task_id}", "VAR_INPUT"]
    declaration.extend(f"    {name} : {kind};" for name, kind in template.inputs.items())
    declaration.append("END_VAR\nVAR_OUTPUT")
    declaration.extend(f"    {name} : {kind};" for name, kind in template.outputs.items())
    declaration.append("END_VAR\nVAR")
    declaration.extend(f"    {name} : {kind};" for name, kind in template.internal.items())
    declaration.append("END_VAR")
    body = template.body.format(**parameters)
    return "\n".join(declaration) + "\n" + body + "\nEND_FUNCTION_BLOCK\n"


def _step(domain: str, state: dict[str, Any], inputs: Mapping[str, Any], p: Mapping[str, int]) -> dict[str, Any]:
    if domain == "SequentialLogicControl":
        if inputs["Stop"] or not inputs["Permit"]:
            state["Latched"] = False
        elif inputs["Start"]:
            state["Latched"] = True
        return {"Run": state["Latched"], "Latched": state["Latched"]}
    if domain == "ProcessControl":
        if not inputs["Enable"]:
            state["Demand"] = False
        elif inputs["Level"] <= p["low"]:
            state["Demand"] = True
        elif inputs["Level"] >= p["high"]:
            state["Demand"] = False
        return {"Fill": state["Demand"], "HighAlarm": inputs["Level"] >= p["alarm"], "Demand": state["Demand"]}
    if domain == "MotionAndSortingControl":
        if not inputs["Enable"]:
            state.update(Moving=False, Done=False)
        elif inputs["Jam"]:
            state["Moving"] = False
        elif inputs["AtTarget"]:
            state.update(Moving=False, Done=True)
        elif not state["Done"]:
            state["Moving"] = True
        return {"Motor": state["Moving"], "Done": state["Done"], "Moving": state["Moving"]}
    if domain == "DataHandlingAndCommunication":
        if inputs["Reset"]:
            state["Samples"] = 0
        elif inputs["Valid"]:
            state["Samples"] += 1
        return {"Count": state["Samples"], "OverLimit": inputs["Valid"] and inputs["Value"] > p["limit"], "Samples": state["Samples"]}
    if inputs["EStop"] or inputs["GuardOpen"]:
        state["FaultLatched"] = True
    elif inputs["Reset"] and inputs["ResetCode"] == p["code"]:
        state["FaultLatched"] = False
    return {"SafeEnable": not state["FaultLatched"], "Fault": state["FaultLatched"], "FaultLatched": state["FaultLatched"]}


def _initial(domain: str) -> dict[str, Any]:
    return {
        "SequentialLogicControl": {"Latched": False},
        "ProcessControl": {"Demand": False},
        "MotionAndSortingControl": {"Moving": False, "Done": False},
        "DataHandlingAndCommunication": {"Samples": 0},
        "SafetyAndMonitoring": {"FaultLatched": False},
    }[domain].copy()


def _scan_sequences(domain: str, p: Mapping[str, int], seed: int) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    sequences: list[list[dict[str, Any]]] = []
    for case_index in range(12):
        scans: list[dict[str, Any]] = []
        for scan_index in range(6):
            if domain == "SequentialLogicControl":
                scans.append({
                    "Start": scan_index == 0 and (case_index % 3 == 0 or rng.random() < 0.2),
                    "Stop": scan_index == 2 and (case_index % 4 == 0 or rng.random() < 0.15),
                    "Permit": case_index % 5 != 0 and rng.random() >= 0.1,
                })
            elif domain == "ProcessControl":
                levels = [p["low"] - 1, p["low"] + 1, p["high"] - 1, p["high"] + 1, p["alarm"] - 1, p["alarm"] + 1]
                scans.append({"Level": levels[(case_index + scan_index) % 6], "Enable": case_index % 6 != 0})
            elif domain == "MotionAndSortingControl":
                scans.append({
                    "Enable": case_index % 6 != 0 and rng.random() >= 0.08,
                    "AtTarget": scan_index == 2 and (case_index % 2 == 0 or rng.random() < 0.15),
                    "Jam": scan_index == 1 and (case_index % 5 == 0 or rng.random() < 0.12),
                })
            elif domain == "DataHandlingAndCommunication":
                scans.append({"Valid": (case_index + scan_index) % 3 != 0, "Value": p["limit"] + rng.randint(-2, 2), "Reset": scan_index == 0 and case_index % 5 == 0})
            else:
                scans.append({
                    "EStop": scan_index == 1 and (case_index % 4 == 0 or rng.random() < 0.1),
                    "GuardOpen": scan_index == 2 and (case_index % 5 == 0 or rng.random() < 0.1),
                    "Reset": scan_index == 3 and rng.random() >= 0.1,
                    "ResetCode": p["code"] if case_index % 3 else p["code"] + 1,
                })
        sequences.append(scans)
    return sequences


def _cases(domain: str, p: Mapping[str, int], seed: int) -> tuple[ScanCase, ...]:
    result: list[ScanCase] = []
    for index, scans in enumerate(_scan_sequences(domain, p, seed)):
        state = _initial(domain)
        initial = state.copy()
        expected = tuple(_step(domain, state, scan, p) for scan in scans)
        result.append(ScanCase(f"case_{index:02d}", ("normal", "boundary", "recovery", "invalid")[index % 4], initial, tuple(scans), expected, (f"branch_{index % 4}",)))
    return tuple(result)


def _parameters(domain_index: int, family_index: int, variant: int) -> dict[str, int]:
    base = 10 + domain_index * 1000 + family_index * 10 + variant
    return {"low": base, "high": base + 5, "alarm": base + 9, "limit": base + 2, "code": base + 17}


def generate_rewardplc() -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for domain_index, domain in enumerate(DOMAINS):
        template = _TEMPLATES[domain]
        for family_index in range(30):
            family_id = f"{domain_index + 1:02d}-{family_index + 1:02d}"
            for variant in range(5):
                seed = 20260903 + domain_index * 10000 + family_index * 100 + variant
                task_id = f"rp-{family_id}-v{variant + 1}"
                parameters = _parameters(domain_index, family_index, variant)
                specification = f"{template.specification} Family {family_index + 1}, variant {variant + 1}; configured parameters: {parameters}."
                records.append(TaskRecord(
                    task_id=task_id,
                    family_id=family_id,
                    domain=domain,
                    difficulty=("basic", "intermediate", "advanced")[family_index % 3],
                    split=family_split(family_index),
                    specification=specification,
                    pou_interface=_interface(template),
                    reference_st=_st(domain, task_id.replace("-", "_"), parameters),
                    cases=_cases(domain, parameters, seed),
                    provenance=Provenance("independently-authored RewardPLC template", "CC-BY-4.0", GENERATOR_VERSION),
                    generator_version=GENERATOR_VERSION,
                    seed=seed,
                    verifier_versions={"oracle": GENERATOR_VERSION, "rusty": "record-at-validation", "preprocessor": "record-at-validation"},
                ))
    return records


def mutation_kill_rate(record: TaskRecord) -> float:
    domain = record.domain
    family_index = int(record.family_id.split("-")[1]) - 1
    variant = int(record.task_id.rsplit("v", 1)[1]) - 1
    base = 10 + DOMAINS.index(domain) * 1000 + family_index * 10 + variant
    parameters = {"low": base, "high": base + 5, "alarm": base + 9, "limit": base + 2, "code": base + 17}
    def mutate(observed: dict[str, Any], mutation_index: int) -> dict[str, Any]:
        changed = dict(observed)
        keys = tuple(changed)
        key = keys[mutation_index % len(keys)]
        value = changed[key]
        if isinstance(value, bool):
            changed[key] = not value
        elif isinstance(value, int):
            changed[key] = value + (1 if mutation_index % 2 == 0 else -1)
        return changed

    killed = 0
    for mutation_index in range(4):
        differs = False
        for case in record.cases:
            state = dict(case.initial_state)
            observed = tuple(mutate(_step(domain, state, scan, parameters), mutation_index) for scan in case.scans)
            if observed != case.expected:
                differs = True
                break
        killed += int(differs)
    return killed / 4
