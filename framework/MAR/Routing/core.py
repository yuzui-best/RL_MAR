"""Public-only routing state, frozen actions, and RewardPLC evaluator projection."""
from __future__ import annotations

import hashlib
import ast
import math
import re

DOMAINS = ("SequentialLogicControl", "ProcessControl", "MotionAndSortingControl",
           "DataHandlingAndCommunication", "SafetyAndMonitoring")
MODELS = ("deepseek-v4-flash", "deepseek-v3.2", "qwen3.5-9b", "gpt-4o-mini", "gpt-4o", "gpt-5.5")
IDENTITIES = ("RequirementsAnalyzer", "PathAndDecisionPlanner", "STSafetyCodeImplementer", "STCodeGenerator")
BUDGETS = (4096, 8192, 16384)
ACTIONS = tuple({"identity": identity, "depth": depth}
                for identity in IDENTITIES for depth in (1, 2, 3)) + (
                    {"identity": "SafetyLogicDesigner", "depth": 1},
                    {"identity": "SafetyLogicDesigner", "depth": 2})
TERMS = {
    "SafetyAndMonitoring": ("emergency", "fault", "guard", "hazard", "interlock", "safe", "safety", "trip", "watchdog"),
    "MotionAndSortingControl": ("axis", "conveyor", "drive", "encoder", "jam", "motion", "position", "robot", "sort", "speed", "stepper", "target"),
    "DataHandlingAndCommunication": ("array", "buffer", "communication", "count", "crc", "data", "decode", "encode", "fifo", "message", "modbus", "protocol", "sample"),
    "ProcessControl": ("analog", "flow", "hysteresis", "level", "pid", "pressure", "pump", "tank", "temperature", "valve"),
    "SequentialLogicControl": ("latch", "permit", "sequence", "start", "state machine", "step", "stop"),
}


def public_view(task):
    view = {k: task[k] for k in ("instruction", "entry_point", "pou_interface")}
    # These fields are released public text used by the historical domain classifier.
    view.update({k: task[k] for k in ("category", "title") if k in task})
    return view


def domain(task):
    view = public_view(task)
    text = " ".join(str(view.get(k, "")) for k in ("category", "title", "instruction")).lower()
    scores = {d: sum(text.count(term) for term in terms) for d, terms in TERMS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else DOMAINS[0]


def features(task, model):
    view = public_view(task)
    text, ports = view["instruction"].lower(), view["pou_interface"]
    d = domain(view)
    return ([float(d == item) for item in DOMAINS]
            + [math.log1p(sum(p["section"] == section for p in ports)) for section in ("input", "output")]
            + [float(any("ARRAY" in p["type"].upper() for p in ports)),
               float(bool(re.search(r"timer|\bton\b|\btof\b|duration|seconds|milliseconds", text))),
               math.log1p(len(text))]
            + [float(bool(re.search(pattern, text))) for pattern in
               (r"state machine|\bstate\b|\bphase\b", r"edge|rising|falling", r"interlock|mutual|exclusive",
                r"deferred|delayed|next scan|previous scan|next invocation")]
            + [float(model == item) for item in MODELS])


def instruction_domain(task):
    return domain({k:task[k] for k in ("instruction", "entry_point", "pou_interface")})


def mask(task, available_roles=None):
    if available_roles is None:
        raise ValueError("domain's frozen available roles are required")
    result = []
    for a in ACTIONS:
        roles = chain(a)
        result.append(len(set(roles)) == len(roles) and all(r in available_roles for r in roles))
    return result


def chain(action):
    identity = action["identity"]
    if identity == "SafetyLogicDesigner" and action["depth"] == 2:
        return ["SafetyLogicDesigner", "STSafetyCodeImplementer"]
    return ([identity] if action["depth"] == 1 else [identity, "STCodeGenerator"]
            if action["depth"] == 2 else ["RequirementsAnalyzer", identity, "STCodeGenerator"])


def reward(passed, tokens, calls, cost_lambda=1.0, call_mu=0.02):
    return float(passed) - cost_lambda * tokens / 16384 - call_mu * calls


def clarified_specification(task):
    """Repair authored development contracts uniformly; no benchmark answers enter here."""
    original = task["specification"]
    parameters = ast.literal_eval(original.split("configured parameters: ", 1)[1].rstrip("."))
    contracts = {
        "SequentialLogicControl": "Retained Latched initially FALSE. On every invocation, if Stop OR NOT Permit set Latched=FALSE; otherwise if Start set Latched=TRUE; otherwise retain it. Publish Run=Latched after that update. Stop/loss of Permit takes priority over Start.",
        "ProcessControl": f"Retained Demand initially FALSE. If NOT Enable, clear Demand. Otherwise if Level <= {parameters['low']}, set Demand TRUE; otherwise if Level >= {parameters['high']}, set Demand FALSE; otherwise retain Demand. Publish Fill=Demand after the update. Independently of Enable, HighAlarm=(Level >= {parameters['alarm']}).",
        "MotionAndSortingControl": "Retained Moving and Done initially FALSE. Apply exactly this priority: if NOT Enable, clear Moving and Done; else if Jam, clear Moving and retain Done; else if AtTarget, clear Moving and set Done; else if NOT Done, set Moving; otherwise retain both values. Publish Motor=Moving after the update. Done remains latched until disabled.",
        "DataHandlingAndCommunication": f"Retained Samples initially zero. Reset has priority: if Reset, clear Samples; else if Valid, increment Samples by one; otherwise retain it. Publish Count=Samples after the update. Independently of Reset, OverLimit is TRUE exactly when Valid AND (Value > {parameters['limit']}); when Valid is FALSE, OverLimit is FALSE even if Value exceeds the limit.",
        "SafetyAndMonitoring": f"Retained FaultLatched initially FALSE. If EStop OR GuardOpen, set FaultLatched TRUE. Otherwise if Reset AND ResetCode={parameters['code']}, clear FaultLatched; otherwise retain it. Hazards take priority over reset. Publish Fault=FaultLatched and SafeEnable=NOT FaultLatched after the update; there is no additional startup-arm requirement.",
    }
    return original + "\n\nExplicit development execution contract:\n" + contracts[task["domain"]] + (
        " One scan is one invocation; retained state persists across scans of a case. Each case starts from a new instance. "
        "Internal state variable names are free; preserve the released input/output names and types.")


def convert_reward_task(task):
    """Keep released specification/I-O; score output traces, not invented internal names."""
    interface = [dict(section=section, name=name, type=kind)
                 for source, section in (("inputs", "input"), ("outputs", "output"))
                 for name, kind in task["pou_interface"][source].items()]
    instruction = clarified_specification(task)
    outputs = task["pou_interface"]["outputs"]
    cases = []
    for original in task["cases"]:
        scans, expected = original["scans"], original["expected"]
        initial = original["initial_state"]
        cases.append(dict(case_id=original["case_id"], description=original["kind"],
            initial_outputs={k: v for k, v in initial.items() if k in outputs}, initial_state={},
            input_trace={k: [scan[k] for scan in scans] for k in task["pou_interface"]["inputs"]},
            simulation_rounds=len(scans), oracle=dict(kind="exact_trace", sample_phase="after_cycle",
            expected_output_trace={k: [scan[k] for scan in expected] for k in outputs},
            expected_internal_state={}, real_tolerance=dict(absolute=0, relative=0))))
    return dict(task_id=task["task_id"], instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        entry_point="FB_" + task["task_id"].replace("-", "_"),
        pou_interface=interface, pou_declarations=interface, cases=cases, oracles=[], stimulus={},
        validation_mode="exact_trace", suite_version="rewardplc-public-contract-clarified-v2",
        category=task["domain"], family_id=task["family_id"], source_split=task["split"])
