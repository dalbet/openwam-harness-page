"""Candidate-decision planner backed by the resident d1-3B model.

d1 is a single-forward-pass decision model: it answers named choice/noul
questions over one state and never writes free text. This module therefore
does what a planner needs around it:

1. build the candidate space from the **live scene registry** (movable objects,
   placement sites, drawer regions, appliances) - never from tasks.json's
   registry objects/targets or BDDL goals;
2. ask staged, legal questions (operation -> source -> target -> relation),
   with composite requests split into an open target / object / place target
   triple that must agree on the same drawer owner;
3. validate the returned bindings against the scene, the relation semantics and
   the user's own wording, and reject ambiguity / unsupported / incomplete
   requests instead of guessing;
4. assemble one legal ``whole_task`` step whose prompt is the user's own
   instruction, so nothing is rewritten from an answer key.

The candidate list is the full scene, so no candidate encodes "the answer for
task N": the model has to read the user instruction to choose.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from common.plan_contract import validate_plan

OPERATIONS = ("pick_place", "open_drawer", "push_object", "turn_on", "composite", "other")

OPERATION_DESC = {
    "pick_place": "grasp one movable object and put it on or in a target",
    "open_drawer": "open one drawer of a cabinet",
    "push_object": "push a movable object along a surface without grasping it",
    "turn_on": "turn on an appliance",
    "composite": "the request contains two or more of these subgoals, such as opening a drawer and putting something inside",
    "other": "none of the above; the request is not a supported manipulation",
}

GENERIC_NAME_WORDS = {
    "main",
    "default",
    "site",
    "region",
    "side",
    "level",
    "cook",
    "front",
    "top",
    "middle",
    "bottom",
    "table",
    "the",
}

NAME_ALIASES = {
    "cabinet": {"drawer"},
    "drawer": {"cabinet"},
    "stove": {"cooktop", "burner"},
    "rack": {"shelf"},
}

CONTAINER_WORDS = {"bowl", "basket", "bin", "pot", "cup", "mug", "container", "drawer"}

RELATION_PATTERNS = {
    "on": r"\b(on|onto|atop|on top of)\b",
    "in": r"\b(in|into|inside|within)\b",
    "open": r"\bopen\b",
    "turn_on": r"\b(turn on|switch on|activate|turn\b[^.,;]{0,40}\bon)\b",
}

# A low-confidence operation answer may still be used when the user's own main
# verb names that exact operation. This is a fallback for model hesitation, not
# a planner: source, target and relation still come from d1 and still have to
# pass every validation below.
VERB_PATTERNS = {
    "open_drawer": r"\bopen\b",
    "push_object": r"\b(push|slide)\b",
    "turn_on": RELATION_PATTERNS["turn_on"],
    "pick_place": r"\b(put|place|set|move|drop)\b",
    "composite": r"\b(open|put|place|set|move|drop)\b",
}

DEFAULT_THRESHOLDS = {
    "supported_min": 0.5,
    "operation_min": 0.5,
    "operation_verb_min": 0.3,
    "specified_min": 0.5,
    "relation_min": 0.5,
    "source_min": 0.5,
    "target_min": 0.5,
    # The top-1 candidate must also lead the runner-up by this margin. The
    # frozen calibration set (18 correct positives) had a minimum source margin
    # of 0.297 and target margin of 0.412, so 0.2 keeps every calibrated
    # binding while rejecting a split top-2 decision.
    "margin_min": 0.2,
}


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"d1 HTTP {exc.code}: {detail}") from exc


def _get_json(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# candidates from the live scene
# ---------------------------------------------------------------------------


def _humanize(name: str) -> str:
    text = re.sub(r"_\d+$", "", str(name))
    return re.sub(r"[_\s]+", " ", text).strip()


def _drawer_parts(name: str) -> tuple[str | None, str | None]:
    text = str(name or "")
    for level in ("top", "middle", "bottom"):
        suffix = f"_{level}_region"
        if text.endswith(suffix):
            return (text[: -len(suffix)] or None), level
    return None, None


def _drawer_regions(snapshot: dict[str, Any]) -> list[str]:
    """Sites whose owner has a matching ``*_level`` joint - real drawers."""
    joints = snapshot.get("joints") or {}
    sites = ((snapshot.get("named_poses") or {}).get("sites") or {})
    out = []
    for name in sites:
        owner, level = _drawer_parts(name)
        if owner and level and f"{owner}_{level}_level" in joints:
            out.append(name)
    return sorted(out)


def _appliances(snapshot: dict[str, Any]) -> list[str]:
    """Owners that expose a ``*_button`` joint; derived from the scene, not tasks.json."""
    joints = snapshot.get("joints") or {}
    out = []
    for name in joints:
        if name.endswith("_button"):
            out.append(name[: -len("_button")])
    return sorted(set(out))


def _public_sites(snapshot: dict[str, Any]) -> list[str]:
    sites = ((snapshot.get("named_poses") or {}).get("sites") or {})
    out = []
    for name in sites:
        if name.startswith("robot0") or name.startswith("gripper0"):
            continue
        if name.endswith("_default_site"):
            continue
        out.append(name)
    return sorted(out)


def _site_xy(snapshot: dict[str, Any], name: str) -> tuple[float, float] | None:
    pose = (((snapshot.get("named_poses") or {}).get("sites") or {}).get(name))
    if isinstance(pose, dict):
        pose = pose.get("position") or pose.get("pos")
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    return None


def _dedupe_targets(snapshot: dict[str, Any], targets: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop a waypoint that names the exact same point as a region.

    LIBERO scenes expose e.g. ``flat_stove_1_burner`` (site, z=0.927) and
    ``flat_stove_1_cook_region`` (region, z=0.905) at identical x/y. Two names
    for one physical spot split the model's mass and make an unambiguous
    request look uncertain, so the region name is kept and the duplicate site
    is not offered. Targets at genuinely different points (cabinet top side vs
    top drawer region) are only millimetres apart and are both kept.
    """
    by_point: dict[tuple[int, int], int] = {}
    kept: list[dict[str, str]] = []
    for row in targets:
        if row["kind"] not in {"region", "surface", "site"}:
            kept.append(row)
            continue
        xy = _site_xy(snapshot, row["name"])
        key = None if xy is None else (round(xy[0] * 1000), round(xy[1] * 1000))
        if key is None or key not in by_point:
            if key is not None:
                by_point[key] = len(kept)
            kept.append(row)
            continue
        index = by_point[key]
        if row["kind"] == "region" and kept[index]["kind"] != "region":
            kept[index] = row
    return kept


def _site_description(name: str, drawers: set[str], appliances: set[str]) -> str:
    owner, level = _drawer_parts(name)
    if name in drawers and owner:
        return f"{_humanize(owner)} {level} drawer: an openable drawer"
    if name.endswith("_cook_region"):
        return f"the cooking surface of {_humanize(name[: -len('_cook_region')])}"
    if name.endswith("_burner"):
        return f"a burner on {_humanize(name[: -len('_burner')])}"
    if name.startswith("main_table_") and name.endswith("_region"):
        where = name[len("main_table_") : -len("_region")].replace("_", " ")
        return f"an empty spot on the main table near the {where}"
    if name.endswith("_top_side"):
        return f"the top surface of {_humanize(name[: -len('_top_side')])}"
    if name.endswith("_top_region"):
        return f"the top of {_humanize(name[: -len('_top_region')])}"
    if name == "table_top":
        return "the top of the main table"
    for appliance in appliances:
        if name in {f"{appliance}_button"}:
            return f"the control of {_humanize(appliance)}"
    if name.endswith("_region"):
        return f"a placement region on {_humanize(name[: -len('_region')])}"
    if name.endswith("_side"):
        return f"a flat surface on {_humanize(name[: -len('_side')])}"
    return f"a placement site on {_humanize(name)}"


def _object_description(name: str) -> str:
    """Affordance from the noun: what the object can be used as, not what to do."""
    thing = _humanize(name)
    tokens = set(re.split(r"[_\s]+", re.sub(r"_\d+$", "", str(name).lower())))
    if tokens & CONTAINER_WORDS:
        return f"{thing}: a movable container that can hold other objects inside"
    return f"{thing}: a movable object that can support other objects on top"


def build_candidates(snapshot: dict[str, Any]) -> dict[str, Any]:
    sim = snapshot.get("sim") or {}
    objects = sorted((sim.get("objects") or {}).keys())
    drawers = _drawer_regions(snapshot)
    appliances = _appliances(snapshot)
    sites = _public_sites(snapshot)
    site_set = set(sites)
    drawer_set, appliance_set = set(drawers), set(appliances)

    targets: list[dict[str, str]] = []
    for name in sites:
        if name in drawer_set:
            kind = "drawer_region"
        elif name in {f"{a}_button" for a in appliances}:
            kind = "appliance_control"
        elif name.endswith("_region"):
            kind = "region"
        elif name.endswith("_side"):
            kind = "surface"
        else:
            kind = "site"
        targets.append({"name": name, "description": _site_description(name, drawer_set, appliance_set), "kind": kind})
    for name in objects:
        if name in site_set:
            continue
        targets.append(
            {
                "name": name,
                "description": _object_description(name),
                "kind": "object",
            }
        )
    for name in appliances:
        if name not in site_set and name not in objects:
            targets.append(
                {"name": name, "description": f"{_humanize(name)}: an appliance", "kind": "appliance"}
            )
    targets = _dedupe_targets(snapshot, targets)
    return {
        "objects": [{"name": name, "description": f"{_humanize(name)}: a movable object"} for name in objects],
        "targets": targets,
        "drawers": [
            {"name": name, "description": _site_description(name, drawer_set, appliance_set)}
            for name in drawers
        ],
        "appliances": [
            {"name": name, "description": f"{_humanize(name)}: an appliance"} for name in appliances
        ],
    }


# ---------------------------------------------------------------------------
# decision calls
# ---------------------------------------------------------------------------


def _ask(client, state: dict[str, Any], questions: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.system_one(state, questions)
    elapsed = time.perf_counter() - started
    if response.get("status") != "ok":
        raise RuntimeError(f"d1 decision failed: {response.get('error') or response.get('status')}")
    return response, elapsed


class D1Client:
    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            f"{self.url}/system_one",
            {"state": state, "questions": questions},
            self.timeout_s,
        )

    def health(self, timeout_s: float = 2.0) -> dict[str, Any]:
        return _get_json(f"{self.url}/health", timeout_s=timeout_s)


def _question_state(instruction: str, candidates: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_request": instruction,
        "scene": {
            "movable_objects": candidates["objects"],
            "placement_targets": candidates["targets"],
        },
    }


def _stage1_questions(candidates: dict[str, Any]) -> dict[str, Any]:
    source_options = {row["name"]: row["description"] for row in candidates["objects"]}
    source_options["none"] = "the request does not ask to move a specific listed object"
    return {
        "supported": {
            "type": "noul",
            "instructions": (
                "Is this request a robot manipulation task that one of these operations can carry out: "
                "pick and place an object, open a drawer, push an object, turn on an appliance, "
                "or a combination of these? Answer yes only if the requested action is supported."
            ),
        },
        "operation": {
            "type": "choice",
            "instructions": "Which operation does the user request ask for?",
            "criteria": {name: OPERATION_DESC[name] for name in OPERATIONS},
        },
        "source": {
            "type": "choice",
            "instructions": "Which movable object does the request act on?",
            "criteria": source_options,
        },
        "source_specified": {
            "type": "noul",
            "instructions": (
                "Does the request name a specific movable object from the listed objects? "
                "Answer no for pronouns like 'it', or when no object is named."
            ),
        },
        "ambiguous": {
            "type": "noul",
            "instructions": (
                "Do two or more listed objects or targets equally match the referring expression "
                "in the request? Answer yes when the request cannot be bound to one item."
            ),
        },
    }


def _stage2_questions(
    operation: str, candidates: dict[str, Any], source: str | None, instruction: str
) -> tuple[dict[str, Any], str | None]:
    if operation == "pick_place":
        relation_hint = _relation_hint(instruction, operation)
        options = _pick_place_targets(candidates, source, relation_hint, instruction)
        questions: dict[str, Any] = {}
        if relation_hint is None:
            questions["relation"] = {
                "type": "choice",
                "instructions": "What support relation does the request ask for?",
                "criteria": {
                    "on": "the object rests on a surface or another object",
                    "in": "the object goes inside a container or drawer",
                },
            }
        questions["target"] = {
            "type": "choice",
            "instructions": "Where should the object be placed?",
            "criteria": {row["name"]: row["description"] for row in options}
            | {"none": "no placement target is specified"},
        }
        questions["target_specified"] = {
            "type": "noul",
            "instructions": "Does the request name a specific destination from the listed targets?",
        }
        return questions, relation_hint
    if operation == "push_object":
        options = _push_targets(candidates, source, instruction)
        return {
            "target": {
                "type": "choice",
                "instructions": "Where should the object be pushed to?",
                "criteria": {row["name"]: row["description"] for row in options}
                | {"none": "no destination is specified"},
            },
            "target_specified": {
                "type": "noul",
                "instructions": "Does the request name a specific destination from the listed targets?",
            },
        }, "on"
    if operation == "open_drawer":
        drawers = _filter_drawer_level(candidates["drawers"], instruction)
        return {
            "target": {
                "type": "choice",
                "instructions": "Which drawer does the request ask to open?",
                "criteria": {row["name"]: row["description"] for row in drawers}
                | {"none": "no drawer is specified"},
            },
            "target_specified": {
                "type": "noul",
                "instructions": "Does the request name a specific drawer from the listed drawers?",
            },
        }, "open"
    if operation == "turn_on":
        return {
            "target": {
                "type": "choice",
                "instructions": "Which appliance does the request ask to turn on?",
                "criteria": {row["name"]: row["description"] for row in candidates["appliances"]}
                | {"none": "no appliance is specified"},
            },
            "target_specified": {
                "type": "noul",
                "instructions": "Does the request name a specific appliance from the listed appliances?",
            },
        }, "turn_on"
    # composite: two subgoals, selected together so they cannot contradict.
    drawers = _filter_drawer_level(candidates["drawers"], instruction)
    return {
        "open_target": {
            "type": "choice",
            "instructions": "Which drawer does the request ask to open?",
            "criteria": {row["name"]: row["description"] for row in drawers}
            | {"none": "no drawer is specified"},
        },
        "object": {
            "type": "choice",
            "instructions": "Which movable object should be put inside the opened drawer?",
            "criteria": {row["name"]: row["description"] for row in candidates["objects"]}
            | {"none": "no object is specified"},
        },
        "place_target": {
            "type": "choice",
            "instructions": "Which drawer should receive the object?",
            "criteria": {row["name"]: row["description"] for row in drawers}
            | {"none": "no destination is specified"},
        },
        "target_specified": {
            "type": "noul",
            "instructions": "Does the request name a specific drawer and a specific object to place inside it?",
        },
    }, "in"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _name_tokens(name: str) -> set[str]:
    tokens = set()
    for token in re.split(r"[_\s]+", re.sub(r"_\d+$", "", str(name).lower())):
        if token and token not in GENERIC_NAME_WORDS:
            tokens.add(token)
    return tokens


def _instruction_tokens(instruction: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", instruction.lower()))


def _covered(instruction: str, name: str | None, *, drawer: bool = False) -> bool:
    if not name:
        return False
    tokens = _name_tokens(name)
    if name == "table_top":
        # "table" and "top" are both generic words, so the token filter would
        # otherwise leave the main table with no way to be named at all.
        tokens = {"table"}
    if drawer:
        tokens |= {"drawer", "cabinet"}
    words = _instruction_tokens(instruction)
    for token in tokens:
        if token in words:
            return True
        if words & NAME_ALIASES.get(token, set()):
            return True
    return False


def _relation_in_instruction(instruction: str, relation: str) -> bool:
    pattern = RELATION_PATTERNS.get(relation)
    if pattern is None:
        return True
    return re.search(pattern, instruction.lower()) is not None


def _noul(answer: dict[str, Any]) -> float:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return 0.0
    return float(answer.get("noul") or 0.0)


def _target_kind(target: str | None, candidates: dict[str, Any]) -> str | None:
    for row in candidates["targets"]:
        if row["name"] == target:
            return row["kind"]
    return None


def _raw_tokens(name: str) -> set[str]:
    return set(re.split(r"[_\s]+", str(name).lower()))


def _is_container(name: str) -> bool:
    return bool(_raw_tokens(re.sub(r"_\d+$", "", str(name).lower())) & CONTAINER_WORDS)


def _relation_hint(instruction: str, operation: str) -> str | None:
    """Use the user's own preposition when it names exactly one support relation.

    "in front of" is a spatial phrase, not containment, so it is removed before
    matching. Both "on" and "in" present means the request itself is ambiguous
    and the model is asked instead.
    """
    if operation == "composite":
        return "in"
    if operation != "pick_place":
        return None
    text = re.sub(r"\bin front of\b", " ", instruction.lower())
    on = re.search(RELATION_PATTERNS["on"], text) is not None
    inside = re.search(RELATION_PATTERNS["in"], text) is not None
    if on and not inside:
        return "on"
    if inside and not on:
        return "in"
    return None


def _pick_place_targets(
    candidates: dict[str, Any], source: str | None, relation: str | None, instruction: str
) -> list[dict[str, str]]:
    """Legal targets for the named support relation.

    A drawer is only a placement target for ``in``; an appliance control is
    never a placement target. For ``in`` the remaining choices are containers
    and drawers; for ``on`` they are surfaces, regions, table top and support
    objects. When the relation is not yet known, the union is offered so the
    relation question can still be asked.
    """
    words = _instruction_tokens(instruction)
    rows: list[dict[str, str]] = []
    for row in candidates["targets"]:
        if row["name"] == source:
            continue
        kind = row["kind"]
        if kind == "appliance_control":
            continue
        if kind == "drawer_region":
            if relation in (None, "in"):
                rows.append(row)
            continue
        if relation == "in":
            if kind == "object" and _is_container(row["name"]):
                rows.append(row)
            continue
        if kind in {"region", "surface", "site", "object"}:
            rows.append(row)
    if "front" in words:
        # "in front of the stove" names the table area in front of the stove,
        # not the cooking surface or the other landmark regions.
        narrowed = [row for row in rows if row["kind"] != "region" or "front" in row["name"].lower()]
        rows = narrowed or rows
    if relation in (None, "in"):
        rows = _filter_drawer_level(rows, instruction)
    return rows


def _filter_drawer_level(rows: list[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    """Keep the drawer level the user named; leave other rows untouched."""
    levels = [level for level in ("top", "middle", "bottom") if re.search(rf"\b{level}\b", instruction.lower())]
    if len(set(levels)) != 1:
        return rows
    level = levels[0]
    kept = [
        row
        for row in rows
        if row.get("kind", "drawer_region") != "drawer_region"
        or _drawer_parts(row["name"])[1] == level
    ]
    has_drawer = any(row.get("kind", "drawer_region") == "drawer_region" for row in kept)
    return kept if has_drawer else rows


def _drawer_level_missing(instruction: str) -> bool:
    return (
        re.search(r"\b(cabinet|drawer)s?\b", instruction.lower()) is not None
        and re.search(r"\b(top|middle|bottom)\b", instruction.lower()) is None
    )


def _push_targets(
    candidates: dict[str, Any], source: str | None, instruction: str
) -> list[dict[str, str]]:
    rows = [
        row
        for row in candidates["targets"]
        if row["kind"] in {"region", "surface", "site"} and row["name"] != source
    ]
    if "front" in _instruction_tokens(instruction):
        narrowed = [row for row in rows if "front" in row["name"].lower()]
        rows = narrowed or rows
    return rows


def _choice_full(answer: dict[str, Any]) -> tuple[str | None, float, float]:
    """Return (choice, confidence, margin over the runner-up)."""
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return None, 0.0, 0.0
    choice = answer.get("choice")
    confidence = float(answer.get("confidence") or 0.0)
    probabilities = answer.get("probabilities") or {}
    others = [
        float(value)
        for name, value in probabilities.items()
        if name != choice and isinstance(value, (int, float))
    ]
    margin = confidence - max(others) if others else confidence
    return choice, confidence, max(0.0, margin)


def _reject(status: str, code: str, detail: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, "reason_code": code, "reason": detail, "evidence": evidence, "plan": None}


def plan_with_d1(
    *,
    instruction: str,
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    d1_url: str,
    task_id: int = 0,
    thresholds: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Decide bindings with d1, validate them, and return a whole_task plan or a rejection.

    ``client`` answers the same ``system_one(state, questions)`` protocol as the
    d1 HTTP service. The offline comparison passes a Qwen-backed client so both
    models run through identical candidate assembly, thresholds and validation;
    production always uses the resident d1 service.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    candidates = build_candidates(snapshot)
    client = client or D1Client(d1_url, timeout_s=timeout_s)
    started = time.perf_counter()
    state = _question_state(instruction, candidates)
    stage1, stage1_s = _ask(client, state, _stage1_questions(candidates), timeout_s)
    answers = stage1.get("answers") or {}
    supported = _noul(answers.get("supported"))
    operation, operation_conf, operation_margin = _choice_full(answers.get("operation"))
    source, source_conf, source_margin = _choice_full(answers.get("source"))
    source_specified = _noul(answers.get("source_specified"))
    ambiguous = _noul(answers.get("ambiguous"))
    evidence: dict[str, Any] = {
        "candidates": {
            "objects": [row["name"] for row in candidates["objects"]],
            "targets": [row["name"] for row in candidates["targets"]],
            "drawers": [row["name"] for row in candidates["drawers"]],
            "appliances": [row["name"] for row in candidates["appliances"]],
        },
        "stage1": {
            "supported": supported,
            "operation": operation,
            "operation_confidence": operation_conf,
            "operation_margin": operation_margin,
            "source": source,
            "source_confidence": source_conf,
            "source_margin": source_margin,
            "source_specified": source_specified,
            "ambiguous": ambiguous,
            "answers": answers,
            "usage": stage1.get("usage"),
        },
        "stage1_s": round(stage1_s, 4),
        "thresholds": limits,
    }
    if supported < limits["supported_min"]:
        return _reject("unsupported", "request_not_supported", "d1 judged the request outside the supported operations", evidence)
    if operation in (None, "other"):
        return _reject("unsupported", "operation_not_supported", f"operation={operation!r} confidence={operation_conf:.3f}", evidence)
    verb = VERB_PATTERNS.get(operation)
    verb_ok = bool(verb and re.search(verb, instruction.lower()))
    operation_source = "d1"
    if operation_conf < limits["operation_min"]:
        # A low-confidence decision is still usable when the user's own main
        # verb names that exact operation; the deterministic relation/goal
        # checks later still have to pass. Otherwise it is not supported.
        if verb_ok and operation_conf >= limits["operation_verb_min"]:
            operation_source = "instruction_verb"
        else:
            return _reject("unsupported", "operation_not_supported", f"operation={operation!r} confidence={operation_conf:.3f}", evidence)
    evidence["stage1"]["operation_source"] = operation_source
    # The ambiguity noul is kept as evidence but is not a hard gate: on this
    # model it saturates near 0.62 even for unambiguous requests. The choice
    # margin plus the deterministic drawer-level checks carry that decision.
    needs_source = operation in {"pick_place", "push_object", "composite"}
    if needs_source and (source in (None, "none") or source_specified < limits["specified_min"]):
        return _reject(
            "missing_information",
            "source_not_specified",
            f"source={source!r} specified P(yes)={source_specified:.3f}",
            evidence,
        )
    if needs_source and source_conf < limits["source_min"]:
        return _reject(
            "needs_clarification",
            "source_uncertain",
            f"source={source!r} confidence={source_conf:.3f}",
            evidence,
        )
    if needs_source and source_margin < limits["margin_min"]:
        return _reject(
            "needs_clarification",
            "source_uncertain",
            f"source={source!r} margin={source_margin:.3f}",
            evidence,
        )
    if source not in (None, "none") and source not in {row["name"] for row in candidates["objects"]}:
        return _reject("rejected", "source_not_in_scene", f"source={source!r}", evidence)

    stage2_questions, relation_hint = _stage2_questions(operation, candidates, source, instruction)
    stage2, stage2_s = _ask(client, state, stage2_questions, timeout_s)
    ans2 = stage2.get("answers") or {}
    evidence["stage2"] = {"answers": ans2, "usage": stage2.get("usage"), "relation_hint": relation_hint}
    evidence["stage2_s"] = round(stage2_s, 4)
    total_s = round(time.perf_counter() - started, 4)

    relation: str | None = relation_hint
    relation_conf = 1.0 if relation_hint else 0.0
    relation_margin = 1.0 if relation_hint else 0.0
    relation_source = "instruction" if relation_hint else "unset"
    target: str | None = None
    target_conf = 0.0
    target_margin = 0.0
    open_target: str | None = None
    open_conf = 0.0
    open_margin = 0.0
    object_margin = 0.0
    if operation == "pick_place":
        target, target_conf, target_margin = _choice_full(ans2.get("target"))
        if "relation" in ans2:
            relation, relation_conf, relation_margin = _choice_full(ans2.get("relation"))
            relation_source = "d1"
        specified = _noul(ans2.get("target_specified"))
        if target in (None, "none"):
            if relation == "in" and _drawer_level_missing(instruction):
                return _reject(
                    "needs_clarification",
                    "drawer_level_not_specified",
                    "the request names a drawer but no level",
                    evidence,
                )
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if specified < limits["specified_min"]:
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if target_conf < limits["target_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} confidence={target_conf:.3f}", evidence)
        if target_margin < limits["margin_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} margin={target_margin:.3f}", evidence)
        if relation_conf < limits["relation_min"]:
            return _reject("needs_clarification", "relation_uncertain", f"relation={relation!r} confidence={relation_conf:.3f}", evidence)
        if relation_margin < limits["margin_min"]:
            return _reject("needs_clarification", "relation_uncertain", f"relation={relation!r} margin={relation_margin:.3f}", evidence)
        kind = _target_kind(target, candidates)
        if kind is None:
            return _reject("rejected", "target_not_in_scene", f"target={target!r}", evidence)
        if kind == "drawer_region" and relation != "in":
            return _reject("rejected", "contradictory_binding", "drawer placement requires relation=in", evidence)
        if kind in {"region", "surface", "site"} and relation != "on":
            return _reject("rejected", "contradictory_binding", f"{kind} placement requires relation=on", evidence)
        if kind == "object" and relation == "in" and not _is_container(target):
            return _reject("rejected", "contradictory_binding", f"{target!r} is not a container", evidence)
    elif operation == "push_object":
        target, target_conf, target_margin = _choice_full(ans2.get("target"))
        specified = _noul(ans2.get("target_specified"))
        relation = "on"
        relation_conf = 1.0
        relation_margin = 1.0
        relation_source = "instruction"
        if target in (None, "none") or specified < limits["specified_min"]:
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if target_conf < limits["target_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} confidence={target_conf:.3f}", evidence)
        if target_margin < limits["margin_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} margin={target_margin:.3f}", evidence)
        kind = _target_kind(target, candidates)
        if kind is None:
            return _reject("rejected", "target_not_in_scene", f"target={target!r}", evidence)
        if kind == "drawer_region":
            return _reject("rejected", "contradictory_binding", "a push target cannot be a drawer", evidence)
    elif operation == "open_drawer":
        target, target_conf, target_margin = _choice_full(ans2.get("target"))
        specified = _noul(ans2.get("target_specified"))
        relation = "open"
        relation_conf = 1.0
        relation_margin = 1.0
        relation_source = "instruction"
        if target in (None, "none"):
            if _drawer_level_missing(instruction):
                return _reject(
                    "needs_clarification",
                    "drawer_level_not_specified",
                    "the request names a drawer but no level",
                    evidence,
                )
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if specified < limits["specified_min"]:
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if target_conf < limits["target_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} confidence={target_conf:.3f}", evidence)
        if target_margin < limits["margin_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} margin={target_margin:.3f}", evidence)
        if target not in {row["name"] for row in candidates["drawers"]}:
            return _reject("rejected", "not_a_drawer", f"target={target!r}", evidence)
    elif operation == "turn_on":
        target, target_conf, target_margin = _choice_full(ans2.get("target"))
        specified = _noul(ans2.get("target_specified"))
        relation = "turn_on"
        relation_conf = 1.0
        relation_margin = 1.0
        relation_source = "instruction"
        if target in (None, "none") or specified < limits["specified_min"]:
            return _reject("missing_information", "target_not_specified", f"target={target!r} specified P(yes)={specified:.3f}", evidence)
        if target_conf < limits["target_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} confidence={target_conf:.3f}", evidence)
        if target_margin < limits["margin_min"]:
            return _reject("needs_clarification", "target_uncertain", f"target={target!r} margin={target_margin:.3f}", evidence)
        if target not in {row["name"] for row in candidates["appliances"]}:
            return _reject("rejected", "not_an_appliance", f"target={target!r}", evidence)
    else:  # composite
        open_target, open_conf, open_margin = _choice_full(ans2.get("open_target"))
        target, target_conf, target_margin = _choice_full(ans2.get("place_target"))
        source, source_conf, object_margin = _choice_full(ans2.get("object"))
        specified = _noul(ans2.get("target_specified"))
        relation = "in"
        relation_conf = 1.0
        relation_margin = 1.0
        relation_source = "instruction"
        drawers = {row["name"] for row in candidates["drawers"]}
        objects = {row["name"] for row in candidates["objects"]}
        if open_target in (None, "none") and _drawer_level_missing(instruction):
            return _reject(
                "needs_clarification",
                "drawer_level_not_specified",
                "the request names a drawer but no level",
                evidence,
            )
        if (
            open_target in (None, "none")
            or target in (None, "none")
            or source in (None, "none")
            or specified < limits["specified_min"]
        ):
            return _reject(
                "missing_information",
                "composite_incomplete",
                f"open_target={open_target!r} object={source!r} place_target={target!r} specified P(yes)={specified:.3f}",
                evidence,
            )
        low = [
            f"{name}={value:.3f}"
            for name, value in (
                ("open_target_conf", open_conf),
                ("object_conf", source_conf),
                ("place_target_conf", target_conf),
                ("open_target_margin", open_margin),
                ("object_margin", object_margin),
                ("place_target_margin", target_margin),
            )
            if value < (limits["margin_min"] if name.endswith("margin") else limits["target_min"])
        ]
        if low:
            return _reject("needs_clarification", "composite_uncertain", ", ".join(low), evidence)
        if open_target not in drawers or target not in drawers or source not in objects:
            return _reject("rejected", "composite_binding_not_in_scene", "a selected composite item is not in the scene", evidence)
        open_owner = _drawer_parts(open_target)[0]
        place_owner = _drawer_parts(target)[0]
        if open_owner != place_owner:
            return _reject(
                "rejected",
                "contradictory_binding",
                f"open target owner {open_owner!r} != place target owner {place_owner!r}",
                evidence,
            )
        if target != open_target:
            return _reject(
                "rejected",
                "contradictory_binding",
                f"place target {target!r} is not the drawer that was opened ({open_target!r})",
                evidence,
            )

    # Goal preservation: the selected items and relation must be the ones the
    # user actually named; a binding that silently changes the goal is rejected.
    drawer_target = operation in {"open_drawer", "composite"} or _target_kind(target, candidates) == "drawer_region"
    if drawer_target and target:
        level = _drawer_parts(target)[1]
        if level and not re.search(rf"\b{level}\b", instruction.lower()):
            # "open the drawer" names no level: that is missing information, not
            # a mismatched binding, so it asks for clarification instead.
            if not re.search(r"\b(top|middle|bottom)\b", instruction.lower()):
                return _reject(
                    "needs_clarification",
                    "drawer_level_not_specified",
                    f"target={target!r} level={level!r} but the request names no drawer level",
                    evidence,
                )
            return _reject(
                "rejected",
                "user_goal_target_not_preserved",
                f"target {target!r} level {level!r} is not named in the request",
                evidence,
            )
    if needs_source and not _covered(instruction, source):
        return _reject("rejected", "user_goal_source_not_preserved", f"source {source!r} is not named in the request", evidence)
    if not _covered(instruction, target, drawer=drawer_target):
        return _reject("rejected", "user_goal_target_not_preserved", f"target {target!r} is not named in the request", evidence)
    if operation == "composite":
        if open_target != target:
            return _reject("rejected", "user_goal_target_not_preserved", "open and place targets differ", evidence)
        if not re.search(r"\b(and|then|,\s*)\b", instruction.lower()):
            return _reject("rejected", "secondary_subgoal_missing", "a composite request must keep both subgoals", evidence)
    if operation in {"pick_place", "composite"} and not _relation_in_instruction(instruction, relation):
        return _reject(
            "rejected",
            "user_goal_relation_not_preserved",
            f"relation {relation!r} is not expressed in the request",
            evidence,
        )
    if operation == "open_drawer" and not _relation_in_instruction(instruction, "open"):
        return _reject("rejected", "user_goal_action_not_preserved", "the request does not ask to open anything", evidence)
    if operation == "turn_on" and not _relation_in_instruction(instruction, "turn_on"):
        return _reject("rejected", "user_goal_action_not_preserved", "the request does not ask to turn something on", evidence)

    interpretation = (
        f"Execute the whole-task policy for the user's request; binding "
        f"operation={operation}, source={source or 'none'}, target={target}, relation={relation}"
        + (f", opened drawer={open_target}" if open_target else "")
    )
    plan = {
        "interpretation": interpretation,
        "success_condition": (
            "The simulation reaches the requested relation between the selected source and target "
            "and the native LIBERO goal is measured true; no success is inferred from language."
        ),
        "steps": [
            {
                "skill": "whole_task",
                "object_n": None,
                "target_n": None,
                "relation": "task",
                "prompt": instruction.strip(),
            }
        ],
    }
    # The same contract the Qwen planner must satisfy; the source stays truthful.
    plan, contract_errors = validate_plan(
        plan,
        instruction=instruction,
        task={"task_id": int(task_id)},
        snapshot=snapshot,
        skills=skills,
        plan_source="d1-3b",
    )
    if contract_errors:
        evidence["contract_errors"] = contract_errors
        return _reject("rejected", "plan_contract_failed", "the assembled plan violates the plan contract", evidence)
    decision = {
        "status": "accepted",
        "reason_code": None,
        "reason": None,
        "operation": operation,
        "operation_confidence": operation_conf,
        "operation_margin": operation_margin,
        "operation_source": operation_source,
        "source": None if source in (None, "none") else source,
        "source_confidence": source_conf,
        "source_margin": source_margin,
        "target": target,
        "target_confidence": target_conf,
        "target_margin": target_margin,
        "relation": relation,
        "relation_confidence": relation_conf,
        "relation_margin": relation_margin,
        "relation_source": relation_source,
        "open_target": open_target,
        "open_confidence": open_conf,
        "open_margin": open_margin,
        "object_margin": object_margin,
        "prompt": instruction.strip(),
        "prompt_source": "user_instruction",
        "plan_source": "d1-3b",
        "evidence": evidence,
        "stage1_s": round(stage1_s, 4),
        "stage2_s": round(stage2_s, 4),
        "total_s": total_s,
        "usage": {
            "input_tokens": (stage1.get("usage") or {}).get("input_tokens", 0)
            + (stage2.get("usage") or {}).get("input_tokens", 0),
            "output_tokens": 0,
        },
    }
    return {"status": "accepted", "reason_code": None, "reason": None, "decision": decision, "plan": plan, "evidence": evidence}


__all__ = ["D1Client", "build_candidates", "plan_with_d1", "DEFAULT_THRESHOLDS"]
