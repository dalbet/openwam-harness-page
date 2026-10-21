"""Load-request contract. The backend owns the task and the seed.

The load button sends ``{"type": "load", "suite": "..."}`` and omits
``task_id``. The previous parser did ``task_id = int(message.get("task_id"))``.
That calls ``int(None)`` and raises ``TypeError: int() argument must be a
string, a bytes-like object or a real number, not 'NoneType'``.

Missing ``task_id`` and empty ``task_text`` select the suite default.
JSON null is a field error. Seed is fixed at 0. ``task_text`` may select a
registered task in the chosen suite. An unimplemented skill is rejected
when the plan is admitted, not by refusing to load the scene.
"""

from __future__ import annotations

import re

from capabilities import DEFAULT_TASK_ID, SUITE_IDS

FIXED_SEED = 0
_SCENE_ID = re.compile(
    r"(libero_spatial|libero_object|libero_goal|libero_10)_t(\d+)(?:_s(\d+))?"
)


class FieldError(ValueError):
    def __init__(self, path: str, code: str, received, expected: str) -> None:
        self.field = {
            "path": path,
            "code": code,
            "received": received,
            "expected": expected,
        }
        super().__init__(f"{path} {code}：收到 {received!r}，期望 {expected}")


def require_int(value, path: str):
    if value is None:
        raise FieldError(path, "null", None, "整数；缺失与 null 不是一回事")
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError(path, "type", value, "整数")
    return value


def resolve_load(message, suites: list[dict]) -> dict:
    """Parse a load message. Does not construct an environment."""
    from suites import goal_relation, task_record

    if isinstance(message, str) or (
        isinstance(message, dict) and message.get("scene") and not message.get("suite")
    ):
        scene_id = message if isinstance(message, str) else str(message.get("scene"))
        matched = _SCENE_ID.fullmatch(scene_id)
        if not matched:
            raise FieldError("scene", "unknown", scene_id, "libero_<suite>_t<task>")
        suite = matched.group(1)
        task_id = int(matched.group(2))
        if matched.group(3) is not None and int(matched.group(3)) != FIXED_SEED:
            raise FieldError("seed", "fixed", int(matched.group(3)), "0")
        seed = FIXED_SEED
        source = "scene_id"
        match = {"status": "scene_id", "matched_task": None, "task_key": None, "source": "scene_id"}
        _reject_init_state(message if isinstance(message, dict) else {}, seed)
    elif not isinstance(message, dict):
        raise FieldError("body", "type", message, "object")
    else:
        suite = _suite(message, suites)
        task_id, source, match = _task_id(message, suite, suites)
        seed = _seed(message)
        _reject_init_state(message, seed)
    task = _lookup(task_record, suites, suite, task_id)
    count = task.get("init_states")
    if count is None:
        raise FieldError("init_states", "null", None, "注册表中的正整数")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise FieldError("init_states", "type", count, "正整数")
    if seed < 0 or seed >= count:
        raise FieldError("seed", "range", seed, f"0 <= seed < {count}")
    return {
        "scene_id": f"{suite}_t{task_id}_s{seed}",
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "init_state_id": seed,
        "task_source": source,
        "matched_task": match.get("matched_task"),
        "match_source": match.get("source") or source,
        "goal": task["language"],
        "task": task["language"],
        "summary_zh": task["summary_zh"],
        "task_key": task.get("task_key"),
        "instances": task.get("instances") or [],
        "fixtures": task.get("fixtures") or [],
        "place_regions": task.get("place_regions") or [],
        "scene_regions": task.get("scene_regions") or {"regions": [], "version": ""},
        "goal_relation": goal_relation(task["language"]),
        "long_candidate": task["long_candidate"],
        "init_states": count,
        "execution": "同步",
    }


def _suite(message: dict, suites: list[dict]) -> str:
    if "suite" not in message or message.get("suite") in (None, ""):
        raise FieldError("suite", "missing", message.get("suite"), "、".join(SUITE_IDS))
    suite = message.get("suite")
    known = {item["id"] for item in suites} or set(SUITE_IDS)
    if not isinstance(suite, str) or suite not in known:
        raise FieldError("suite", "unknown", suite, "、".join(sorted(known)))
    return suite


def _task_id(message: dict, suite: str, suites: list[dict]) -> tuple[int, str, dict]:
    text = message.get("task_text")
    if isinstance(text, str):
        from task_catalog import match_task_text

        tasks = []
        for item in suites:
            if item.get("id") == suite:
                tasks = list(item.get("tasks") or [])
                break
        match = match_task_text(suite, text, tasks)
        if match["status"] == "ambiguous":
            raise FieldError("task_text", "ambiguous", text, match.get("detail") or "唯一任务")
        if match["status"] == "unknown":
            raise FieldError("task_text", "unknown", text, match.get("detail") or "该套件的标准任务名")
        if match["status"] == "matched":
            if "task_id" in message and message.get("task_id") is not None:
                explicit = require_int(message.get("task_id"), "task_id")
                if explicit != match["task_id"]:
                    raise FieldError(
                        "task_text",
                        "conflict",
                        text,
                        f"task_id {explicit} 与文本匹配的 task {match['task_id']} 不同",
                    )
            return int(match["task_id"]), str(match["source"]), match
        if "task_id" not in message:
            return int(match["task_id"]), "default", match
    if "task_id" not in message:
        task_id = DEFAULT_TASK_ID[suite]
        return task_id, "default", {"status": "empty", "matched_task": None, "task_key": None, "source": "default"}
    value = message.get("task_id")
    if value is None:
        raise FieldError("task_id", "null", None, "省略该字段以使用套件默认任务，或发送整数")
    task_id = require_int(value, "task_id")
    if task_id < 0:
        raise FieldError("task_id", "range", task_id, ">= 0")
    return task_id, "request", {"status": "request", "matched_task": None, "task_key": None, "source": "request"}


def _seed(message: dict) -> int:
    if "seed" not in message:
        return FIXED_SEED
    value = message.get("seed")
    if value is None:
        raise FieldError("seed", "null", None, "省略该字段；seed 固定为 0")
    seed = require_int(value, "seed")
    if seed != FIXED_SEED:
        raise FieldError("seed", "fixed", seed, "0")
    return FIXED_SEED


def _reject_init_state(message: dict, seed: int) -> None:
    if "init_state_id" not in message:
        return
    value = message.get("init_state_id")
    if value is None:
        raise FieldError("init_state_id", "null", None, "省略该字段；初始状态由 seed 0 决定")
    init_id = require_int(value, "init_state_id")
    if init_id != seed:
        raise FieldError("init_state_id", "mismatch", init_id, str(seed))


def _lookup(task_record, suites: list[dict], suite: str, task_id: int) -> dict:
    try:
        return task_record(suites, suite, task_id)
    except KeyError as exc:
        raise FieldError("task_id", "unknown", task_id, f"{suite} 中的已注册任务") from exc


