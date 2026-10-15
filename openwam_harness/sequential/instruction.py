"""Match a free Chinese or English instruction to one registered task.

An ambiguous phrase asks one question. It does not select the first object
or the first task. A different task must be loaded explicitly; its native
success check replaces the previous one.
"""

from __future__ import annotations

import re

from capabilities import GROUPS, classify
from task_zh import TASK_ZH


def _norm(text: str) -> str:
    return re.sub(r"[\s。！？，、,.!?:：；;\"'“”]+", "", text.strip().lower())


_PHRASES = (
    "奶油奶酪盒", "巧克力布丁", "番茄酱瓶", "字母汤", "番茄酱", "奶油奶酪", "沙拉酱", "烧烤酱",
    "橙汁", "牛奶", "黄油", "篮子", "盘子", "黑碗", "小碗", "抽屉", "柜子", "炉灶", "微波炉",
    "酒瓶", "摩卡壶", "饼干盒", "木柜", "桌面中央", "中间抽屉", "上层抽屉", "下层抽屉", "打开炉灶",
    "book", "soup", "basket", "plate", "bowl", "stove", "drawer",
)


def _tokens(text: str) -> list[str]:
    compact = _norm(text)
    found = [phrase for phrase in _PHRASES if phrase in compact]
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    stop = {"pick", "place", "the", "and", "put", "both", "from", "into"}
    return found + [word for word in words if word not in stop]


def catalog_from_builtin() -> list[dict]:
    rows = []
    for suite, languages in GROUPS:
        for index, language in enumerate(languages):
            rows.append(
                {
                    "suite": suite,
                    "id": index,
                    "language": language,
                    "summary_zh": TASK_ZH[language],
                    "task_key": f"{suite}:t{index}",
                }
            )
    return rows


def resolve_instruction(text: str, loaded: dict, tasks: list[dict] | None = None) -> dict:
    """Decide whether the typed text matches the loaded scene, another task, or neither."""
    tasks = tasks if tasks is not None else catalog_from_builtin()
    suite = loaded.get("suite")
    if "task_id" not in loaded or loaded.get("task_id") is None:
        return {
            "status": "unsupported",
            "changed": False,
            "user_text": text.strip(),
            "language": None,
            "reason": "已加载场景缺少 task_id，不会猜测或切换任务。",
        }
    raw_id = loaded.get("task_id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        return {
            "status": "unsupported",
            "changed": False,
            "user_text": text.strip(),
            "language": None,
            "reason": f"已加载场景的 task_id 不是整数（收到 {raw_id!r}），不会猜测或切换任务。",
        }
    loaded_id = raw_id
    canonical = str(loaded.get("goal") or loaded.get("language") or "").strip()
    summary = str(loaded.get("summary_zh") or TASK_ZH.get(canonical, "")).strip()
    typed = text.strip()
    if not typed:
        return _matched(loaded, canonical, summary, changed=False, user_text="")
    pool = [task for task in tasks if task.get("suite") == suite]
    exact = [task for task in pool if _norm(typed) in {_norm(task["language"]), _norm(task["summary_zh"])}]
    chosen = exact
    if not chosen:
        tokens = _tokens(typed)
        if tokens:
            chosen = []
            for task in pool:
                hay = _norm(task["language"] + task["summary_zh"])
                if all(token in hay for token in tokens):
                    chosen.append(task)
    if len(chosen) == 1:
        task = chosen[0]
        changed = int(task["id"]) != loaded_id
        return _matched(task, task["language"], task["summary_zh"], changed=changed, user_text=typed)
    if len(chosen) > 1:
        return {
            "status": "ambiguous",
            "changed": False,
            "user_text": typed,
            "language": None,
            "question": _question(chosen),
            "candidates": [{"task_id": task["id"], "summary_zh": task["summary_zh"], "language": task["language"]} for task in chosen],
        }
    elsewhere = [
        task for task in tasks
        if task.get("suite") != suite and _norm(typed) in {_norm(task["language"]), _norm(task["summary_zh"])}
    ]
    if elsewhere:
        other = elsewhere[0]["suite"]
        return {
            "status": "unsupported",
            "changed": False,
            "user_text": typed,
            "language": None,
            "reason": f"当前场景不支持这条指令。它属于另一个套件 {other}，不会沿用现在的成功标准。",
        }
    return {
        "status": "unsupported",
        "changed": False,
        "user_text": typed,
        "language": None,
        "reason": "当前场景不支持这条指令，不会沿用已加载任务的成功标准。",
    }


def _task_id(task: dict) -> int:
    if "id" in task:
        value = task["id"]
    elif "task_id" in task:
        value = task["task_id"]
    else:
        value = None
    if isinstance(value, bool) or not isinstance(value, int):
        from scene_request import FieldError

        raise FieldError("task_id", "null" if value is None else "type", value, "已注册任务的整数 id")
    return value


def _matched(task: dict, language: str, summary: str, *, changed: bool, user_text: str) -> dict:
    row = classify(language, task.get("suite") or "")
    return {
        "status": "matched",
        "changed": changed,
        "user_text": user_text,
        "language": language,
        "summary_zh": summary,
        "task_id": _task_id(task),
        "suite": task.get("suite"),
        "task_key": task.get("task_key"),
        "capability": row,
    }


def _question(tasks: list[dict]) -> str:
    shown = tasks[:4]
    options = "；".join(task["summary_zh"] for task in shown)
    if len(tasks) > len(shown):
        options += f"；以及其他 {len(tasks) - len(shown)} 条"
    return f"哪一条才是要执行的任务？{options}。不会默认第一条。"
