"""Standard LIBERO suites from the installed benchmark registry.

Spatial, Object, Goal, and Long are the four standard suites. Perturbation
suites (*_task, *_swap, *_lan) and libero_90 are not presented as those four.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RPENT = ROOT / "RPent"

STANDARD = (
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
)

_PHRASES = (
    ("alphabet soup", "字母汤"),
    ("tomato sauce", "番茄酱"),
    ("cream cheese box", "奶油奶酪盒"),
    ("cream cheese", "奶油奶酪"),
    ("chocolate pudding", "巧克力布丁"),
    ("orange juice", "橙汁"),
    ("salad dressing", "沙拉酱"),
    ("bbq sauce", "烧烤酱"),
    ("ketchup", "番茄酱瓶"),
    ("milk", "牛奶"),
    ("butter", "黄油"),
    ("basket", "篮子"),
    ("plate", "盘子"),
    ("bowl", "碗"),
    ("mug", "杯子"),
    ("drawer", "抽屉"),
    ("cabinet", "柜子"),
    ("stove", "炉灶"),
    ("microwave", "微波炉"),
    ("put both the", "把"),
    ("put the", "把"),
    ("pick up the", "拿起"),
    ("place it in the basket", "放进篮子"),
    ("place it on the plate", "放到盘子上"),
    ("place it in the", "放进"),
    ("place it on the", "放到"),
    ("in the basket", "放进篮子"),
    ("on the plate", "放到盘子上"),
    ("on the stove", "放到炉灶上"),
    ("in the drawer", "放进抽屉"),
    ("in the cabinet", "放进柜子"),
    ("in the bowl", "放进碗里"),
    (" and the ", "和"),
    (" and ", "和"),
)


def summarize_zh(language: str) -> str:
    """Glossary gloss of a registry instruction. The English line stays authoritative."""
    text = " " + language.strip().lower() + " "
    for src, dst in sorted(_PHRASES, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    gloss = " ".join(text.split())
    return gloss or language


def goal_relation(language: str) -> str:
    lowered = language.lower()
    if "drawer" in lowered:
        return "inside_drawer"
    if " on " in f" {lowered} ":
        return "on"
    return "in"


def load_suites() -> list[dict]:
    bench = _benchmark()
    suites = []
    for suite_id, label in STANDARD:
        suite = bench.get_benchmark(suite_id)()
        count = int(suite.get_num_tasks())
        tasks = []
        for index in range(count):
            language = str(suite.get_task(index).language).strip()
            states = suite.get_task_init_states(index)
            tasks.append(
                {
                    "id": index,
                    "language": language,
                    "summary_zh": summarize_zh(language),
                    "init_states": len(states),
                    "long_candidate": suite_id == "libero_10" and index in {0, 1},
                }
            )
        suites.append({"id": suite_id, "label": label, "standard": True, "tasks": tasks})
    return suites


def task_record(suites: list[dict], suite_id: str, task_id: int) -> dict:
    for suite in suites:
        if suite["id"] != suite_id:
            continue
        for task in suite["tasks"]:
            if int(task["id"]) == int(task_id):
                if task_id < 0 or task_id >= len(suite["tasks"]):
                    break
                return task
        raise KeyError(f"task {task_id} is not in {suite_id}")
    raise KeyError(f"suite {suite_id} is not a standard suite")


def single_manual_plan(language: str, object_names: list[str], *, suite: str, task_id: int, seed: int) -> dict:
    """One explicit manual pick-and-place. Not a model plan."""
    from schema import prepare_plan

    objects = _objects_from_names(object_names)
    relation = goal_relation(language)
    lowered = language.lower()
    movables = [ref for ref, spec in objects.items() if spec["role"] == "movable"]
    targets = [ref for ref, spec in objects.items() if spec["role"] in {"container", "support"}]
    if not movables or not targets:
        raise ValueError(f"场景对象不足以组成一次抓放: {object_names}")

    def rank(ref: str) -> int:
        phrase = objects[ref]["phrases"][0]
        found = lowered.find(phrase)
        return found if found >= 0 else 10_000

    chosen = sorted(movables, key=rank)[0]
    target = sorted(targets, key=rank)[0]
    if rank(chosen) == 10_000:
        raise ValueError(f"指令里没有可对应的食品对象: {language}")
    plan = {
        "plan_id": f"manual-single-{suite}-t{task_id}-s{seed}",
        "goal": language,
        "task": language,
        "goal_relation": relation,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "plan_source": "manual",
        "steps": [
            {
                "step_id": "once",
                "skill": "transfer",
                "object_ref": chosen,
                "target_ref": target,
                "relation": relation,
                "prompt": f"pick up the {objects[chosen]['phrases'][0]}",
                "depends_on": [],
                "preconditions": [],
                "completion_check": "object_placed",
                "budget": {"max_s": 180, "max_attempts": 2},
                "on_failure": "recover",
            }
        ],
    }
    return prepare_plan(plan, task=language, scene_objects={key: objects[key] for key in (chosen, target)})


def _objects_from_names(names: list[str]) -> dict:
    objects = {}
    for name in names:
        stem = name[:-2] if name.endswith("_1") else name
        phrase = stem.replace("_", " ")
        if any(word in phrase for word in ("basket", "plate", "bowl", "bin", "drawer", "cabinet")):
            role = "container"
        elif any(word in phrase for word in ("stove", "microwave", "table")):
            role = "support"
        else:
            role = "movable"
        objects[stem] = {"sim_name": name, "phrases": [phrase], "role": role}
    return objects


def _benchmark():
    if str(RPENT) not in sys.path:
        sys.path.insert(0, str(RPENT))
    from rlinf.envs.sim.libero.utils import benchmark as bench

    return bench
