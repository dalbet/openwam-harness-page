"""Standard LIBERO suites from the installed benchmark registry.

Spatial, Object, Goal, and Long are the four standard suites. Perturbation
suites (*_task, *_swap, *_lan) and libero_90 are not presented as those four.
Object capabilities come from each task BDDL :objects and :fixtures lists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from task_zh import SUITE_LABELS, TASK_ZH

ROOT = Path(__file__).resolve().parents[1]
RPENT = ROOT / "RPent"

STANDARD = (
    ("libero_spatial", SUITE_LABELS["libero_spatial"]),
    ("libero_object", SUITE_LABELS["libero_object"]),
    ("libero_goal", SUITE_LABELS["libero_goal"]),
    ("libero_10", SUITE_LABELS["libero_10"]),
)

# Relations a type can receive. Membership in :objects, not this table, sets movable.
TARGET_RELATIONS = {
    "plate": ["on"],
    "basket": ["in"],
    "akita_black_bowl": ["in", "on"],
    "glazed_rim_porcelain_ramekin": ["on", "in"],
    "cookies": ["on"],
    "flat_stove": ["on"],
    "wooden_cabinet": ["on", "inside_drawer"],
    "white_cabinet": ["on", "inside_drawer"],
    "microwave": ["in"],
    "wine_rack": ["on"],
    "desk_caddy": ["in"],
}
ALIASES = {
    "akita_black_bowl": ["black bowl", "akita black bowl"],
    "glazed_rim_porcelain_ramekin": ["ramekin"],
    "cookies": ["cookie box", "cookies"],
    "alphabet_soup": ["alphabet soup"],
    "cream_cheese": ["cream cheese", "cream cheese box"],
    "salad_dressing": ["salad dressing"],
    "bbq_sauce": ["bbq sauce"],
    "tomato_sauce": ["tomato sauce"],
    "orange_juice": ["orange juice"],
    "chocolate_pudding": ["chocolate pudding"],
    "wooden_cabinet": ["wooden cabinet", "cabinet"],
    "white_cabinet": ["cabinet"],
    "flat_stove": ["stove"],
    "wine_bottle": ["wine bottle"],
    "wine_rack": ["wine rack", "rack"],
    "moka_pot": ["moka pot", "moka pots"],
    "porcelain_mug": ["white mug", "porcelain mug"],
    "white_yellow_mug": ["yellow and white mug"],
    "red_coffee_mug": ["red coffee mug"],
    "black_book": ["book"],
    "desk_caddy": ["caddy"],
    "microwave": ["microwave"],
    "plate": ["plate"],
    "basket": ["basket"],
    "butter": ["butter"],
    "milk": ["milk"],
    "ketchup": ["ketchup"],
}

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
    """Display translation. Does not replace the English execution string."""
    return TASK_ZH.get(language.strip(), language)


def goal_relation(language: str) -> str:
    from schema import transfer_slots

    slots = transfer_slots(language)
    if slots:
        return slots[0]["relation"]
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
            catalog = _catalog(suite.get_task_bddl_file_path(index))
            tasks.append(
                {
                    "id": index,
                    "task_key": f"{suite_id}:t{index}",
                    "language": language,
                    "summary_zh": summarize_zh(language),
                    "init_states": len(states),
                    "long_candidate": suite_id == "libero_10" and index in {0, 1},
                    "instances": catalog["instances"],
                    "fixtures": catalog["fixtures"],
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


def single_manual_plan(
    language: str,
    object_names: list[str],
    *,
    suite: str,
    task_id: int,
    seed: int,
    task_record: dict | None = None,
    poses: dict | None = None,
    observation_id: str | None = None,
    scene_epoch: int | None = None,
    binding: dict | None = None,
) -> dict:
    """One explicit manual pick-and-place. Not a model plan."""
    from binding import bind_task
    from schema import prepare_plan

    objects = objects_for_names(object_names, task_record)
    relation = goal_relation(language)
    if not binding or binding.get("status") != "resolved":
        binding = bind_task(
            language,
            objects,
            poses,
            observation_id=observation_id,
            scene_epoch=scene_epoch,
        )
    if binding["status"] != "resolved":
        raise ValueError(f"对象未唯一绑定，不默认第一个，也不开始动作: {binding}")
    from schema import transfer_slots

    chosen = binding["object_ref"]
    slots = transfer_slots(language) or []
    if len(slots) != 1:
        raise ValueError(f"人工单次计划只接受一个搬运: {language}")
    slot = slots[0]
    targets = [
        ref
        for ref, spec in objects.items()
        if _phrase_match(spec, slot["target"]) and relation in (spec.get("target_relations") or [])
    ]
    if len(targets) != 1:
        raise ValueError(f"目标没有唯一实例，不默认第一个: {targets}")
    target = targets[0]
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
    prepared = prepare_plan(
        plan,
        task=language,
        scene_objects={key: objects[key] for key in (chosen, target)},
        required_object_ref=chosen,
    )
    prepared["binding"] = binding
    return prepared


def objects_for_names(names: list[str], task: dict | None) -> dict:
    """Instance ids stay intact. A bowl in :objects is movable and can still be a target."""
    catalog = {}
    if task:
        for item in list(task.get("instances") or []) + list(task.get("fixtures") or []):
            catalog[item["name"]] = item
    objects = {}
    for name in names:
        spec = catalog.get(name)
        if spec is None:
            objects[name] = {
                "sim_name": name,
                "phrases": [name.replace("_", " ")],
                "semantic_type": "unlisted",
                "movable": False,
                "can_be_target": False,
                "target_relations": [],
                "capability_source": "unlisted",
            }
            continue
        objects[name] = dict(spec)
        objects[name]["sim_name"] = name
    return objects


def _objects_from_names(names: list[str]) -> dict:
    """Legacy helper. Unknown names are not marked movable."""
    return objects_for_names(names, None)


def _catalog(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return {
        "instances": _section(text, "objects", movable=True),
        "fixtures": _section(text, "fixtures", movable=False),
    }


def _section(text: str, name: str, *, movable: bool) -> list[dict]:
    match = re.search(rf"\(:{name}\b(.*?)\)\s*\n", text, re.S)
    rows = []
    if not match:
        return rows
    for line in match.group(1).splitlines():
        if " - " not in line:
            continue
        raw_names, semantic = line.split(" - ", 1)
        semantic = semantic.strip()
        relations = list(TARGET_RELATIONS.get(semantic, []))
        aliases = list(ALIASES.get(semantic) or [semantic.replace("_", " ")])
        for instance in raw_names.split():
            rows.append(
                {
                    "name": instance,
                    "semantic_type": semantic,
                    "phrases": aliases,
                    "grounding_query": min(aliases, key=len),
                    "movable": movable,
                    "can_be_target": bool(relations),
                    "target_relations": relations,
                    "capability_source": "bddl:objects" if movable else "bddl:fixtures",
                }
            )
    return rows


def _phrase_match(spec: dict, needle: str) -> bool:
    needle = needle.strip().lower()
    phrases = [str(item).lower() for item in spec.get("phrases") or []]
    semantic = str(spec.get("semantic_type") or "").replace("_", " ")
    return any(phrase and (phrase in needle or needle in phrase) for phrase in phrases) or bool(semantic and (semantic in needle or needle in semantic))


def _benchmark():
    if str(RPENT) not in sys.path:
        sys.path.insert(0, str(RPENT))
    from rlinf.envs.sim.libero.utils import benchmark as bench

    return bench
