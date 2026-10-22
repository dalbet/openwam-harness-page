"""Standard LIBERO tasks for the loaded scene.

The catalog is read from the installed benchmark and BDDL. It is context for
the planner, not a whitelist that rejects Chinese before the request.
A row existing here is not a successful execution.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from capabilities import classify

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "libero_task_catalog.json"

PARAPHRASES = {
    "put the bowl on the stove": ["碗放炉灶上", "把黑碗放到灶台上"],
    "put the bowl on the plate": ["把黑碗放到盘子里", "将碗搁到盘子上"],
    "put the cream cheese in the bowl": ["把奶油奶酪放进碗中", "把那盒奶酪放进黑碗里"],
    "put the wine bottle on the rack": ["把红酒瓶放到酒架上", "将酒瓶搁上酒架"],
    "pick up the alphabet soup and place it in the basket": ["把字母汤放进篮子", "将字母汤罐头放进篮子中"],
    "pick up the tomato sauce and place it in the basket": ["把番茄酱放进篮子", "将番茄酱罐头放进篮子中"],
    "pick up the black bowl between the plate and the ramekin and place it on the plate": [
        "把盘子和小碗中间的黑碗放到盘子上",
        "拿起盘子与小碗之间的那只黑碗，放到盘子上",
    ],
}


def load_task_catalog() -> list[dict[str, Any]]:
    """Return the checked catalog. Rebuild only when the fixture is absent."""
    if FIXTURE.is_file():
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rows = payload.get("tasks") if isinstance(payload, dict) else payload
        if isinstance(rows, list) and rows:
            return rows
    return build_task_catalog()


def build_task_catalog() -> list[dict[str, Any]]:
    """Read the installed registry. This does not reset or step an environment."""
    from libero.libero.benchmark import get_benchmark

    rows = []
    for suite_id in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = get_benchmark(suite_id)()
        for task_id in range(int(suite.get_num_tasks())):
            task = suite.get_task(task_id)
            language = str(task.language).strip()
            path = suite.get_task_bddl_file_path(task_id)
            text = Path(path).read_text(encoding="utf-8")
            rows.append(_row(suite_id, task_id, language, text))
    return rows


def write_task_catalog(path: Path | None = None) -> Path:
    target = path or FIXTURE
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_task_catalog()
    target.write_text(
        json.dumps({"source": "installed libero benchmark and bddl", "tasks": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def tasks_for_scene(suite: str, names: set[str] | list[str]) -> list[dict[str, Any]]:
    """Standard tasks whose objects are already in this scene. Does not switch scenes."""
    present = set(names)
    found = []
    for row in load_task_catalog():
        if row.get("suite") != suite:
            continue
        needed = [item for item in (row.get("source_object"), row.get("target_object")) if item]
        if needed and all(item in present for item in needed):
            found.append(row)
            continue
        region = row.get("target_region")
        if (
            not needed
            and isinstance(region, str)
            and region.endswith("_middle_region")
            and any(isinstance(name, str) and name.startswith("wooden_cabinet_") for name in present)
        ):
            found.append(row)
    return found


GOAL_MANUAL = {
    1: {
        "code_status": "已实现",
        "acceptance_label": "人工曾成功，不稳定；本轮修复后尚未再验收",
        "loadable": True,
        "manual_steps": "加载 task 1 后输入“把碗放到炉灶上。”点开始。核对已加载任务是 1，失败时看步骤名和 sim_action_count，不要只看“使用旧观测”。",
    },
    6: {
        "code_status": "已实现",
        "acceptance_label": "人工测试失败于旧观测；尚未验收",
        "loadable": True,
        "manual_steps": "先加载 task 6，确认画面和目标句是奶酪与碗，再输入“把奶油奶酪放进碗里。”。核对 target 是 akita_black_bowl_1，官方目标是 On，不是碗内部半径。",
    },
    8: {
        "code_status": "已实现",
        "acceptance_label": "人工测试失败于旧观测；尚未验收",
        "loadable": True,
        "manual_steps": "先加载 task 8，确认目标是盘子，再输入“把碗放到盘子上。”。核对 xy 限制是 0.03，落点是盘面而不是盘心加一个固定高度。",
    },
    2: {
        "code_status": "已实现，待人工验收",
        "acceptance_label": "尚未验收",
        "loadable": True,
        "manual_steps": "加载 task 2。输入“把酒瓶放到柜子顶上。”。核对区域是 wooden_cabinet_1_top_side，不是抽屉。",
    },
    4: {
        "code_status": "已实现，待人工验收",
        "acceptance_label": "尚未验收",
        "loadable": True,
        "manual_steps": "加载 task 4。输入“把碗放到柜子顶上。”。核对区域是 wooden_cabinet_1_top_side，碗和酒瓶的下降高度不同。",
    },
}


def goal_manual(task_id: int | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Labels for the page. A CPU test is not an acceptance record."""
    if task_id is None:
        return [{"task_id": key, **value} for key, value in GOAL_MANUAL.items()]
    row = GOAL_MANUAL.get(task_id)
    if row is None:
        return {
            "code_status": "未实现",
            "acceptance_label": "尚未验收",
            "loadable": False,
            "manual_steps": "这一项还没有可执行技能，选择后不会加载，开始也不会改去 task 1。",
        }
    return dict(row)


def scene_examples(suite: str, names: set[str] | list[str]) -> list[dict[str, str]]:
    """Chinese fill-ins for tasks this scene can name. Clicking does not execute."""
    examples = []
    for row in tasks_for_scene(suite, names):
        if row.get("executor_support") in {"unsupported_skill", "unresolved_region"}:
            continue
        if row.get("instruction_en") not in PARAPHRASES and row.get("task_id") != _loaded_default(suite):
            continue
        examples.append(
            {
                "input_zh": row["instruction_zh"],
                "instruction_zh": row["instruction_zh"],
                "instruction_en": row["instruction_en"],
                "task_key": row["task_key"],
                "executor_support": row["executor_support"],
            }
        )
    return examples


def _loaded_default(suite: str) -> int | None:
    from capabilities import DEFAULT_TASK_ID

    return DEFAULT_TASK_ID.get(suite)


def planner_task_context(suite: str, names: set[str] | list[str]) -> list[dict[str, Any]]:
    """Short rows for the planner. Candidate numbers stay in the scene graph."""
    context = []
    for row in tasks_for_scene(suite, names):
        context.append(
            {
                "task_key": row["task_key"],
                "instruction_en": row["instruction_en"],
                "instruction_zh": row["instruction_zh"],
                "source_object": row["source_object"],
                "target_object": row["target_object"],
                "target_region": row["target_region"],
                "goal_relation": row["goal_relation"],
                "executor_support": row["executor_support"],
                "support_reason": row.get("support_reason") or "",
            }
        )
    return context


_TRAILING = re.compile(r"[\s。.!！?？,，;；:：]+$")


def normalize_phrase(text: str) -> str:
    """Case, inner whitespace, and trailing punctuation only. on/in and the objects stay distinct."""
    value = str(text or "").strip().lower().replace("\u3000", " ")
    value = re.sub(r"\s+", " ", value)
    return _TRAILING.sub("", value)


def _row_phrases(row: dict[str, Any]) -> list[tuple[str, str]]:
    phrases = [
        (row.get("instruction_en"), "instruction_en"),
        (row.get("instruction_zh"), "instruction_zh"),
    ]
    for item in row.get("paraphrases_zh") or []:
        phrases.append((item, "paraphrase_zh"))
    return [(phrase, source) for phrase, source in phrases if isinstance(phrase, str) and phrase.strip()]


def lookup_task(task_key: str | None, instruction_en: str | None, suite: str, names: set[str] | list[str]) -> dict[str, Any] | None:
    """Metadata for a standard sentence. A missing row is not an execution denial."""
    rows = tasks_for_scene(suite, names)
    wanted = instruction_en.strip() if isinstance(instruction_en, str) else ""
    if wanted:
        norm = normalize_phrase(wanted)
        for row in rows:
            if any(normalize_phrase(phrase) == norm for phrase, _source in _row_phrases(row)):
                return row
        return None
    if isinstance(task_key, str) and task_key:
        for row in rows:
            if row["task_key"] == task_key:
                return row
    return None


def goal_alignment(typed: str, spec: dict[str, Any]) -> str:
    """native when the request is the loaded task sentence. Otherwise the request is judged on its own."""
    text = typed.strip() if isinstance(typed, str) else ""
    if not text:
        return "native"
    phrases = [spec.get("summary_zh"), spec.get("goal"), spec.get("task")]
    if any(isinstance(item, str) and item.strip() and normalize_phrase(item) == normalize_phrase(text) for item in phrases):
        return "native"
    return "requested"


def match_task_text(suite: str, text: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve one loaded task inside the selected suite. Load routing, not a start-time whitelist."""
    from capabilities import DEFAULT_TASK_ID

    default_id = DEFAULT_TASK_ID[suite]
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "empty",
            "task_id": default_id,
            "source": "default",
            "matched_task": None,
            "task_key": None,
        }
    wanted = normalize_phrase(text)
    found: dict[int, tuple[dict[str, Any], str]] = {}
    for task in tasks:
        phrases = [
            (task.get("summary_zh") or task.get("instruction_zh"), "instruction_zh"),
            (task.get("language") or task.get("instruction_en"), "instruction_en"),
        ]
        language = task.get("language") or task.get("instruction_en") or ""
        for item in PARAPHRASES.get(language) or []:
            phrases.append((item, "paraphrase_zh"))
        for extra in task.get("paraphrases_zh") or []:
            phrases.append((extra, "paraphrase_zh"))
        for phrase, source in phrases:
            if isinstance(phrase, str) and phrase.strip() and normalize_phrase(phrase) == wanted:
                found.setdefault(int(task["id"] if "id" in task else task["task_id"]), (task, source))
                break
    if len(found) == 1:
        task_id, (task, source) = next(iter(found.items()))
        return {
            "status": "matched",
            "task_id": task_id,
            "source": source,
            "matched_task": task.get("summary_zh") or task.get("instruction_zh") or task.get("language") or task.get("instruction_en"),
            "task_key": task.get("task_key"),
        }
    if len(found) > 1:
        listed = "、".join(
            f"task {task_id} {found[task_id][0].get('summary_zh') or found[task_id][0].get('language') or ''}"
            for task_id in sorted(found)
        )
        return {
            "status": "ambiguous",
            "task_id": None,
            "source": None,
            "matched_task": None,
            "task_key": None,
            "detail": f"匹配到多个任务，不会默认第一条：{listed}",
        }
    return {
        "status": "unknown",
        "task_id": None,
        "source": None,
        "matched_task": None,
        "task_key": None,
        "detail": "不是该套件已核对的中文或英文标准任务名",
    }


def single_task_block(plan: dict[str, Any], row: dict[str, Any] | None) -> str | None:
    """Refuse combo plans and a matched standard task whose skill or region is not executable."""
    count = plan.get("semantic_step_count")
    if not isinstance(count, int):
        count = len([step for step in (plan.get("steps") or []) if isinstance(step, dict) and step.get("skill") == "transfer"])
    if count > 1:
        return "当前仅执行单任务，不会按组合目标继续。"
    if row is None:
        return None
    if row.get("executor_support") == "unsupported_skill":
        return row.get("support_reason") or "执行器缺少该任务所需技能。"
    if row.get("executor_support") == "unresolved_region":
        return row.get("support_reason") or "当前场景没有该任务的放置区域。"
    return None


def _row(suite: str, task_id: int, language: str, text: str) -> dict[str, Any]:
    from regions import parse_bddl_regions

    capability = classify(language, suite)
    goals = _goal_atoms(text)
    regions = parse_bddl_regions(text)
    region_by_key = {item["query_key"]: item for item in regions}
    source, target_object, target_region, relation = _slots(goals, region_by_key)
    support, reason = _support(capability, goals, target_region, region_by_key, language)
    return {
        "suite": suite,
        "task_id": task_id,
        "task_key": f"{suite}:t{task_id}",
        "instruction_en": language,
        "instruction_zh": capability["summary_zh"],
        "source_object": source,
        "target_object": target_object,
        "target_region": target_region,
        "goal_relation": relation,
        "required_skills": list(capability["required_skills"]),
        "executor_support": support,
        "support_reason": reason,
        "acceptance": "not_run",
        "paraphrases_zh": list(PARAPHRASES.get(language) or []),
    }


def _support(capability: dict, goals: list[tuple[str, str, str]], target_region: str | None, regions: dict, language: str) -> tuple[str, str]:
    if len(goals) > 1:
        return "unsupported_skill", "当前仅执行单任务，这条官方指令包含多个独立目标。"
    missing = list(capability.get("missing_skills") or [])
    if capability.get("required_skills") == ["open_drawer"] and not missing:
        return "implemented", "中层抽屉由 Open 谓词和 WoodenCabinet.is_open 判断。acceptance=not_run，不能写成已经拉开。"
    if missing or capability.get("kind") in {"contact", "unsupported"}:
        detail = "、".join(missing) or "、".join(capability.get("limitations") or [])
        return "unsupported_skill", f"缺少技能：{detail or capability.get('kind')}"
    if target_region:
        region = regions.get(target_region)
        if region is None:
            return "unresolved_region", f"BDDL 没有放置区域 {target_region}。"
        if _region_rotated(target_region):
            return "unsupported_skill", f"{target_region} 带旋转，当前 move_to 只能平移，不能按该朝向放置。"
    return "implemented", "执行器已实现抓取、搬运、下降和释放。acceptance=not_run，不能把任务集存在写成已执行成功。"


def _region_rotated(region_key: str) -> bool:
    """Wine-rack top sites are authored with a non-identity quaternion."""
    if not region_key.endswith("wine_rack_1_top_region") and "wine_rack" not in region_key:
        return False
    root = Path(__file__).resolve().parents[1] / "envs" / "rpent" / "lib" / "python3.10" / "site-packages" / "libero" / "libero" / "assets"
    for path in root.glob("**/wine_rack.xml"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'name="top_region"[^>]*quat="([^"]+)"', text) or re.search(r'quat="([^"]+)"[^>]*name="top_region"', text)
        if not match:
            continue
        values = [float(item) for item in match.group(1).split()]
        return not _identity_quat(values)
    return True


def _identity_quat(values: list[float]) -> bool:
    if len(values) != 4:
        return False
    return abs(abs(values[0]) - 1.0) < 1e-3 and all(abs(item) < 1e-3 for item in values[1:])


def _goal_atoms(text: str) -> list[tuple[str, str, str]]:
    match = re.search(r"\(:goal\b(.*)\)\s*$", text, re.S)
    body = match.group(1) if match else ""
    atoms = [(rel, obj, dst) for rel, obj, dst in re.findall(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", body)]
    if atoms:
        return atoms
    opened = re.findall(r"\(Open\s+([A-Za-z0-9_]+)\)", body)
    if len(opened) == 1:
        return [("Open", opened[0], opened[0])]
    return []


def _slots(goals: list[tuple[str, str, str]], regions: dict[str, dict]) -> tuple[str | None, str | None, str | None, str | None]:
    if not goals:
        return None, None, None, None
    rel, obj, dst = goals[0]
    if rel == "Open":
        region = regions.get(dst)
        owner = region.get("owner_instance") if isinstance(region, dict) else None
        return None, owner, dst, "open"
    relation = "in" if rel == "In" else "on"
    region = regions.get(dst)
    if region:
        return obj, region.get("owner_instance"), dst, relation
    return obj, dst, None, relation


@lru_cache(maxsize=1)
def cached_catalog() -> tuple[dict[str, Any], ...]:
    return tuple(load_task_catalog())
