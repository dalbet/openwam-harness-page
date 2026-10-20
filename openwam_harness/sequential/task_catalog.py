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
    return found


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


def lookup_task(task_key: str | None, instruction_en: str | None, suite: str, names: set[str] | list[str]) -> dict[str, Any] | None:
    rows = tasks_for_scene(suite, names)
    if isinstance(task_key, str) and task_key:
        for row in rows:
            if row["task_key"] == task_key:
                return row
    if isinstance(instruction_en, str) and instruction_en.strip():
        wanted = instruction_en.strip()
        for row in rows:
            if row["instruction_en"] == wanted:
                return row
    return None


def single_task_block(plan: dict[str, Any], row: dict[str, Any] | None) -> str | None:
    """Refuse combo plans and unsupported skills after the model answers. Do not execute them."""
    count = plan.get("semantic_step_count")
    if not isinstance(count, int):
        count = len([step for step in (plan.get("steps") or []) if isinstance(step, dict) and step.get("skill") == "transfer"])
    if count > 1:
        return "当前仅执行单任务，不会按组合目标继续。"
    if row is None:
        return "标准指令不在当前场景的任务目录中，不会切换场景。"
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
    return [(rel, obj, dst) for rel, obj, dst in re.findall(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", body)]


def _slots(goals: list[tuple[str, str, str]], regions: dict[str, dict]) -> tuple[str | None, str | None, str | None, str | None]:
    if not goals:
        return None, None, None, None
    rel, obj, dst = goals[0]
    relation = "in" if rel == "In" else "on"
    region = regions.get(dst)
    if region:
        return obj, region.get("owner_instance"), dst, relation
    return obj, dst, None, relation


@lru_cache(maxsize=1)
def cached_catalog() -> tuple[dict[str, Any], ...]:
    return tuple(load_task_catalog())
