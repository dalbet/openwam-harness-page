"""Free-task understanding on a fixed scene.

The task catalog is not a gate. An unknown sentence is not rewritten into
the nearest official task, and it does not change ``task_id``.
"""

from __future__ import annotations

from typing import Any

from schema import THRESHOLDS, _err

# Optional context for the planner. Matching a phrase does not bind an instance
# and does not reject an unknown wording such as 奶酪 or 那盒乳酪.
_ZH = (
    ("奶油奶酪", ("cream cheese",)),
    ("黑碗", ("black bowl", "akita black bowl")),
    ("酒瓶", ("wine bottle",)),
    ("炉灶", ("stove",)),
    ("盘子", ("plate",)),
    ("酒架", ("wine rack", "rack")),
    ("木柜", ("wooden cabinet", "cabinet")),
    ("柜子", ("cabinet", "wooden cabinet")),
    ("碗", ("bowl", "black bowl")),
)


class EpisodeFlags:
    """Official single tasks stop when the native goal is met. Combo is explicit."""

    def __init__(self, mode: str = "benchmark", task_scope: str = "official_single") -> None:
        if mode not in {"benchmark", "free"}:
            raise ValueError(mode)
        if task_scope not in {"official_single", "combo"}:
            raise ValueError(task_scope)
        self.mode = mode
        self.task_scope = task_scope
        self.native_goal_met = False
        self.truncated = False

    def absorb(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        if truncated:
            self.truncated = True
            return {"kind": "time_limit", "clear_terminated": False, "continuable": False}
        if terminated:
            self.native_goal_met = True
            if self.mode == "free" and self.task_scope == "combo":
                return {"kind": "native_goal", "clear_terminated": True, "continuable": True}
            return {"kind": "native_goal", "clear_terminated": False, "continuable": False}
        return {"kind": "running", "clear_terminated": False, "continuable": self.continuable()}

    def continuable(self) -> bool:
        if self.truncated or self.native_goal_met and not (self.mode == "free" and self.task_scope == "combo"):
            return False
        return True


def prepare_free_request(
    text: str,
    objects: dict[str, Any],
    poses: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Let the planner read the sentence. Do not bind or reject names here."""
    del poses
    typed = text.strip()
    return {
        "status": "ready",
        "changed": False,
        "user_text": typed,
        "predicates": [],
        "ordered": any(mark in typed for mark in ("先", "再", "然后")),
        "alias_hints": _alias_hints(typed, objects),
        "pre_gate": False,
        "evidence_source": "sim_state",
        "vision": False,
    }


def candidate_number_errors(plan: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return errors
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("skill") != "transfer":
            continue
        if step.get("object_n") is None:
            errors.append(_err(f"steps[{index}].object_n", "candidate_number", None, "当前场景候选编号"))
        if step.get("target_n") is None:
            errors.append(_err(f"steps[{index}].target_n", "candidate_number", None, "当前场景候选编号"))
    return errors


def coverage_errors(plan: dict[str, Any], predicates: list[dict[str, Any]], *, ordered: bool) -> list[dict[str, Any]]:
    """The plan must place every user predicate. Official task text is not the target."""
    errors = []
    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    open_steps = list(steps)
    chosen = []
    for predicate in predicates:
        found = None
        for step in open_steps:
            if (
                step.get("object_ref") == predicate["object_ref"]
                and step.get("target_ref") == predicate["target_ref"]
                and step.get("relation") == predicate["relation"]
                and step.get("skill") == "transfer"
            ):
                found = step
                break
        if found is None:
            errors.append(
                _err(
                    "plan.steps",
                    "user_goal_uncovered",
                    predicate,
                    "一个覆盖该用户要求的 transfer",
                )
            )
            continue
        open_steps.remove(found)
        chosen.append(found)
    chain = chosen if predicates else [step for step in steps if step.get("skill") == "transfer"]
    if ordered and len(chain) > 1:
        earlier = chain[0].get("step_id")
        for step in chain[1:]:
            depends = step.get("depends_on") or []
            if earlier not in depends:
                errors.append(
                    _err(
                        f"steps.{step.get('step_id')}.depends_on",
                        "sequence_dependency",
                        depends,
                        f"依赖已完成的 {earlier}。先后关系不是重复的 transfer",
                    )
                )
            earlier = step.get("step_id")
    return errors


def judge_user_goal(
    predicates: list[dict[str, Any]],
    sim: dict[str, Any] | None,
    observation: dict[str, Any] | None = None,
    scene_objects: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backend check. A planner success flag is not read.

    A fixture with a placement site is judged by SiteObject.under. Center distance is only
    for a target that has no site declaration.
    """
    state = sim or {}
    objects = state.get("objects") or {}
    cfg = THRESHOLDS["place"]
    rows = []
    sources = set()
    for predicate in predicates:
        obj = _pos(objects, predicate["object_ref"])
        spec = (scene_objects or {}).get(predicate["target_ref"]) if isinstance(scene_objects, dict) else None
        site = spec.get("place_site") if isinstance(spec, dict) else None
        held = False
        source = "sim_state"
        metrics: dict[str, Any] = {"relation": predicate["relation"], "hold": predicate.get("hold") or "end"}
        if site:
            native = _site_relation(obj, site, observation)
            metrics["query_key"] = site
            metrics["native_on"] = native
            if native is None:
                source = "pose_not_provided"
                metrics["reason_code"] = "pose_not_provided"
            else:
                held = bool(native["ok"])
                source = "mujoco_site"
        else:
            from transfer_geometry import relation_check

            dst = _pos(objects, predicate["target_ref"])
            shaped = relation_check(obj, dst, predicate["object_ref"], predicate["target_ref"])
            if shaped is not None:
                held = bool(shaped["ok"])
                source = "asset_geometry"
                metrics.update(shaped)
            elif obj is not None and dst is not None:
                xy = ((obj[0] - dst[0]) ** 2 + (obj[1] - dst[1]) ** 2) ** 0.5
                radius = float(cfg["in_radius_m"] if predicate["relation"] == "in" else cfg["on_radius_m"])
                z_ok = abs(obj[2] - dst[2]) <= float(cfg["support_z_tolerance_m"])
                held = xy <= radius and z_ok
                metrics["xy_m"] = round(xy, 4)
                metrics["z_delta_m"] = round(obj[2] - dst[2], 4)
                metrics["predicate"] = "center_distance"
        sources.add(source)
        rows.append(
            {
                "object_ref": predicate["object_ref"],
                "target_ref": predicate["target_ref"],
                "relation": predicate["relation"],
                "hold": predicate.get("hold") or "end",
                "held_at_end": held,
                "instant_only": False,
                "evidence_source": source,
                "metrics": metrics,
            }
        )
    end_rows = [row for row in rows if row["hold"] == "end"]
    return {
        "subgoal_results": rows,
        "user_goal_success": bool(end_rows) and all(row["held_at_end"] for row in end_rows),
        "evidence_source": "per_item" if len(sources) > 1 else (next(iter(sources)) if sources else "sim_state"),
        "vision": False,
    }


def _site_relation(obj, site: str, observation: dict[str, Any] | None) -> dict[str, Any] | None:
    named = (observation or {}).get("named_poses") if isinstance(observation, dict) else None
    if not isinstance(named, dict) or not isinstance(obj, list):
        return None
    geometry = (named.get("site_geometry") or {}).get(site)
    xyz = (named.get("sites") or {}).get(site)
    if not isinstance(geometry, dict) or not geometry.get("mat") or not geometry.get("size") or not isinstance(xyz, list):
        return None
    from regions import on_site

    return on_site(obj, xyz, geometry["mat"], geometry["size"])


def _alias_hints(text: str, objects: dict[str, Any]) -> list[dict[str, Any]]:
    """Catalog phrases the sentence happens to contain. Not a binding."""
    hints = []
    for zh, english in _ZH:
        if zh not in text:
            continue
        refs = []
        for ref, spec in objects.items():
            hay = " ".join(str(item) for item in (spec.get("phrases") or [])).lower()
            if any(key in hay for key in english):
                refs.append(ref)
        hints.append({"user_phrase": zh, "candidate_refs": refs, "binding": "hint"})
    return hints


def _pos(objects: dict[str, Any], ref: str) -> list[float] | None:
    item = objects.get(ref)
    if isinstance(item, dict):
        pos = item.get("pos")
        if isinstance(pos, list) and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    return None
