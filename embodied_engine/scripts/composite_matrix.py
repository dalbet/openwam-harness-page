#!/usr/bin/env python
"""Same-scene two-step feasibility, derived from the installed LIBERO assets.

The question is narrow: can two of the ten LIBERO-goal tasks be run back to back
in one episode without reloading the scene? Answering it needs two kinds of
evidence, and they are kept apart on purpose.

Derived (recomputed every run, cannot drift from the benchmark):
    the fixtures and objects each BDDL declares, the init region of every
    asset, the goal atoms, and the sha256 of each pruned_init.

Curated (a human judgement, one line of reason each):
    whether step A destroys step B's precondition, and whether step A's success
    is checkable by the engine's own predicates.

The curated rows are not the verdict of a run. They are the shortlist the next
phase is allowed to start from; nothing in this file has been executed as a
composite episode yet.

    python scripts/composite_matrix.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBERO = Path(
    "/home/host/page-dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/libero/libero"
)
TASKS = ROOT / "config" / "tasks.json"
OUT = ROOT / "records" / "acceptance" / "composite_matrix_20261027.json"


def section(text: str, name: str) -> str:
    match = re.search(r"\(" + name + r"\s(.*?)\n\s*\)\n", text, re.S)
    return match.group(1) if match else ""


def parse_bddl(path: Path) -> dict:
    text = path.read_text()
    fixtures = [name for name, _ in re.findall(r"^\s+(\S+)\s+-\s+(\w+)\s*$", section(text, ":fixtures"), re.M)]
    objects = [name for name, _ in re.findall(r"^\s+(\S+)\s+-\s+(\w+)\s*$", section(text, ":objects"), re.M)]
    init = dict(re.findall(r"\(On (\S+) (\S+)\)", section(text, ":init")))
    goal = " ".join(section(text, ":goal").split())
    return {
        "bddl": path.name,
        "fixtures": sorted(fixtures),
        "objects": sorted(objects),
        "init_regions": init,
        "goal": goal,
    }


# One line per pair: what the second goal needs, and whether the first goal
# leaves it true. "step_a_verifiable" names the engine predicate that would
# accept step A without the env's terminated flag.
CURATED = {
    "0->8": {
        "verdict": "viable",
        "order": 1,
        "reason": "opening the middle drawer moves nothing the bowl-on-plate goal needs; step A is accepted by completion.drawer_open (qpos < -0.14)",
        "step_a_verifiable": "drawer_open",
        "second_step_precondition": "akita_black_bowl_1 and plate_1 untouched by step A",
        "risk": "none identified",
    },
    "1->7": {
        "verdict": "viable",
        "order": 2,
        "reason": "cook_region sits 0.15 m along +x from the stove button hinge, so a bowl on the burner need not cover the knob; the only pair whose two goals describe one cooking act",
        "step_a_verifiable": "object_on_target (On bowl flat_stove_1_cook_region)",
        "second_step_precondition": "flat_stove_1 button joint free and reachable",
        "risk": "the bowl may sit in the front approach path to the knob; needs one real probe before it is called feasible",
    },
    "7->1": {
        "verdict": "allowed_but_backwards",
        "order": 3,
        "reason": "same assets as 1->7 in reverse; no thermal model in the scene, so nothing breaks, but the sentence is semantically odd",
        "step_a_verifiable": "stove_on",
        "second_step_precondition": "cook_region free",
        "risk": "an already-on stove is off the OpenWAM training distribution",
    },
    "6->8": {
        "verdict": "coupled",
        "order": 4,
        "reason": "step B carries the bowl that step A's goal relation (cream cheese on bowl) depends on",
        "step_a_verifiable": "object_on_target (On cream_cheese_1 akita_black_bowl_1)",
        "second_step_precondition": "cream cheese stays in the bowl while the bowl is carried",
        "risk": "the first goal can be silently broken during transport; the composite would need both atoms re-checked at the end, not just the last one",
    },
    "8->4": {
        "verdict": "same_object_relocation",
        "order": 5,
        "reason": "both steps move only the bowl; this is one skill applied twice, not two skills composed",
        "step_a_verifiable": "object_on_target",
        "second_step_precondition": "re-grasp the bowl from the plate",
        "risk": "still needs release + retreat + a fresh observation between steps, which the finalizer now provides",
    },
    "3->8": {
        "verdict": "rejected",
        "order": 6,
        "reason": "step A leaves the bowl inside the top drawer, so step B has to reach into a drawer the policy has never been trained to grasp from",
        "step_a_verifiable": "object_in_region",
        "second_step_precondition": "bowl graspable outside a drawer",
        "risk": "deep-region grasp is the exact failure mode task 3 already shows (round 7)",
    },
    "0->3": {
        "verdict": "rejected",
        "order": 7,
        "reason": "task 3 is itself the official drawer-then-place composite; using it as a step would sell an existing capability as a new one",
        "step_a_verifiable": "drawer_open",
        "second_step_precondition": "top drawer open, which task 3's own goal already requires",
        "risk": "would double-count the official composite",
    },
}


def main() -> int:
    catalog = json.loads(TASKS.read_text())
    rows = []
    for task in catalog["tasks"]:
        facts = parse_bddl(LIBERO / "bddl_files" / "libero_goal" / task["bddl"])
        rows.append(
            {
                "task_id": task["task_id"],
                "key": task["key"],
                **facts,
                "init_sha256": task["init_sha256"],
                "init_file": task["init_file"],
            }
        )

    asset_sets = {json.dumps([row["fixtures"], row["objects"]]) for row in rows}
    init_regions = {json.dumps(row["init_regions"], sort_keys=True) for row in rows}
    inits = {row["init_sha256"] for row in rows}
    derived = {
        "tasks": len(rows),
        "distinct_asset_sets": len(asset_sets),
        "distinct_semantic_init_layouts": len(init_regions),
        "distinct_pruned_init_files": len(inits),
        "co_present_assets": rows[0]["fixtures"] + rows[0]["objects"],
        "finding": (
            "all ten BDDLs declare the same four fixtures and four objects and the same "
            "semantic init regions; only the goal atoms and the sampled pruned_init differ. "
            "Any pair is therefore same-scene by construction, but a composed episode must "
            "start from one task's pruned_init, so the second goal begins from a state that "
            "is not its own official start."
        ),
    }

    pairs = []
    for key, item in CURATED.items():
        first, second = (int(part) for part in key.split("->"))
        pairs.append(
            {
                "pair": key,
                "first": {"task_id": first, "key": next(r["key"] for r in rows if r["task_id"] == first)},
                "second": {"task_id": second, "key": next(r["key"] for r in rows if r["task_id"] == second)},
                **item,
            }
        )
    pairs.sort(key=lambda item: item["order"])

    payload = {
        "schema": "embodied_engine.composite_matrix.v1",
        "status": "design_only_no_composite_episode_executed",
        "derived": derived,
        "pairs": pairs,
        "entry_point": {
            "episode_bddl": "the second task's BDDL, so its native goal stays benchmark-valid",
            "first_step": "run in mode=free against the same env; accept it with the engine predicate named in step_a_verifiable, never with env terminated",
            "handoff": "PlacementFinalizer release + retreat + new observation + cleared policy buffer before step two",
            "combo_goal_evidence": "both atoms recorded separately; env terminated only ever attests the second atom",
            "refused": "no reset, no scene reload, no synthetic BDDL editing",
        },
        "not_done": [
            "no composite episode has been run",
            "the router still returns one task per utterance",
            "the legacy openwam_harness/sequential combo flag is a different stack and is not wired to this engine",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(json.dumps(derived, ensure_ascii=False, indent=1))
    for pair in pairs:
        print(f"{pair['pair']:>6} {pair['verdict']:<24} {pair['reason'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
