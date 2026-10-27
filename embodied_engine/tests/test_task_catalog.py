"""Lock the ten-task catalog to the installed LIBERO-Goal benchmark.

The catalog is the only place task names, instructions and goal atoms are
written by hand (``config/tasks.json``). These tests re-derive the same facts
from the installed LIBERO package, so a typo in the catalog, a silent upgrade
of the package, or a reworded instruction all fail here instead of showing up
as a confusing routing result weeks later.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import task_catalog  # noqa: E402

LIBERO = Path(
    "/home/host/page-dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/libero/libero"
)
BDDL_DIR = LIBERO / "bddl_files" / "libero_goal"
INIT_DIR = LIBERO / "init_files" / "libero_goal"
TASK_MAP = LIBERO / "benchmark" / "libero_suite_task_map.py"

BDDL_LANGUAGE = {
    0: "Open the middle layer of the drawer",
    2: "Put the wine bottle on the top of the drawer",
    3: "Open the top layer of the drawer and put the bowl inside",
    4: "Put the bowl on the top of the drawer",
    6: "Put the cream cheese on the bowl",
}
"""The five BDDL ``:language`` strings that differ from the benchmark prompt."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _benchmark_task_order() -> list[str]:
    tree = ast.parse(TASK_MAP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)["libero_goal"]
    raise AssertionError("libero_task_map not found in the installed LIBERO package")


def _filename_language(bddl: str) -> str:
    """Reimplementation of libero/benchmark/__init__.py::grab_language_from_filename."""
    language = " ".join(Path(bddl).stem.split("_"))
    end = language.find(".bddl")
    return language[:end] if end >= 0 else language


class CatalogShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = task_catalog.tasks()

    def test_ten_tasks_in_order(self) -> None:
        self.assertEqual(len(self.rows), 10)
        self.assertEqual([row["task_id"] for row in self.rows], list(range(10)))

    def test_keys_unique_and_non_empty(self) -> None:
        keys = [row["key"] for row in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        for key in keys:
            self.assertRegex(key, r"^[a-z][a-z0-9_]*$")

    def test_required_fields_present(self) -> None:
        for row in self.rows:
            for field in (
                "official_instruction",
                "bddl_language",
                "bddl",
                "kind",
                "objects",
                "targets",
                "goal_atoms",
                "semantics",
                "capability",
            ):
                self.assertIn(field, row, f"task {row['task_id']} missing {field}")
            self.assertTrue(row["official_instruction"].strip())
            self.assertTrue(row["goal_atoms"])
            self.assertTrue(row["capability"].strip())

    def test_legacy_instruction_matches_official(self) -> None:
        for row in self.rows:
            self.assertEqual(row["instruction"], row["official_instruction"])

    def test_pick_place_has_source_and_target(self) -> None:
        for row in self.rows:
            semantics = row["semantics"]
            if row["kind"] in {"pick_place", "push", "composite"}:
                self.assertTrue(semantics["source"], f"task {row['task_id']} has no source")
                self.assertTrue(semantics["target"], f"task {row['task_id']} has no target")

    def test_only_task_three_is_composite(self) -> None:
        composite = [row["task_id"] for row in self.rows if row["kind"] == "composite"]
        self.assertEqual(composite, [3])
        task3 = task_catalog.task_for(3)
        self.assertEqual(len(task3["semantics"]["subgoals"]), 2)

    def test_capability_rows_feed_the_router(self) -> None:
        rows = task_catalog.capability_rows()
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertTrue({"key", "operation", "capability"} <= set(row))
            self.assertTrue(row["capability"].strip())
            self.assertTrue(row["capability"].endswith("."))


class CatalogMatchesLiberoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = task_catalog.tasks()

    def test_task_order_matches_installed_benchmark(self) -> None:
        expected = _benchmark_task_order()
        self.assertEqual([Path(row["bddl"]).stem for row in self.rows], expected)
        self.assertEqual(task_catalog.libero_identity()["task_order"], expected)

    def test_official_instruction_is_derived_from_filename(self) -> None:
        for row in self.rows:
            self.assertEqual(
                row["official_instruction"],
                _filename_language(row["bddl"]),
                f"task {row['task_id']} instruction is not the filename language",
            )

    def test_bddl_and_init_hashes_match_install(self) -> None:
        for row in self.rows:
            bddl = BDDL_DIR / row["bddl"]
            init = INIT_DIR / row["init_file"]
            self.assertTrue(bddl.is_file(), f"missing {bddl}")
            self.assertTrue(init.is_file(), f"missing {init}")
            self.assertEqual(row["bddl_sha256"], _sha256(bddl))
            self.assertEqual(row["init_sha256"], _sha256(init))

    def test_goal_atoms_match_bddl_goal(self) -> None:
        for row in self.rows:
            text = (BDDL_DIR / row["bddl"]).read_text(encoding="utf-8")
            goal = text[text.index("(:goal") :]
            goal = goal[: goal.index(")\n\n") + 1] if ")\n\n" in goal else goal
            atoms = set()
            for match in re.finditer(r"\(([A-Za-z]+)([^()]*)\)", goal):
                tokens = [match.group(1), *match.group(2).split()]
                if len(tokens) >= 2 and tokens[0] not in {"And", "and"}:
                    atoms.add(" ".join(tokens))
            for atom in row["goal_atoms"]:
                self.assertIn(
                    " ".join(atom.split()),
                    atoms,
                    f"task {row['task_id']} atom {atom!r} not in BDDL goal {sorted(atoms)}",
                )

    def test_recorded_bddl_language_is_stable(self) -> None:
        for task_id, expected in BDDL_LANGUAGE.items():
            row = task_catalog.task_for(task_id)
            self.assertEqual(row["bddl_language"], expected)
            self.assertNotEqual(
                row["bddl_language"],
                row["official_instruction"],
                f"task {task_id} is expected to differ from its BDDL language",
            )

    def test_identity_is_pinned(self) -> None:
        identity = task_catalog.libero_identity()
        self.assertEqual(identity["commit"], "a8323074d93a09e32bd898630a70531b1f51bc77")
        self.assertEqual(
            identity["tasks_info_sha256"],
            "547286ceb6485e85aa1472c03466a613573982d92284d792c1ab63524b479382",
        )


class CatalogFingerprintTest(unittest.TestCase):
    def test_fingerprint_is_sha256_of_file_bytes(self) -> None:
        raw = (PROJECT / "config" / "tasks.json").read_bytes()
        self.assertEqual(task_catalog.fingerprint(), hashlib.sha256(raw).hexdigest())
        self.assertRegex(task_catalog.fingerprint(), r"^[0-9a-f]{64}$")

    def test_version_and_schema_are_recorded(self) -> None:
        self.assertEqual(task_catalog.schema(), "embodied_engine.task_catalog.v2")
        self.assertEqual(task_catalog.version(), "2026-10-27.1")

    def test_catalog_is_json_and_round_trips(self) -> None:
        raw = json.loads((PROJECT / "config" / "tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], task_catalog.schema())
        self.assertEqual(len(raw["tasks"]), 10)


class SceneSupportTest(unittest.TestCase):
    def _snapshot(self, objects: tuple[str, ...] = (), sites: tuple[str, ...] = ()) -> dict:
        return {
            "sim": {"objects": {name: {"pos": [0.0, 0.0, 0.0]} for name in objects}},
            "named_poses": {"sites": {name: {"position": [0.0, 0.0, 0.0]} for name in sites}},
        }

    def test_empty_scene_supports_nothing(self) -> None:
        for row in task_catalog.tasks():
            self.assertFalse(task_catalog.scene_support(row, self._snapshot())["ok"])

    def test_stove_is_found_through_its_cook_region_site(self) -> None:
        """Regression: the stove body is only visible as ``flat_stove_1_*`` sites."""
        row = task_catalog.task_for(7)
        snapshot = self._snapshot(
            objects=("flat_stove_1_main",),
            sites=("flat_stove_1_cook_region",),
        )
        support = task_catalog.scene_support(row, snapshot)
        self.assertTrue(support["ok"], support["missing"])

    def test_pick_place_needs_both_object_and_target(self) -> None:
        row = task_catalog.task_for(1)
        only_bowl = self._snapshot(objects=("akita_black_bowl_1", "flat_stove_1_main"))
        self.assertFalse(task_catalog.scene_support(row, only_bowl)["ok"])
        full = self._snapshot(
            objects=("akita_black_bowl_1", "flat_stove_1_main"),
            sites=("flat_stove_1_cook_region",),
        )
        self.assertTrue(task_catalog.scene_support(row, full)["ok"])


if __name__ == "__main__":
    unittest.main()
