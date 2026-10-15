"""Stage B: capability matrix, Chinese matching, and numbered candidates."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from candidates import link_visual_candidate, number_candidates, resolve_numbers, sam3_contract_from_source, scene_graph
from capabilities import builtin_matrix, reject_reason
from instruction import resolve_instruction
from planner import request_plan
from schema import semantic_example


ROOT = Path(__file__).resolve().parent
SAM3 = ROOT.parents[0] / "RPent" / "rpent" / "robots" / "components" / "sam3_server.py"


class StageBTest(unittest.TestCase):
    def test_matrix_covers_forty_tasks_without_calling_them_successful(self):
        rows = builtin_matrix()
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["task_key"] for row in rows}), 40)
        self.assertTrue(all(row["local_acceptance"] == "not_run" for row in rows))
        self.assertTrue(all(row["compilable_is_success"] is False for row in rows))
        opened = [row["kind"] for row in rows if row["demo_open"]]
        self.assertIn("object_transport", opened)
        self.assertIn("spatial_pick_place", opened)
        self.assertIn("goal_transport", opened)
        self.assertIn("long_two_foods_basket", opened)
        stove = next(row for row in rows if row["language"] == "turn on the stove")
        self.assertFalse(stove["demo_open"])
        self.assertIn("turn_on_stove", stove["missing_skills"])
        self.assertIsNotNone(reject_reason(stove))
        self.assertIsNone(reject_reason(next(row for row in rows if row["kind"] == "object_transport")))

    def test_chinese_match_asks_once_and_does_not_keep_a_different_goal(self):
        loaded = {"suite": "libero_spatial", "task_id": 0, "goal": builtin_matrix()[0]["language"], "summary_zh": builtin_matrix()[0]["summary_zh"]}
        ambiguous = resolve_instruction("把黑碗放到盘子上", loaded)
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertIn("不会默认第一条", ambiguous["question"])
        self.assertGreater(len(ambiguous["candidates"]), 1)
        exact = resolve_instruction("抓起字母汤，并放进篮子。", {"suite": "libero_object", "task_id": 1, "goal": "pick up the cream cheese and place it in the basket"})
        self.assertEqual(exact["status"], "matched")
        self.assertTrue(exact["changed"])
        self.assertEqual(exact["language"], "pick up the alphabet soup and place it in the basket")
        unsupported = resolve_instruction("飞到月球", loaded)
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertIn("不会沿用", unsupported["reason"])

    def test_candidate_numbers_keep_duplicates_and_record_evidence(self):
        objects = {
            "akita_black_bowl_1": {"semantic_type": "akita_black_bowl", "phrases": ["black bowl"], "movable": True, "capability_source": "bddl:objects"},
            "akita_black_bowl_2": {"semantic_type": "akita_black_bowl", "phrases": ["black bowl"], "movable": True, "capability_source": "bddl:objects"},
            "plate_1": {"semantic_type": "plate", "phrases": ["plate"], "movable": False, "can_be_target": True, "target_relations": ["on"], "capability_source": "bddl:fixtures"},
        }
        numbered = number_candidates(objects)
        self.assertEqual([item["semantic_type"] for item in numbered].count("akita_black_bowl"), 2)
        graph = scene_graph(objects, {"akita_black_bowl_1": [0, 0, 0], "plate_1": [0.05, 0, 0]})
        self.assertEqual(graph["evidence_source"], "sim_state")
        plan = semantic_example()
        plan["steps"][0]["object_n"] = 1
        plan["steps"][0]["target_n"] = 3
        plan["steps"][0]["object_ref"] = "akita_black_bowl_1"
        plan["steps"][0]["target_ref"] = "plate_1"
        resolved, evidence = resolve_numbers(plan, numbered)
        self.assertEqual(resolved["steps"][0]["object_ref"], "akita_black_bowl_1")
        self.assertNotIn("object_n", resolved["steps"][0])
        self.assertEqual(evidence[0]["evidence_source"], "planner_candidate_number")
        with self.assertRaises(ValueError):
            bad = semantic_example()
            bad["steps"][0]["object_n"] = 1
            bad["steps"][0]["object_ref"] = "akita_black_bowl_2"
            resolve_numbers(bad, numbered)
        visual = link_visual_candidate({"found": True, "score": 0.9, "semantic_type": "akita_black_bowl"}, numbered)
        self.assertIsNone(visual["object_ref"])
        self.assertEqual(visual["evidence_source"], "sam3_top1_unlinked")

    def test_sam3_source_returns_only_the_top_score(self):
        if not SAM3.is_file():
            self.skipTest("SAM3 server source is not on this machine")
        contract = sam3_contract_from_source(SAM3.read_text(encoding="utf-8"))
        self.assertTrue(contract["source_has_argmax"])
        self.assertTrue(contract["source_has_point"])
        self.assertEqual(contract["text_segment"], "top_score_only")
        self.assertIsNone(contract["multi_candidate_rpc"])

    def test_planner_resolution_is_kept_beside_the_validated_plan(self):
        objects = {
            "alphabet_soup": {"sim_name": "alphabet_soup_1", "phrases": ["alphabet soup"], "semantic_type": "alphabet_soup", "movable": True, "role": "movable"},
            "basket": {"sim_name": "basket_1", "phrases": ["basket"], "semantic_type": "basket", "movable": False, "can_be_target": True, "target_relations": ["in"], "role": "container"},
        }
        numbered = number_candidates(objects)
        example = semantic_example()
        example["goal"] = "pick up the alphabet soup and place it in the basket"
        example["steps"][0]["object_n"] = 1
        example["steps"][0]["target_n"] = 2

        def client(messages, max_tokens, options=None):
            return {
                "content": None,
                "tool_calls": [{"id": "call", "type": "function", "function": {"name": "submit_plan", "arguments": json.dumps(example)}}],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

        result = request_plan(
            task=example["goal"],
            scene_observation="candidates",
            client=client,
            scene_objects=objects,
            numbered_candidates=numbered,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["candidate_resolution"][0]["candidate_n"], 1)
        self.assertEqual(result["candidate_resolution"][0]["evidence_source"], "planner_candidate_number")
        self.assertNotIn("_candidate_resolution", result["plan"])


if __name__ == "__main__":
    unittest.main()
