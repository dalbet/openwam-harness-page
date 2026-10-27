#!/usr/bin/env python
"""Offline planning comparison: Qwen whole-task planner vs d1 decision planner.

Three engines answer the same frozen cases against the same live scene:

* ``qwen_full`` - the production Qwen whole-task planner (``planner.request_plan``).
  It emits one ``whole_task`` step, so it has no explicit object/site binding;
  binding accuracy is reported as not applicable rather than invented.
* ``d1`` - the resident d1-3B decision service with candidate assembly,
  thresholds, goal-preservation checks and contract validation.
* ``qwen_decision`` - Qwen answering the *same* staged choice/noul questions in
  d1's answer schema, then running through the *same* assembly and validation.
  This isolates the model difference from the output-format change.

The scene is loaded per task through the engine, and the snapshot is read from
the same env server the engine uses. Cases only contain instructions; the
planner never receives BDDL goals. Gold bindings are used for scoring only.

Usage (RPent python, which has numpy and the RPent RPC client):
    python scripts/eval_offline.py --set config/offline_calibration.json \
        --out records/offline/calibration.json --raw
    python scripts/eval_offline.py --set config/offline_eval.json \
        --out records/offline/eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.env_client import EnvClient  # noqa: E402
from agent import d1_planner, planner  # noqa: E402
from common import config  # noqa: E402
from common.plan_contract import validate_plan  # noqa: E402


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


# ---------------------------------------------------------------------------
# Qwen answering d1's staged questions
# ---------------------------------------------------------------------------

QWEN_DECISION_SYSTEM = """You are the decision module of a LIBERO-Goal robot planner.
You do not act and you do not write prose. You answer the named questions about one scene.

Return exactly one JSON object:
{"answers": {"<question name>": <answer>, ...}}

Answer forms:
- type "choice": {"type": "choice", "choice": "<exactly one criteria key>", "confidence": <0..1>}
  The confidence is your probability that this choice is the correct binding.
- type "noul": {"type": "noul", "noul": <0..1>}
  The value is the probability that the answer is yes.

Use exactly the question names that were asked, no extra keys. A choice must be
one of that question's criteria keys, copied verbatim. Do not explain.
"""


class QwenDecisionClient:
    """``system_one``-compatible client backed by the Qwen chat endpoint."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": QWEN_DECISION_SYSTEM},
            {"role": "user", "content": payload},
        ]
        started = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        content, timing = planner.chat(messages, max_tokens=600)
        input_tokens += int(timing.get("prompt_tokens") or 0)
        output_tokens += int(timing.get("completion_tokens") or 0)
        answers, errors = _parse_decision_answers(content, questions)
        attempts = [
            {
                "kind": "initial",
                "elapsed_s": timing.get("elapsed_s"),
                "prompt_tokens": timing.get("prompt_tokens"),
                "completion_tokens": timing.get("completion_tokens"),
                "ttft_s": timing.get("ttft_s"),
                "decode_s": timing.get("decode_s"),
            }
        ]
        if errors:
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "validation_errors": errors,
                            "request": "Return one corrected JSON object with exactly the asked question names and valid answer forms. Do not explain.",
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            content, timing = planner.chat(messages, max_tokens=600)
            input_tokens += int(timing.get("prompt_tokens") or 0)
            output_tokens += int(timing.get("completion_tokens") or 0)
            answers, errors = _parse_decision_answers(content, questions)
            attempts.append(
                {
                    "kind": "format_repair",
                    "elapsed_s": timing.get("elapsed_s"),
                    "prompt_tokens": timing.get("prompt_tokens"),
                    "completion_tokens": timing.get("completion_tokens"),
                    "ttft_s": timing.get("ttft_s"),
                    "decode_s": timing.get("decode_s"),
                }
            )
        elapsed = time.perf_counter() - started
        if errors:
            raise RuntimeError(f"Qwen decision format invalid after retry: {errors[:3]}")
        record = {
            "attempts": attempts,
            "elapsed_s": round(elapsed, 4),
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
        self.calls.append(record)
        return {
            "status": "ok",
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_s": round(elapsed, 4),
        }


def _parse_decision_answers(
    content: str, questions: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value, parse_error = planner.parse_json_object(content)
    if value is None:
        return {}, [{"field": "response", "code": "invalid_json", "value": parse_error}]
    answers = value.get("answers")
    if not isinstance(answers, dict):
        return {}, [{"field": "answers", "code": "missing_object", "value": type(answers).__name__}]
    out: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for name, spec in questions.items():
        if name not in answers:
            errors.append({"field": name, "code": "answer_missing"})
            continue
        raw = answers[name]
        if not isinstance(raw, dict):
            errors.append({"field": name, "code": "answer_not_object"})
            continue
        if spec.get("type") == "choice":
            choice = raw.get("choice")
            criteria = spec.get("criteria") or {}
            if choice not in criteria:
                errors.append({"field": name, "code": "choice_not_in_criteria", "value": choice})
                continue
            out[name] = {
                "type": "choice",
                "choice": choice,
                "confidence": _probability(raw.get("confidence"), errors, name),
            }
        elif spec.get("type") == "noul":
            out[name] = {"type": "noul", "noul": _probability(raw.get("noul"), errors, name)}
        else:
            errors.append({"field": name, "code": "unknown_question_type"})
    if errors:
        return {}, errors
    return out, []


def _probability(value: Any, errors: list[dict[str, Any]], name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append({"field": name, "code": "probability_not_number", "value": value})
        return 0.0
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _norm(value: Any) -> str | None:
    if value in (None, "", "none"):
        return None
    return str(value)


def _binding(result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision") or {}
    return {
        "operation": _norm(decision.get("operation")),
        "source": _norm(decision.get("source")),
        "target": _norm(decision.get("target")),
        "relation": _norm(decision.get("relation")),
        "open_target": _norm(decision.get("open_target")),
    }


def _binding_checks(got: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in ("operation", "source", "target", "relation"):
        if key in expected:
            checks[key] = got.get(key) == _norm(expected.get(key))
    if "open_target" in expected:
        checks["open_target"] = got.get("open_target") == _norm(expected.get("open_target"))
    return checks


REJECT_GROUPS = {
    "unsupported": {"request_not_supported", "operation_not_supported"},
    "missing_information": {"source_not_specified", "target_not_specified", "composite_incomplete"},
    "needs_clarification": {
        "ambiguous_reference",
        "source_uncertain",
        "target_uncertain",
        "composite_uncertain",
        "relation_uncertain",
        "drawer_level_not_specified",
    },
}


def _reject_group(status: str, reason_code: str | None) -> str | None:
    if status == "accepted":
        return None
    for group, codes in REJECT_GROUPS.items():
        if reason_code in codes:
            return group
    if status in REJECT_GROUPS:
        return status
    return "rejected"


def _contract_ok(
    plan: dict[str, Any] | None,
    *,
    case: dict[str, Any],
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    plan_source: str,
) -> tuple[bool, list[dict[str, Any]]]:
    if not plan:
        return False, [{"field": "plan", "code": "missing"}]
    _cleaned, errors = validate_plan(
        plan,
        instruction=case["instruction"],
        task=task,
        snapshot=snapshot,
        skills=skills,
        plan_source=plan_source,
    )
    return not errors, errors


DRAWER_LEVELS = ("top", "middle", "bottom")


def _prompt_words(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (prompt or "").lower()))


def _prompt_covers_goal(prompt: str, expected: dict[str, Any]) -> dict[str, bool]:
    words = _prompt_words(prompt)
    operation = expected.get("operation")
    checks: dict[str, bool] = {}
    if operation == "open_drawer":
        checks["open"] = "open" in words
        checks["target"] = d1_planner._covered(prompt, expected.get("target"), drawer=True)
        level = d1_planner._drawer_parts(expected.get("target") or "")[1]
        if level:
            checks["level"] = level in words
        return checks
    if operation == "turn_on":
        checks["turn_on"] = d1_planner._relation_in_instruction(prompt, "turn_on")
        checks["target"] = d1_planner._covered(prompt, expected.get("target"))
        return checks
    relation = expected.get("relation")
    checks["source"] = d1_planner._covered(prompt, expected.get("source"))
    checks["target"] = d1_planner._covered(
        prompt, expected.get("target"), drawer=relation == "in"
    )
    # push_object encodes the destination in the target region; the official
    # LIBERO instruction ("push the plate to the front of the stove") contains
    # no lexical "on", so demanding one would mark a correct plan wrong.
    if expected.get("operation") != "push_object":
        checks["relation"] = d1_planner._relation_in_instruction(prompt, relation or "on")
    if operation == "composite":
        checks["open"] = "open" in words
        level = d1_planner._drawer_parts(expected.get("target") or "")[1]
        if level:
            checks["level"] = level in words
    return checks


# ---------------------------------------------------------------------------
# result extraction
# ---------------------------------------------------------------------------


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stage1_s": evidence.get("stage1_s"),
        "stage2_s": evidence.get("stage2_s"),
        "thresholds": evidence.get("thresholds"),
    }
    stage1 = evidence.get("stage1") or {}
    out["stage1"] = {key: stage1.get(key) for key in (
        "supported",
        "operation",
        "operation_confidence",
        "source",
        "source_confidence",
        "source_specified",
        "ambiguous",
    )}
    stage2 = evidence.get("stage2") or {}
    out["stage2"] = stage2.get("answers") if isinstance(stage2, dict) else None
    return out


def _decision_result(
    result: dict[str, Any] | None,
    *,
    case: dict[str, Any],
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    engine_label: str,
    raw: bool,
    error: str | None = None,
) -> dict[str, Any]:
    if result is None:
        return {"engine": engine_label, "error": error or "no result"}
    status = result.get("status")
    accepted = status == "accepted"
    decision = result.get("decision") or {}
    got = _binding(result) if accepted else {}
    checks = _binding_checks(got, case.get("expected") or {}) if accepted else {}
    evidence = result.get("evidence") or {}
    contract_ok = False
    contract_errors: list[dict[str, Any]] = []
    if accepted:
        contract_ok, contract_errors = _contract_ok(
            result.get("plan"),
            case=case,
            task=task,
            snapshot=snapshot,
            skills=skills,
            plan_source=str(decision.get("plan_source") or engine_label),
        )
    row: dict[str, Any] = {
        "engine": engine_label,
        "status": status,
        "reason_code": result.get("reason_code"),
        "reason": result.get("reason"),
        "accepted": accepted,
        "decision": {
            "operation": got.get("operation"),
            "operation_confidence": decision.get("operation_confidence"),
            "operation_margin": decision.get("operation_margin"),
            "operation_source": decision.get("operation_source"),
            "source": got.get("source"),
            "source_confidence": decision.get("source_confidence"),
            "source_margin": decision.get("source_margin"),
            "target": got.get("target"),
            "target_confidence": decision.get("target_confidence"),
            "target_margin": decision.get("target_margin"),
            "relation": got.get("relation"),
            "relation_confidence": decision.get("relation_confidence"),
            "relation_margin": decision.get("relation_margin"),
            "relation_source": decision.get("relation_source"),
            "open_target": got.get("open_target"),
            "open_confidence": decision.get("open_confidence"),
            "open_margin": decision.get("open_margin"),
            "object_margin": decision.get("object_margin"),
        },
        "checks": checks,
        "full_binding": all(checks.values()) if checks else None,
        "contract_ok": contract_ok,
        "contract_error_count": len(contract_errors),
        "prompt": decision.get("prompt"),
        "stage1_s": decision.get("stage1_s") or evidence.get("stage1_s"),
        "stage2_s": decision.get("stage2_s") or evidence.get("stage2_s"),
        "total_s": result.get("total_s") or decision.get("total_s") or (
            round(float(evidence.get("stage1_s") or 0) + float(evidence.get("stage2_s") or 0), 4)
        ),
        "tokens": decision.get("usage") or {},
        "evidence": _compact_evidence(evidence),
    }
    if raw:
        row["evidence_full"] = evidence
        row["plan"] = result.get("plan")
    return row


def _qwen_full_result(
    result: dict[str, Any] | None,
    *,
    case: dict[str, Any],
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    raw: bool,
    error: str | None = None,
) -> dict[str, Any]:
    if result is None:
        return {"engine": "qwen_full", "error": error or "no result"}
    plan = result.get("plan")
    prompt = None
    if plan and (plan.get("steps") or []):
        prompt = (plan["steps"][0] or {}).get("prompt")
    contract_ok = False
    contract_errors: list[dict[str, Any]] = []
    if plan:
        contract_ok, contract_errors = _contract_ok(
            plan,
            case=case,
            task=task,
            snapshot=snapshot,
            skills=skills,
            plan_source="qwen3.5-9b",
        )
    expected = case.get("expected") or {}
    semantic = None
    semantic_checks: dict[str, bool] = {}
    if case.get("kind") == "positive" and prompt:
        semantic_checks = _prompt_covers_goal(prompt, expected)
        semantic = all(semantic_checks.values())
    attempts = result.get("attempts") or []
    row: dict[str, Any] = {
        "engine": "qwen_full",
        "status": result.get("status"),
        "repairs": result.get("repairs"),
        "attempt_count": len(attempts),
        "total_s": result.get("total_s"),
        "prompt": prompt,
        "contract_ok": contract_ok,
        "contract_error_count": len(contract_errors),
        "semantic": semantic,
        "semantic_checks": semantic_checks,
        "prompt_equals_official": _same_text(prompt, task.get("instruction")),
        "prompt_equals_user": _same_text(prompt, case.get("instruction")),
        "tokens": {
            "prompt": sum(int(((row.get("timing") or {}).get("prompt_tokens") or 0)) for row in attempts),
            "completion": sum(int(((row.get("timing") or {}).get("completion_tokens") or 0)) for row in attempts),
        },
        "vllm": {
            "queue_s": (attempts[-1].get("timing") or {}).get("vllm_queue_s") if attempts else None,
            "ttft_s": (attempts[-1].get("timing") or {}).get("ttft_s") if attempts else None,
            "decode_s": (attempts[-1].get("timing") or {}).get("decode_s") if attempts else None,
            "server_e2e_s": (attempts[-1].get("timing") or {}).get("server_e2e_s") if attempts else None,
        },
        "attempts": [
            {
                "kind": row.get("kind"),
                "elapsed_s": (row.get("timing") or {}).get("elapsed_s"),
                "prompt_tokens": (row.get("timing") or {}).get("prompt_tokens"),
                "completion_tokens": (row.get("timing") or {}).get("completion_tokens"),
            }
            for row in attempts
        ],
    }
    if raw:
        row["raw_plan"] = plan
    return row


def _same_text(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    normalize = lambda text: re.sub(r"\s+", " ", text.strip().lower())  # noqa: E731
    return normalize(left) == normalize(right)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _latency(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    if not clean:
        return {"n": 0}
    clean.sort()
    p95 = clean[min(len(clean) - 1, max(0, int(round(0.95 * (len(clean) - 1)))))]
    return {
        "n": len(clean),
        "median_s": round(statistics.median(clean), 4),
        "mean_s": round(statistics.fmean(clean), 4),
        "p95_s": round(p95, 4),
        "max_s": round(clean[-1], 4),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _summarize_decisions(rows: list[dict[str, Any]], positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    positive_by_id = {case["id"]: case for case in positives}
    pos_rows = [row for row in rows if row.get("kind_id") in positive_by_id]
    neg_by_id = {row.get("kind_id"): row for row in rows if row.get("kind_id") not in positive_by_id}
    neg_rows = [neg_by_id[case["id"]] for case in negatives if case["id"] in neg_by_id]
    expected_by_id = {case["id"]: case for case in negatives}
    accepted = [row for row in pos_rows if row.get("accepted")]
    operation = sum(1 for row in pos_rows if row.get("checks", {}).get("operation"))
    # A null expected source means "no source must be bound"; those cases are
    # checked inside full_binding but are not part of source-selection accuracy.
    source_rows = [
        row for row in pos_rows
        if (positive_by_id[row["kind_id"]].get("expected") or {}).get("source")
    ]
    source = sum(1 for row in source_rows if row.get("checks", {}).get("source"))
    target = sum(1 for row in pos_rows if row.get("checks", {}).get("target"))
    relation = sum(1 for row in pos_rows if row.get("checks", {}).get("relation"))
    open_rows = [
        row for row in pos_rows
        if "open_target" in (positive_by_id[row["kind_id"]].get("expected") or {})
    ]
    open_target = sum(1 for row in open_rows if row.get("checks", {}).get("open_target"))
    full = sum(1 for row in pos_rows if row.get("full_binding"))
    contract = sum(1 for row in accepted if row.get("contract_ok"))
    reject_correct = sum(
        1
        for row in neg_rows
        if row.get("status") != "accepted"
        and _reject_group(row.get("status"), row.get("reason_code"))
        == expected_by_id[row["kind_id"]].get("expected_reject")
    )
    reject_any = sum(1 for row in neg_rows if row.get("status") != "accepted")
    false_accept = len(neg_rows) - reject_any
    source_cases = sum(1 for case in positives if case.get("expected", {}).get("source"))
    open_cases = sum(1 for case in positives if "open_target" in case.get("expected", {}))
    warm = [row["total_s"] for row in rows if row.get("phase") == "warm"]
    first = [row["total_s"] for row in rows if row.get("phase") == "first_in_task"]
    return {
        "positives": len(pos_rows),
        "negatives": len(neg_rows),
        "accepted": len(accepted),
        "rejected": len(pos_rows) - len(accepted),
        "false_reject": len(pos_rows) - len(accepted),
        "operation_accuracy": _rate(operation, len(pos_rows)),
        "operation_accuracy_accepted": _rate(operation, len(accepted)),
        "source_accuracy": _rate(source, len(source_rows)),
        "source_cases": len(source_rows),
        "target_accuracy": _rate(target, len(pos_rows)),
        "relation_accuracy": _rate(relation, len(pos_rows)),
        "open_target_accuracy": _rate(open_target, len(open_rows)),
        "open_target_cases": len(open_rows),
        "full_binding_accuracy": _rate(full, len(pos_rows)),
        "full_binding_accuracy_accepted": _rate(full, len(accepted)),
        "contract_pass_rate": _rate(contract, len(accepted)),
        "contract_pass_rate_all": _rate(contract, len(pos_rows)),
        "reject_accuracy": _rate(reject_correct, len(neg_rows)),
        "reject_any_rate": _rate(reject_any, len(neg_rows)),
        "false_accept": false_accept,
        "latency_all": _latency([row["total_s"] for row in rows]),
        "latency_warm": _latency(warm),
        "latency_first_in_task": _latency(first),
    }


def _summarize_qwen_full(rows: list[dict[str, Any]], positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    pos_rows = [row for row in rows if row.get("kind_id") in {case["id"] for case in positives}]
    neg_rows = [row for row in rows if row.get("kind_id") in {case["id"] for case in negatives}]
    ok = sum(1 for row in rows if row.get("status") == "ok")
    contract = sum(1 for row in rows if row.get("contract_ok"))
    semantic = sum(1 for row in pos_rows if row.get("semantic"))
    official = sum(1 for row in pos_rows if row.get("prompt_equals_official"))
    user = sum(1 for row in pos_rows if row.get("prompt_equals_user"))
    warm = [row["total_s"] for row in rows if row.get("phase") == "warm"]
    first = [row["total_s"] for row in rows if row.get("phase") == "first_in_task"]
    return {
        "positives": len(pos_rows),
        "negatives": len(neg_rows),
        "plan_ok_rate": _rate(ok, len(rows)),
        "contract_pass_rate": _rate(contract, len(rows)),
        "semantic_accuracy": _rate(semantic, len(pos_rows)),
        "prompt_equals_official_rate": _rate(official, len(pos_rows)),
        "prompt_equals_user_rate": _rate(user, len(pos_rows)),
        "negative_rejections": sum(1 for row in neg_rows if row.get("status") != "ok"),
        "binding_accuracy": None,
        "binding_note": "whole_task emits no object/site binding; binding accuracy is not applicable",
        "latency_all": _latency([row["total_s"] for row in rows]),
        "latency_warm": _latency(warm),
        "latency_first_in_task": _latency(first),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="config/offline_eval.json")
    parser.add_argument("--out", default="records/offline/eval.json")
    parser.add_argument("--tasks", default="", help="comma-separated task ids to run (default: from cases)")
    parser.add_argument("--engines", default="qwen_full,d1,qwen_decision")
    parser.add_argument("--thresholds", default="", help="JSON overrides for d1 thresholds")
    parser.add_argument("--raw", action="store_true", help="keep full d1 evidence and raw plans")
    parser.add_argument("--dump-snapshots", default="", help="write compact snapshots to this directory")
    parser.add_argument("--snapshot-only", action="store_true", help="load and dump scenes, run no cases")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    cfg = config.engine_config()
    suite = cfg["env"]["suite"]
    seed = cfg["env"]["seed"]
    d1_cfg = cfg["agent"]["d1"]
    thresholds = {**(d1_cfg.get("thresholds") or {})}
    if args.thresholds:
        thresholds.update(json.loads(args.thresholds))
    engines = {name.strip() for name in args.engines.split(",") if name.strip()}

    case_set = json.loads((PROJECT / args.set).read_text(encoding="utf-8"))
    cases = case_set["cases"]
    task_ids = sorted({int(case["task_id"]) for case in cases})
    if args.tasks:
        wanted = {int(token) for token in args.tasks.split(",") if token.strip()}
        task_ids = [task_id for task_id in task_ids if task_id in wanted]
        cases = [case for case in cases if int(case["task_id"]) in wanted]
    skills = config.skill_config()
    # The OpenWAM backend exposes exactly one skill. Passing the full catalog
    # would make whole_task_catalog() false and select the sub-skill system
    # prompt, which is not the production planner; task 3 then decomposes and
    # loses the "put the bowl inside" subgoal. Evaluate the same contract the
    # engine sends to the Agent.
    skills = {
        **skills,
        "skills": [row for row in (skills.get("skills") or []) if row.get("name") == "whole_task"],
    }
    env_client = EnvClient(config.service_url("env"))
    engine_url = config.service_url("engine")
    snapshot_dir = Path(args.dump_snapshots) if args.dump_snapshots else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows: list[dict[str, Any]] = []
    scenes: dict[int, dict[str, Any]] = {}
    load_s: dict[int, float] = {}

    for task_id in task_ids:
        task = config.task_for(task_id)
        print(f"[scene] loading task {task_id}", flush=True)
        load_started = time.perf_counter()
        loaded = _post_json(f"{engine_url}/api/load", {"task_id": task_id})
        load_s[task_id] = round(time.perf_counter() - load_started, 3)
        if loaded.get("status") != "ok":
            raise RuntimeError(f"engine load task {task_id} failed: {loaded}")
        snapshot = env_client.state_snapshot()
        scenes[task_id] = snapshot
        if snapshot_dir:
            compact = {
                "task_id": task_id,
                "suite": suite,
                "seed": seed,
                "objects": sorted(((snapshot.get("sim") or {}).get("objects") or {}).keys()),
                "sites": sorted(((snapshot.get("named_poses") or {}).get("sites") or {}).keys()),
                "site_poses": {
                    name: pose
                    for name, pose in (((snapshot.get("named_poses") or {}).get("sites") or {}).items())
                },
                "joints": snapshot.get("joints"),
                "native_goal": snapshot.get("native_goal"),
                "scene_epoch": snapshot.get("scene_epoch"),
                "observation_id": snapshot.get("observation_id"),
            }
            (snapshot_dir / f"task{task_id}.json").write_text(
                json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if args.snapshot_only:
            continue
        task_cases = [case for case in cases if int(case["task_id"]) == task_id]
        for index, case in enumerate(task_cases):
            phase = "first_in_task" if index == 0 else "warm"
            print(f"  [{case['id']}] {case['instruction']}", flush=True)
            row: dict[str, Any] = {
                "id": case["id"],
                "kind_id": case["id"],
                "task_id": task_id,
                "kind": case["kind"],
                "instruction": case["instruction"],
                "expected": case.get("expected"),
                "expected_reject": case.get("expected_reject"),
                "phase": phase,
                "results": {},
            }
            if "qwen_full" in engines:
                try:
                    result = planner.request_plan(
                        instruction=case["instruction"],
                        task=task,
                        snapshot=snapshot,
                        skills=skills,
                    )
                    row["results"]["qwen_full"] = _qwen_full_result(
                        result, case=case, task=task, snapshot=snapshot, skills=skills, raw=args.raw
                    )
                except Exception as exc:
                    row["results"]["qwen_full"] = _qwen_full_result(
                        None, case=case, task=task, snapshot=snapshot, skills=skills, raw=args.raw,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            if "d1" in engines:
                try:
                    result = d1_planner.plan_with_d1(
                        instruction=case["instruction"],
                        snapshot=snapshot,
                        skills=skills,
                        d1_url=d1_cfg["url"],
                        task_id=task_id,
                        thresholds=thresholds,
                        timeout_s=float(d1_cfg.get("timeout_s") or 30.0),
                    )
                    row["results"]["d1"] = _decision_result(
                        result, case=case, task=task, snapshot=snapshot, skills=skills,
                        engine_label="d1", raw=args.raw,
                    )
                except Exception as exc:
                    row["results"]["d1"] = {"engine": "d1", "error": f"{type(exc).__name__}: {exc}"}
            if "qwen_decision" in engines:
                client = QwenDecisionClient()
                try:
                    result = d1_planner.plan_with_d1(
                        instruction=case["instruction"],
                        snapshot=snapshot,
                        skills=skills,
                        d1_url=d1_cfg["url"],
                        task_id=task_id,
                        thresholds=thresholds,
                        timeout_s=float(d1_cfg.get("timeout_s") or 30.0),
                        client=client,
                    )
                    decision_row = _decision_result(
                        result, case=case, task=task, snapshot=snapshot, skills=skills,
                        engine_label="qwen_decision", raw=args.raw,
                    )
                    decision_row["qwen_calls"] = [
                        {
                            "elapsed_s": call.get("elapsed_s"),
                            "prompt_tokens": call.get("prompt_tokens"),
                            "completion_tokens": call.get("completion_tokens"),
                            "attempts": len(call.get("attempts") or []),
                        }
                        for call in client.calls
                    ]
                    row["results"]["qwen_decision"] = decision_row
                except Exception as exc:
                    row["results"]["qwen_decision"] = {
                        "engine": "qwen_decision",
                        "error": f"{type(exc).__name__}: {exc}",
                        "qwen_calls": [
                            {
                                "elapsed_s": call.get("elapsed_s"),
                                "prompt_tokens": call.get("prompt_tokens"),
                                "completion_tokens": call.get("completion_tokens"),
                                "attempts": len(call.get("attempts") or []),
                            }
                            for call in client.calls
                        ],
                    }
            rows.append(row)
            if args.sleep:
                time.sleep(args.sleep)

    positives = [case for case in cases if case.get("kind") == "positive"]
    negatives = [case for case in cases if case.get("kind") == "negative"]
    attempted = {engine for row in rows for engine in row["results"]}
    summary: dict[str, Any] = {}
    for engine in ("qwen_full", "d1", "qwen_decision"):
        if engine not in attempted:
            summary[engine] = {"errors": 0, "not_run": True}
            continue
        engine_rows = [
            {**row["results"][engine], "kind_id": row["id"], "phase": row["phase"]}
            for row in rows
            if engine in row["results"] and "error" not in row["results"][engine]
        ]
        if engine == "qwen_full":
            summary[engine] = _summarize_qwen_full(engine_rows, positives, negatives)
        else:
            summary[engine] = _summarize_decisions(engine_rows, positives, negatives)
        errors = sum(1 for row in rows if engine in row["results"] and "error" in row["results"][engine])
        summary[engine]["errors"] = errors

    per_task: dict[str, Any] = {}
    for task_id in task_ids:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        positive_rows = [row for row in task_rows if row["kind"] == "positive"]
        uses_qwen_full = any("qwen_full" in row["results"] for row in task_rows)
        uses_d1 = any("d1" in row["results"] for row in task_rows)
        uses_qwen_decision = any("qwen_decision" in row["results"] for row in task_rows)
        per_task[str(task_id)] = {
            "cases": len(task_rows),
            "positives": len(positive_rows),
            "qwen_full_semantic": _rate(
                sum(1 for row in positive_rows if (row["results"].get("qwen_full") or {}).get("semantic")),
                len(positive_rows),
            ) if uses_qwen_full else None,
            "d1_full_binding": _rate(
                sum(1 for row in positive_rows if (row["results"].get("d1") or {}).get("full_binding")),
                len(positive_rows),
            ) if uses_d1 else None,
            "d1_accepts": sum(1 for row in task_rows if (row["results"].get("d1") or {}).get("accepted")),
            "qwen_decision_full_binding": _rate(
                sum(1 for row in positive_rows if (row["results"].get("qwen_decision") or {}).get("full_binding")),
                len(positive_rows),
            ) if uses_qwen_decision else None,
        }

    output = {
        "schema": "embodied_engine.offline_eval_result.v1",
        "set": args.set,
        "set_frozen": bool(case_set.get("frozen")),
        "started_at": started_at,
        "suite": suite,
        "seed": seed,
        "scene_load_s": load_s,
        "engines": sorted(engines),
        "models": {
            "qwen_full": {
                "model": cfg["agent"]["llm_model"],
                "base_url": cfg["agent"]["llm_base_url"],
                "temperature": cfg["agent"]["temperature"],
                "max_tokens": cfg["agent"]["max_tokens"],
            },
            "d1": {
                "model": d1_cfg["model"],
                "model_path": d1_cfg["model_path"],
                "model_sha256": d1_cfg["model_sha256"],
                "url": d1_cfg["url"],
                "output_tokens_per_decision": 0,
            },
            "qwen_decision": {
                "model": cfg["agent"]["llm_model"],
                "interface": "same staged choice/noul questions as d1, parsed into d1's answer schema",
            },
        },
        "thresholds": thresholds,
        "skills": [row.get("name") for row in (skills.get("skills") or [])],
        "summary": summary,
        "per_task": per_task,
        "cases": rows,
    }
    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary": summary, "per_task": per_task}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
