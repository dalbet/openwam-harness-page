"""Independent re-seat: can the policy re-grasp an object that is *already* placed?

Why this is a separate capability
---------------------------------
``put_cream_cheese_in_bowl`` completes on the official ``In`` relation, and that
relation is **already true** when the object sits on the rim and has to be
re-seated. Re-running the same catalog capability therefore asks a question
whose answer is known: the pre-check refuses it, and the record must say
``unsupported`` instead of counting a refused rollout as a repair. This module
defines the *different* capability the experiment actually needs:

    re-grasp -> lift clear of the rim -> move -> release -> settle in the bowl

It has its own instruction, its own stage machine and its own budget. It never
calls the original placement predicate a success and never treats the
execution layer's release (``finalize``) as a re-placement: every action must
come from the policy, and if the policy cannot do it the verdict is
``no_progress`` or ``unsupported``.

Geometry vocabulary (measured, not assumed)
-------------------------------------------
All numbers come from the same evaluator the step goals use
(``completion.goal_state`` -> ``relation``/``height``), i.e. from
``akita_black_bowl.xml``'s own ``horizontal_radius_site`` and interior floor:

* ``floor_z``   — the bowl's published interior floor (asset solid bottom + floor
  offset, in world coordinates at the current pose);
* ``rim_z``     — the top of the bowl's collision rim;
* ``bottom_z``  — the object's bottom as the relation evaluator sees it;
* ``seat``      — ``bottom_z - floor_z`` (the readiness module's seat depth);
* ``xy_m``      — object centre to container centre, horizontal.

Stage criteria (candidates, frozen for the round — see the note on each):

* ``lift``   — the object bottom rises until it is ``clearance_above_rim_m``
  (default 0.02 m) above ``rim_z``. This is a *relative* rise from the state the
  attempt started in, so a frame offset shared by both readings cancels out.
* ``carry``  — while lifted, the object's horizontal position moves at least
  ``min_offset_m`` from where the attempt started: the policy transported it,
  it did not just hover in place.
* ``release``— the fingers open and the finger contacts no longer name the
  object for ``confirm_samples`` consecutive samples, at a horizontal distance
  ``<= max_xy_m`` from the bowl centre and no more than ``drop_below_rim_m``
  below the rim, i.e. released *over the bowl*, not beside it.
* ``settle`` — after the release the official relation holds again, the seat is
  inside ``[seat_min_m, seat_max_m]``, ``xy_m <= max_xy_m``, the object is not
  touching the fingers, and the last ``samples`` readings drift less than
  ``max_drift_m``.

``[0.006, 0.029]`` is a *candidate* window: it brackets the archived seat
distribution of rounds that actually carried the cheese (0.0102..0.0263) and
stops at the readiness threshold above which a seat is called perched
(0.029 m). It is not evidence that a re-seat reaches it; the record reports the
measured seat whatever it turns out to be. The window is frozen for the round
and must not be moved after seeing results.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "embodied_engine.composite.independent_reseat.v1"

STAGE_ORDER = ("grasp", "lift", "carry", "release", "settle")

DEFAULTS: dict[str, Any] = {
    # Off by default: this is an experiment, and it must never run in a
    # delivery round unless the config copy turns it on explicitly.
    "enabled": False,
    # Its own instruction. It is *not* the catalog prompt: the policy is asked
    # to move an object that is already in the bowl.
    "prompt": "pick up the cream cheese and put it back in the bowl",
    "skill": "pick_place",
    # Step ref used for the per-request noise seed ("rr"), distinct from "c1"
    # so a re-seat chunk can never reuse the step's own noise stream.
    "step_ref": "rr",
    "chunk_budget": 48,
    "max_actions": 240,
    "timeout_s": 180.0,
    "grasp": {
        # A closed gripper that is also *at* the object. 0.05 m is the opening
        # the existing skill runner uses to call a gripper closed; 0.06 m is the
        # finger-frame reach of the object.
        "max_opening_m": 0.05,
        "max_eef_object_m": 0.06,
    },
    # The container must stay put while a stage claims the cheese was picked
    # out of it. Two numbers, both measured rather than assumed:
    # ``max_move_m`` is the horizontal drift of a free body resting on the
    # table, read from archived snapshots of a bowl nobody touched (p99
    # per-sample change over the whole step), times a safety factor of 3;
    # ``max_lift_m`` is the same quantity on the vertical axis. They are not a
    # universal 5 mm: they are this asset, this timestep, this noise.
    "container_guard": {
        "stages": ["grasp", "lift", "settle"],
        "max_move_m": 0.004,
        "max_lift_m": 0.004,
    },
    "lift": {
        # "clear of the rim by 2 cm", in the container's own frame: the object
        # bottom has to be 2 cm above the *current* rim plane. A bowl lifted
        # together with the cheese inside it does not change this number, so it
        # cannot be mistaken for a lift. ``clearance_above_rim_m`` is kept as
        # the name the previous round used; the criterion is the same number
        # applied to a relative measurement.
        "clearance_above_rim_m": 0.02,
    },
    "carry": {
        # Horizontal displacement from the attempt's start position. The bowl's
        # opening is ~0.05 m across, so 0.01 m is a move, not jitter.
        "min_offset_m": 0.01,
    },
    "release": {
        "min_opening_m": 0.04,
        "confirm_samples": 2,
        "max_xy_m": 0.02,
        "drop_below_rim_m": 0.02,
    },
    "settle": {
        "samples": 3,
        "max_xy_m": 0.015,
        "seat_min_m": 0.006,
        "seat_max_m": 0.029,
        "max_drift_m": 0.002,
    },
    # How often a compact sample row is kept for diagnosis (the full row set is
    # reduced to this trace plus peak/end numbers in the record).
    "trace_every_samples": 5,
    # Only these catalog steps are re-seated. An explicit whitelist keeps the
    # experiment from silently touching "put the bowl on the stove" (a
    # container-to-region placement has no re-grasp meaning).
    "step_task_keys": ["put_cream_cheese_in_bowl"],
    # What the engine does with the verdict: ``report`` never changes the
    # episode's own outcome; ``stop`` ends it with an explicit failure code.
    "on_result": "report",
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def options(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Effective options; config overrides defaults, never the reverse."""
    block = ((cfg or {}).get("composite") or {}).get("independent_reseat") or {}
    return _merge(DEFAULTS, block)


def candidate_step(step: dict[str, Any]) -> bool:
    """Only a placement step whose own goal put an object into its target.

    The step's *execution* skill is ``whole_task`` on the pi05 backend; the
    placement binding lives in ``completion`` (object, target, relation), which
    is what makes a re-seat meaningful.
    """
    completion = step.get("completion") or {}
    if str(completion.get("kind") or "") != "relation":
        return False
    return bool(completion.get("object_n")) and bool(completion.get("target_n"))


def measure(row: dict[str, Any]) -> dict[str, Any]:
    """One reading in the numbers the stage rules use.

    ``row`` is what the engine reads from the live snapshot; every field is
    optional and a missing one stays ``None`` (never a guessed zero).
    """
    bottom = row.get("object_bottom_z")
    floor = row.get("floor_z")
    rim = row.get("rim_z")
    seat = None
    if isinstance(bottom, (int, float)) and isinstance(floor, (int, float)):
        seat = round(float(bottom) - float(floor), 5)
    above_rim = None
    if isinstance(bottom, (int, float)) and isinstance(rim, (int, float)):
        above_rim = round(float(bottom) - float(rim), 5)
    contacts = row.get("contacts")
    if contacts is None:
        # Older callers pass only the finger pairs; the split is the same.
        contacts = row.get("gripper_contacts")
    available = row.get("contacts_available")
    if available is None:
        available = isinstance(contacts, list)
    split = _split_contacts(contacts)
    return {
        "sim_action_count": row.get("sim_action_count"),
        "observation_id": row.get("observation_id"),
        "object_pos": row.get("object_pos"),
        "target_pos": row.get("target_pose"),
        "container_pos": row.get("container_pos"),
        "container_bottom_z": row.get("container_bottom_z"),
        "eef_pos": row.get("eef_pos"),
        "opening_m": row.get("opening_m"),
        "eef_object_m": row.get("eef_object_m"),
        # Tri-state on purpose: ``True`` the fingers touch the object,
        # ``False`` the contact list was readable and does not name it,
        # ``None`` no contact list was available at all. Collapsing the last
        # two into "no contact" is what made a missing probe look like a
        # measured negative.
        "contacts_available": bool(available),
        "gripper_object_contact": split["gripper_object"],
        "gripper_container_contact": split["gripper_container"],
        "object_container_contact": split["object_container"],
        "gripper_contacts": row.get("gripper_contacts"),
        "relation_ok": row.get("relation_ok"),
        "xy_m": row.get("xy_m"),
        "object_bottom_z": bottom,
        "floor_z": floor,
        "rim_z": rim,
        "seat_m": seat,
        "above_rim_m": above_rim,
    }


def _split_contacts(contacts: Any) -> dict[str, bool | None]:
    """Separate the three contact questions a re-seat has to tell apart.

    ``None`` (no list at all) is *unavailable evidence*, not "nothing is
    touching"; the stage rules treat it as a refusal instead of a pass.

    * ``gripper_object``    — a finger body touches the carried object;
    * ``gripper_container`` — a finger body touches the container (the policy
      grabbed the bowl, so any object motion is passive);
    * ``object_container``  — the object touches the container (the object is
      being supported by the bowl, which is the state a re-seat starts from).
    """
    if not isinstance(contacts, (list, tuple)):
        return {"gripper_object": None, "gripper_container": None, "object_container": None}
    gripper_object = False
    gripper_container = False
    object_container = False
    for row in contacts:
        if not isinstance(row, dict):
            continue
        sides = [str(row.get("body1") or ""), str(row.get("body2") or "")]
        finger = bool(row.get("gripper1") or row.get("gripper2")) or any(
            "finger" in name.lower() or "gripper" in name.lower() for name in sides
        )
        object_side = any(
            "cream" in name.lower() or "cheese" in name.lower() or name.lower().startswith("object")
            for name in sides
        )
        container_side = any("bowl" in name.lower() for name in sides)
        if finger and object_side:
            gripper_object = True
        if finger and container_side:
            gripper_container = True
        if object_side and container_side:
            object_container = True
    return {
        "gripper_object": gripper_object,
        "gripper_container": gripper_container,
        "object_container": object_container,
    }


def trigger(reading: dict[str, Any]) -> dict[str, Any]:
    """Is the start state "the object is already in the container"?

    This is a *precondition*, not a success criterion: it says the capability
    has something to act on. When it is false the attempt is
    ``unsupported`` — there is no placed object to re-seat.
    """
    relation_ok = bool(reading.get("relation_ok"))
    bottom = reading.get("object_bottom_z")
    floor = reading.get("floor_z")
    xy = reading.get("xy_m")
    in_container = None
    if isinstance(bottom, (int, float)) and isinstance(floor, (int, float)):
        in_container = float(bottom) >= float(floor) - 0.002
    inside_xy = None
    if isinstance(xy, (int, float)):
        inside_xy = float(xy) <= 0.03
    ok = bool(relation_ok and in_container is not False and inside_xy is not False)
    return {
        "ok": ok,
        "relation_ok": relation_ok,
        "bottom_above_floor": in_container,
        "xy_within_opening": inside_xy,
        "detail": (
            "object already in the container: the original placement predicate holds, "
            "so this attempt is a re-seat, not a re-run of the placement capability"
        ),
    }


def _object_contact(reading: dict[str, Any]) -> bool | None:
    """Finger-to-object contact from the older ``gripper_contacts`` rows.

    Kept for the ``contact_samples`` counter; the stage rules use the
    tri-state field ``gripper_object_contact`` produced by :func:`measure`,
    which separates "unavailable" from "no contact".
    """
    return _split_contacts(reading.get("gripper_contacts"))["gripper_object"]


class Tracker:
    """Stage machine over one re-seat attempt.

    Pure bookkeeping: it reads measurements and decides which stage has been
    reached. It never issues an action, never opens the gripper and never
    writes to the environment.
    """

    def __init__(self, *, start: dict[str, Any], opts: dict[str, Any]):
        self.opts = opts
        self.start = start
        self.rows: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.trace_every = max(1, int(opts.get("trace_every_samples") or 5))
        self.min_eef_object_m: float | None = None
        self.min_opening_m: float | None = None
        self.max_opening_m: float | None = None
        self.closed_samples = 0
        self.contact_samples = 0
        self.stages: dict[str, dict[str, Any]] = {
            name: {"achieved": False, "at_sim_action": None, "evidence": None}
            for name in STAGE_ORDER
        }
        self.not_measured: list[str] = []
        self.max_rise_m: float | None = None
        self.max_offset_m: float | None = None
        self.max_above_rim_m: float | None = None
        # Container motion is measured from this attempt's own first sample, so
        # it is a displacement, not an absolute pose.
        self.container_move_m: float | None = None
        self.container_lift_m: float | None = None
        self.container_moved_at: dict[str, Any] | None = None
        self.refusals: list[dict[str, Any]] = []
        self._release_streak = 0
        self._lifted = False
        self._released = False
        self._release_at: dict[str, Any] | None = None
        # Where the fingers first opened. The confirm window may finish while
        # the object is already falling, so the *onset* reading is the one that
        # has to have been over the bowl and at or above the rim.
        self._release_onset: dict[str, Any] | None = None

    # -- helpers ---------------------------------------------------------

    def _reach(self, stage: str, reading: dict[str, Any], evidence: dict[str, Any]) -> None:
        if self.stages[stage]["achieved"]:
            return
        self.stages[stage]["achieved"] = True
        self.stages[stage]["at_sim_action"] = reading.get("sim_action_count")
        self.stages[stage]["evidence"] = evidence

    def _xy_offset(self, reading: dict[str, Any]) -> float | None:
        start_pos = self.start.get("object_pos")
        pos = reading.get("object_pos")
        if not (isinstance(start_pos, (list, tuple)) and isinstance(pos, (list, tuple))):
            return None
        if len(start_pos) < 2 or len(pos) < 2:
            return None
        return round(math.hypot(float(pos[0]) - float(start_pos[0]), float(pos[1]) - float(start_pos[1])), 5)

    def _required_rise(self) -> float | None:
        bottom0 = self.start.get("object_bottom_z")
        rim = self.start.get("rim_z")
        if not (isinstance(bottom0, (int, float)) and isinstance(rim, (int, float))):
            return None
        return round((float(rim) - float(bottom0)) + float(self.opts["lift"]["clearance_above_rim_m"]), 5)

    # -- container guard -------------------------------------------------

    def _update_container(self, reading: dict[str, Any]) -> dict[str, Any]:
        """How far the container moved since this attempt started.

        Judged only against ``container_guard.stages``: during ``carry`` the
        arm is moving the object and may legitimately brush the bowl, but while
        ``grasp``/``lift``/``settle`` are being claimed the bowl has to be a
        fixture. A cheese that rose because the bowl rose is not a lift.
        """
        start_pos = self.start.get("container_pos")
        pos = reading.get("container_pos")
        move = lift = None
        if isinstance(start_pos, (list, tuple)) and isinstance(pos, (list, tuple)) and len(start_pos) >= 3 and len(pos) >= 3:
            move = round(
                math.hypot(float(pos[0]) - float(start_pos[0]), float(pos[1]) - float(start_pos[1])),
                5,
            )
            lift = round(float(pos[2]) - float(start_pos[2]), 5)
            self.container_move_m = move if self.container_move_m is None else max(self.container_move_m, move)
            self.container_lift_m = lift if self.container_lift_m is None else max(self.container_lift_m, lift)
        guard = self.opts["container_guard"]
        moved = (
            (move is not None and move > float(guard["max_move_m"]))
            or (lift is not None and lift > float(guard["max_lift_m"]))
        )
        return {
            "move_m": move,
            "lift_m": lift,
            "max_move_m": guard["max_move_m"],
            "max_lift_m": guard["max_lift_m"],
            "moved": moved,
        }

    def _refuse(self, stage: str, reading: dict[str, Any], reason: str, detail: dict[str, Any] | None = None) -> None:
        if any(item.get("stage") == stage and item.get("reason") == reason for item in self.refusals):
            return
        self.refusals.append(
            {
                "stage": stage,
                "reason": reason,
                "sim_action_count": reading.get("sim_action_count"),
                "detail": detail or {},
            }
        )
        if stage not in self.not_measured:
            self.not_measured.append(stage)

    # -- main entry ------------------------------------------------------

    def observe(self, reading: dict[str, Any]) -> dict[str, Any]:
        row = dict(reading)
        # Cheap diagnostics that answer "did the policy approach at all?" even
        # when no stage is reached: the closest the fingers came, and how far
        # the gripper actually closed.
        near = reading.get("eef_object_m")
        if isinstance(near, (int, float)):
            self.min_eef_object_m = float(near) if self.min_eef_object_m is None else min(self.min_eef_object_m, float(near))
        opening = reading.get("opening_m")
        if isinstance(opening, (int, float)):
            self.min_opening_m = float(opening) if self.min_opening_m is None else min(self.min_opening_m, float(opening))
            self.max_opening_m = float(opening) if self.max_opening_m is None else max(self.max_opening_m, float(opening))
            if float(opening) <= float(self.opts["grasp"]["max_opening_m"]):
                self.closed_samples += 1
        if _object_contact(reading) is True:
            self.contact_samples += 1
        if len(self.rows) % self.trace_every == 0:
            self.trace.append(
                {
                    "sim_action_count": reading.get("sim_action_count"),
                    "opening_m": round(float(opening), 4) if isinstance(opening, (int, float)) else None,
                    "eef_object_m": round(float(near), 4) if isinstance(near, (int, float)) else None,
                    "seat_m": reading.get("seat_m"),
                    "xy_m": round(float(reading["xy_m"]), 4) if isinstance(reading.get("xy_m"), (int, float)) else None,
                    "relation_ok": reading.get("relation_ok"),
                }
            )
        rise = None
        bottom0 = self.start.get("object_bottom_z")
        if isinstance(bottom0, (int, float)) and isinstance(reading.get("object_bottom_z"), (int, float)):
            rise = round(float(reading["object_bottom_z"]) - float(bottom0), 5)
            self.max_rise_m = rise if self.max_rise_m is None else max(self.max_rise_m, rise)
        if isinstance(reading.get("above_rim_m"), (int, float)):
            value = float(reading["above_rim_m"])
            self.max_above_rim_m = value if self.max_above_rim_m is None else max(self.max_above_rim_m, value)

        # 1. grasp ------------------------------------------------------
        grasp_opts = self.opts["grasp"]
        opening = reading.get("opening_m")
        near = reading.get("eef_object_m")
        contact = reading.get("gripper_object_contact")
        on_container = reading.get("gripper_container_contact")
        container = self._update_container(reading)
        row["container"] = container
        closed = isinstance(opening, (int, float)) and float(opening) <= float(grasp_opts["max_opening_m"])
        at_object = isinstance(near, (int, float)) and float(near) <= float(grasp_opts["max_eef_object_m"])
        if contact is None:
            # No contact evidence at all: say so instead of scoring it as a
            # miss or a hit. The stage cannot be reached on proximity alone.
            self._refuse("grasp", reading, "contact_evidence_unavailable")
        elif contact is False:
            self._refuse("grasp", reading, "no_gripper_object_contact", {"opening_m": opening, "eef_object_m": near})
        if on_container:
            self._refuse("grasp", reading, "gripper_on_container", {"opening_m": opening})
        if container["moved"] and "grasp" in self.opts["container_guard"]["stages"]:
            self._refuse("grasp", reading, "container_moved", container)
        grasp_ok = bool(
            closed
            and at_object
            and contact is True
            and not on_container
            and not (container["moved"] and "grasp" in self.opts["container_guard"]["stages"])
        )
        if grasp_ok:
            self._reach(
                "grasp",
                reading,
                {
                    "opening_m": opening,
                    "eef_object_m": near,
                    "contact": contact,
                    "container_move_m": container["move_m"],
                    "source": "finger_object_contact",
                },
            )

        # 2. lift clear of the rim --------------------------------------
        # The criterion is stated in the container's frame: the object bottom
        # has to be ``clearance_above_rim_m`` above the rim *now*. A bowl that
        # was picked up with the cheese still riding in it leaves this number
        # unchanged, so it cannot produce a lift.
        clearance = float(self.opts["lift"]["clearance_above_rim_m"])
        above = reading.get("above_rim_m")
        if above is None:
            self._refuse("lift", reading, "above_rim_unavailable")
        else:
            guard = self.opts["container_guard"]["stages"]
            if float(above) >= clearance and "lift" in guard and container["moved"]:
                self._refuse("lift", reading, "container_lifted_with_object", {**container, "above_rim_m": above})
            elif float(above) >= clearance:
                self._lifted = True
                self._reach(
                    "lift",
                    reading,
                    {
                        "above_rim_m": above,
                        "clearance_above_rim_m": clearance,
                        "rise_m": rise,
                        "container_move_m": container["move_m"],
                        "container_lift_m": container["lift_m"],
                    },
                )

        # 3. carried while lifted ---------------------------------------
        offset = self._xy_offset(reading)
        if offset is not None:
            self.max_offset_m = offset if self.max_offset_m is None else max(self.max_offset_m, offset)
        if self._lifted and offset is not None and offset >= float(self.opts["carry"]["min_offset_m"]):
            self._reach(
                "carry",
                reading,
                {"offset_m": offset, "min_offset_m": self.opts["carry"]["min_offset_m"]},
            )

        # 4. release over the bowl --------------------------------------
        release_opts = self.opts["release"]
        opened = isinstance(opening, (int, float)) and float(opening) >= float(release_opts["min_opening_m"])
        free = contact is False
        if opened and free:
            self._release_streak += 1
            if self._release_onset is None and self._lifted:
                self._release_onset = dict(reading)
        else:
            self._release_streak = 0
            self._release_onset = None
        onset = self._release_onset or {}
        xy = onset.get("xy_m")
        rim = onset.get("rim_z")
        bottom = onset.get("object_bottom_z")
        over_bowl = (
            isinstance(xy, (int, float)) and float(xy) <= float(release_opts["max_xy_m"])
        )
        at_rim = (
            isinstance(rim, (int, float))
            and isinstance(bottom, (int, float))
            and float(bottom) >= float(rim) - float(release_opts["drop_below_rim_m"])
        )
        if (
            self._lifted
            and self._release_streak >= int(release_opts["confirm_samples"])
            and over_bowl
            and at_rim
        ):
            self._released = True
            self._release_at = dict(onset)
            self._reach(
                "release",
                reading,
                {
                    "opening_m": onset.get("opening_m"),
                    "contact": onset.get("gripper_object_contact"),
                    "confirm_samples": self._release_streak,
                    "onset_sim_action_count": onset.get("sim_action_count"),
                    "onset_xy_m": onset.get("xy_m"),
                    "onset_above_rim_m": onset.get("above_rim_m"),
                },
            )

        # 5. settled in the bowl ---------------------------------------
        settle_opts = self.opts["settle"]
        row["stage_flags"] = {name: bool(self.stages[name]["achieved"]) for name in STAGE_ORDER}
        row["rise_m"] = rise
        if self._released:
            recent = [
                item
                for item in self.rows[-int(settle_opts["samples"]) :]
                if isinstance(item.get("object_bottom_z"), (int, float))
                and isinstance(item.get("object_pos"), (list, tuple))
            ] + [reading]
            window = recent[-int(settle_opts["samples"]) :]
            drift = None
            if len(window) >= 2:
                first, last = window[0], window[-1]
                drift = round(
                    math.dist(
                        [float(first["object_pos"][0]), float(first["object_pos"][1]), float(first["object_bottom_z"])],
                        [float(last["object_pos"][0]), float(last["object_pos"][1]), float(last["object_bottom_z"])],
                    ),
                    5,
                )
            seat = reading.get("seat_m")
            relation_ok = bool(reading.get("relation_ok"))
            inside_window = (
                isinstance(seat, (int, float))
                and float(settle_opts["seat_min_m"]) <= float(seat) <= float(settle_opts["seat_max_m"])
            )
            xy_ok = isinstance(xy, (int, float)) and float(xy) <= float(settle_opts["max_xy_m"])
            # "resting" is only claimed when the contact probe was readable and
            # reports no finger contact; an unavailable probe is not a rest.
            resting = contact is False
            stable = drift is not None and drift <= float(settle_opts["max_drift_m"])
            still_bowl = not container["moved"] or "settle" not in self.opts["container_guard"]["stages"]
            if len(window) >= int(settle_opts["samples"]) and relation_ok and inside_window and xy_ok and resting and stable and still_bowl:
                self._reach(
                    "settle",
                    reading,
                    {
                        "seat_m": seat,
                        "seat_window_m": [settle_opts["seat_min_m"], settle_opts["seat_max_m"]],
                        "xy_m": xy,
                        "max_xy_m": settle_opts["max_xy_m"],
                        "drift_m": drift,
                        "max_drift_m": settle_opts["max_drift_m"],
                        "relation_ok": relation_ok,
                        "contact": contact,
                        "samples": len(window),
                    },
                )
        row["release_streak"] = self._release_streak
        self.rows.append(row)
        return row

    # -- verdict ---------------------------------------------------------

    @property
    def done(self) -> bool:
        return bool(self.stages["settle"]["achieved"])

    def result(self) -> dict[str, Any]:
        achieved = [name for name in STAGE_ORDER if self.stages[name]["achieved"]]
        missing = [name for name in STAGE_ORDER if not self.stages[name]["achieved"]]
        if self.stages["settle"]["achieved"]:
            status = "completed"
        elif not achieved:
            status = "no_progress"
        else:
            status = "partial"
        last = self.rows[-1] if self.rows else {}
        return {
            "schema": SCHEMA,
            "status": status,
            "completed": bool(self.stages["settle"]["achieved"]),
            "stage_order": list(STAGE_ORDER),
            "stages": {name: dict(self.stages[name]) for name in STAGE_ORDER},
            "achieved": achieved,
            "missing": missing,
            "samples": len(self.rows),
            "start": {
                "object_pos": self.start.get("object_pos"),
                "object_bottom_z": self.start.get("object_bottom_z"),
                "seat_m": self.start.get("seat_m"),
                "xy_m": self.start.get("xy_m"),
                "rim_z": self.start.get("rim_z"),
                "container_pos": self.start.get("container_pos"),
                "opening_m": self.start.get("opening_m"),
                "above_rim_m": self.start.get("above_rim_m"),
                "contacts_available": self.start.get("contacts_available"),
            },
            "container": {
                "move_m": self.container_move_m,
                "lift_m": self.container_lift_m,
                "guard": dict(self.opts["container_guard"]),
                "note": (
                    "the guard applies only to the stages where the bowl must stay a fixture; "
                    "a container that moved is reported, never rounded down to zero"
                ),
            },
            "refusals": list(self.refusals),
            "evidence": {
                "contacts_available": bool(self.start.get("contacts_available")),
                "contact_samples": self.contact_samples,
                "closed_samples": self.closed_samples,
                "note": (
                    "an unavailable contact probe is refused, not counted as a measured absence"
                ),
            },
            "peak": {
                "rise_m": self.max_rise_m,
                "above_rim_m": self.max_above_rim_m,
                "offset_m": self.max_offset_m,
                "min_eef_object_m": None if self.min_eef_object_m is None else round(self.min_eef_object_m, 4),
                "min_opening_m": None if self.min_opening_m is None else round(self.min_opening_m, 4),
                "max_opening_m": None if self.max_opening_m is None else round(self.max_opening_m, 4),
                "closed_samples": self.closed_samples,
                "contact_samples": self.contact_samples,
            },
            "trace": list(self.trace),
            "end": {
                "object_pos": last.get("object_pos"),
                "object_bottom_z": last.get("object_bottom_z"),
                "seat_m": last.get("seat_m"),
                "xy_m": last.get("xy_m"),
                "relation_ok": last.get("relation_ok"),
                "opening_m": last.get("opening_m"),
                "gripper_contacts": last.get("gripper_contacts"),
                "sim_action_count": last.get("sim_action_count"),
            },
            "release_at": None
            if self._release_at is None
            else {
                "sim_action_count": self._release_at.get("sim_action_count"),
                "xy_m": self._release_at.get("xy_m"),
                "above_rim_m": self._release_at.get("above_rim_m"),
                "opening_m": self._release_at.get("opening_m"),
            },            "not_measured": list(self.not_measured),
        }


def summarise(block: dict[str, Any]) -> dict[str, Any]:
    """Small row for the top-level record: never more than the verdict."""
    if not isinstance(block, dict):
        return {}
    return {
        "status": block.get("status"),
        "completed": block.get("completed"),
        "achieved": block.get("achieved"),
        "missing": block.get("missing"),
        "policy_actions": block.get("policy_actions"),
        "finalizer_actions": block.get("finalizer_actions"),
        "trigger_ok": (block.get("trigger") or {}).get("ok"),
        "seat_after_m": ((block.get("end") or {}).get("seat_m")),
        "xy_after_m": ((block.get("end") or {}).get("xy_m")),
        "peak_rise_m": (block.get("peak") or {}).get("rise_m"),
        "peak_above_rim_m": (block.get("peak") or {}).get("above_rim_m"),
        "container_move_m": (block.get("container") or {}).get("move_m"),
        "container_lift_m": (block.get("container") or {}).get("lift_m"),
        "contacts_available": (block.get("evidence") or {}).get("contacts_available"),
        "refusals": [row.get("reason") for row in (block.get("refusals") or [])],
        "reason": block.get("reason"),
    }
