"""ScenarioRunner against FakeWorld + RulePipeline (no sim, no agents, no model)."""
import asyncio
import time

import pytest

from homedemo.bus import Bus
from homedemo.contracts import Event, Settings, T
from homedemo.scenarios import HERO_TAG, ScenarioRunner, load_home, load_scenarios
from homedemo.scenarios.steps import describe_step
from homedemo.scenarios.homeindex import HomeIndex

from .fakes import FakeWorld, RulePipeline

HOME = load_home()


def rig(speed: float = 50.0, *, pipeline: bool = True, delay: float = 0.0, **runner_kw):
    bus = Bus()
    world = FakeWorld(bus, HOME)
    pipe = RulePipeline(bus, HOME, delay=delay) if pipeline else None
    settings = Settings(speed=speed, decider_mode="fake")
    runner = ScenarioRunner(bus, world, settings, HOME, **runner_kw)
    return bus, world, pipe, runner


def scenario(*steps, sid="t1"):
    return {"id": sid, "title": "Test " + sid, "blurb": "b", "tags": ["test"], "steps": list(steps)}


def statuses(bus: Bus):
    return [e.data for e in bus.history if e.topic == T.SCENARIO]


SAY_LIGHTS = {"do": "say", "who": "alex", "text": "Can you turn on the lights?"}        # 0.87 -> allow
SAY_CANT_SEE = {"do": "say", "who": "alex", "text": "I can't see a thing in here."}      # 0.67 -> ask alex


# ------------------------------------------------------------------ the whole library
@pytest.mark.parametrize("sc", load_scenarios(), ids=lambda s: s["id"])
async def test_every_library_scenario_passes_on_the_fake_pipeline(sc):
    bus, world, pipe, runner = rig(speed=100.0)
    result = await runner.run(sc)
    failed = [c.text for c in result.checks if not c.ok]
    assert result.state == "finished" and not failed, (sc["id"], failed, result.error)
    assert result.passed == sum(1 for s in sc["steps"] if s["do"] == "expect") - _skipped_replies(sc, result)
    assert not bus.errors
    fin = [s for s in statuses(bus) if s["state"] == "finished"]
    assert fin and fin[0]["passed"] == result.passed and fin[0]["failed"] == 0


def _skipped_replies(sc, result):
    # a non-if_asked reply with no ask counts as a failed check; none expected here
    return 0


async def test_hero_scenarios_exercise_both_bands_and_reply_paths():
    """Across the heroes the fake pipeline both acts directly and asks first, and replies are consumed."""
    acted = asked = replied = 0
    for sc in load_scenarios():
        if HERO_TAG not in sc["tags"]:
            continue
        bus, world, pipe, runner = rig(speed=100.0)
        res = await runner.run(sc)
        assert res.ok, (sc["id"], [c.text for c in res.checks if not c.ok])
        acted += any("done" in c.text for c in res.checks)
        asked += any(": asked" in c.text for c in res.checks)
        replied += len(pipe.replies)
    assert acted >= 3 and asked >= 2 and replied >= 2


# ------------------------------------------------------------------ statuses and narration
async def test_status_stream_and_result():
    bus, world, pipe, runner = rig()
    sc = scenario(
        {"do": "note", "text": "Evening."},
        {"do": "set_env", "room": "living", "lux": 15, "wait": 0.5},
        SAY_LIGHTS,
        {"do": "expect", "what": "proposal", "by": "keyword", "action": "lights_on"},
        {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True}, "within": 2},
        {"do": "expect", "what": "verdict", "verdict": "allow", "action": "lights_on", "who": "alex"},
        {"do": "expect", "what": "outcome", "action": "lights_on", "room": "living", "accept": ["done"]},
    )
    res = await runner.run(sc)
    assert res.ok and res.passed == 4 and res.failed == 0 and res.state == "finished"
    assert [c.ok for c in res.checks] == [True] * 4
    st = statuses(bus)
    assert [s["state"] for s in st] == ["started", "step", "step", "step", "step", "expect_pass", "step", "expect_pass",
                                        "step", "expect_pass", "step", "expect_pass", "finished"]
    assert st[0] == {"scenario_id": "t1", "state": "started", "text": "Test t1", "passed": 0, "failed": 0}
    assert st[1] == {"scenario_id": "t1", "state": "step", "step_index": 0, "text": "Evening.", "do": "note",
                     "passed": 0, "failed": 0}
    assert st[2]["text"] == "Living room: 15 lux" and st[2]["do"] == "set_env"
    assert st[3]["text"] == 'Alex says "Can you turn on the lights?"'
    assert st[4]["text"] == "Expect keyword proposes lights_on within 5 s"
    assert st[5]["text"].startswith("keyword proposes lights_on within 5 s: keyword proposed lights_on (shadow")
    assert st[7]["text"].startswith("light_living reaches {on: True} within 2 s: light_living reached {on: True}")
    assert st[9]["text"] == "verdict allow for lights_on within 5 s: verdict allow for lights_on (confidence 0.87 >= 0.80)"
    assert st[11]["text"] == "lights_on in the Living room is done within 5 s: lights_on: done (on -> light_living)"
    assert st[-1] == {"scenario_id": "t1", "state": "finished", "text": "4 passed, 0 failed", "passed": 4, "failed": 0}
    assert all(e.source == "scenario" for e in bus.history if e.topic == T.SCENARIO)
    # the world saw only world steps, without the runner-only "wait" key, and was reset first
    assert [s["do"] for s in world.steps] == ["set_env", "say"]
    assert "wait" not in world.steps[0] and world.resets == 1
    assert runner.last_result is res and not runner.running and runner.current is None


def test_describe_step_texts():
    home = HomeIndex(HOME)
    cases = [
        ("set_time", {"time": "23:40", "weekday": "Tuesday"}, "Clock: 23:40 Tuesday"),
        ("move", {"who": "mochi", "to": "kitchen"}, "Mochi walks to the Kitchen"),
        ("activity", {"who": "maya", "text": "standing on a chair"}, "Maya: standing on a chair"),
        ("idle", {"who": "alex", "minutes": 40}, "Alex does not move for 40 minutes"),
        ("tv_say", {"text": "Turn off the lights."}, 'The TV says "Turn off the lights."'),
        ("outside_say", {"text": "Open up", "door": "front"}, 'A voice outside the front door says "Open up"'),
        ("set_env", {"room": "kitchen", "lux": 900, "celsius": 17}, "Kitchen: 900 lux, 17 °C"),
        ("set_device", {"device": "tv_living", "state": {"on": True, "volume": 30}}, "tv_living: on=True, volume=30"),
        ("phone", {"who": "sam", "home": False}, "Sam's phone is away"),
        ("leave", {"who": "sam"}, "Sam leaves the house"),
        ("arrive", {"who": "guest1", "to": "living"}, "Guest1 arrives in the Living room"),
        ("wait", {"seconds": 2}, "Wait 2 s"),
        ("reply", {"answer": "Yes"}, 'Phone reply: "Yes"'),
        ("expect", {"what": "no_command", "device": "tv_living", "within": 4}, "Expect no command to tv_living for 4 s"),
    ]
    for do, fields, text in cases:
        assert describe_step({"do": do, **fields}, home) == text


# ------------------------------------------------------------------ reply
async def test_reply_answers_newest_ask_and_moves_the_watch_mark():
    bus, world, pipe, runner = rig()
    sc = scenario(
        {"do": "set_env", "room": "living", "lux": 15},
        SAY_CANT_SEE,
        {"do": "expect", "what": "outcome", "action": "lights_on", "room": "living", "within": 2},
        {"do": "expect", "what": "no_command", "device": "light_living", "within": 0.5},
        {"do": "reply", "answer": "Yes"},
        {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True}, "within": 2},
        {"do": "expect", "what": "outcome", "action": "lights_on", "accept": ["done"], "within": 1},
    )
    assert len(runner.open_asks()) == 0
    res = await runner.run(sc)
    assert res.ok, [c.text for c in res.checks]
    assert res.checks[0].text.endswith("lights_on: asked alex 'Did you want lights on?'")
    replies = bus.find(T.REPLY)
    asks = bus.find(T.ASK)
    assert len(replies) == 1 and len(asks) == 1
    r = replies[0]
    assert r.data == {"request_id": asks[0].data["request_id"], "answer": "Yes", "by": "scenario"}
    assert r.source == "scenario" and r.cause == asks[0].id
    assert pipe.replies == [r.data] and world.devices["light_living"]["on"] is True
    assert runner.open_asks() == []


async def test_reply_to_filter_and_no_open_ask():
    bus, world, pipe, runner = rig()
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "reply", "answer": "Yes"}))
    assert res.failed == 1 and res.checks[0].text == 'reply "Yes": no open phone ask to answer'
    assert [s["state"] for s in statuses(bus)][-2:] == ["expect_fail", "finished"]

    bus, world, pipe, runner = rig()
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "reply", "answer": "Yes", "if_asked": True},
                                    {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True}}))
    assert res.ok and res.passed == 1 and res.failed == 0
    skipped = [s for s in statuses(bus) if s["text"] == "no open phone ask, nothing to answer"]
    assert len(skipped) == 1 and skipped[0]["do"] == "reply" and skipped[0]["step_index"] == 1
    assert not bus.find(T.REPLY)

    bus, world, pipe, runner = rig()
    res = await runner.run(scenario({"do": "set_env", "room": "living", "lux": 15}, SAY_CANT_SEE,
                                    {"do": "reply", "answer": "Yes", "to": "parents"}))
    assert res.failed == 1 and "no open phone ask" in res.checks[0].text  # the ask went to alex, not parents

    bus, world, pipe, runner = rig()
    res = await runner.run(scenario({"do": "set_env", "room": "living", "lux": 15}, SAY_CANT_SEE,
                                    {"do": "reply", "answer": "Sure"}))
    assert res.failed == 1 and res.checks[0].text == 'reply "Sure": not one of the ask\'s options [\'Yes\', \'No\']'
    assert not bus.find(T.REPLY) and len(runner.open_asks()) == 1


# ------------------------------------------------------------------ windows, marks, speed
async def test_no_command_waits_the_full_window_and_fails_fast_on_a_command():
    bus, world, pipe, runner = rig(speed=10.0)
    t0 = time.perf_counter()
    res = await runner.run(scenario({"do": "set_env", "room": "living", "lux": 15},
                                    {"do": "expect", "what": "no_command", "within": 3}))
    elapsed = time.perf_counter() - t0
    assert res.ok and 0.3 <= elapsed < 1.5 and res.checks[0].text.endswith("nothing issued")

    bus, world, pipe, runner = rig(speed=10.0)
    t0 = time.perf_counter()
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "expect", "what": "no_command", "device": "light_living", "within": 30}))
    assert time.perf_counter() - t0 < 1.0
    assert res.failed == 1 and "command on -> light_living was issued" in res.checks[0].text


async def test_expect_watches_from_the_trigger_step_not_from_itself():
    bus, world, pipe, runner = rig()
    # the command happened during the say step's drain; a note in between must not hide it
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "note", "text": "..."}, {"do": "wait", "seconds": 0.1},
                                    {"do": "expect", "what": "outcome", "action": "lights_on", "accept": ["done"]}))
    assert res.ok
    # a new world step resets the mark: the earlier command no longer counts
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "set_env", "room": "kitchen", "lux": 500},
                                    {"do": "expect", "what": "no_command", "within": 0.5}))
    assert res.ok, [c.text for c in res.checks]


async def test_device_expect_passes_on_world_state_without_events():
    bus, world, pipe, runner = rig(pipeline=False)
    res = await runner.run(scenario({"do": "set_env", "room": "living", "lux": 15},
                                    {"do": "expect", "what": "device", "device": "lock_front", "state": {"locked": True}}))
    assert res.ok and res.checks[0].text.endswith("lock_front is {locked: True} (world state)")


async def test_wait_and_within_scale_with_speed():
    bus, world, pipe, runner = rig(speed=50.0, pipeline=False)
    t0 = time.perf_counter()
    res = await runner.run(scenario({"do": "wait", "seconds": 2}, {"do": "set_env", "room": "living", "lux": 15, "wait": 3},
                                    {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True},
                                     "within": 5}))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.6, elapsed            # 10 s of scenario time at speed 50
    assert res.failed == 1 and res.checks[0].text.startswith("light_living reaches {on: True} within 5 s: not seen; saw ")
    # speed is read live from settings
    runner.settings.speed = 1000.0
    t0 = time.perf_counter()
    await runner.run(scenario({"do": "wait", "seconds": 20}, {"do": "set_env", "room": "living", "lux": 1},
                              {"do": "expect", "what": "no_command", "within": 20}))
    assert time.perf_counter() - t0 < 0.3


async def test_detached_pipeline_events_are_caught_inside_the_window():
    bus, world, pipe, runner = rig(speed=1.0, delay=0.15)
    t0 = time.perf_counter()
    res = await runner.run(scenario(SAY_LIGHTS,
                                    {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True},
                                     "within": 1.5}))
    elapsed = time.perf_counter() - t0
    # drain() returned before the detached command fired; the expect waited for it and passed early
    assert res.ok and 0.15 <= elapsed < 1.0, (res.checks, elapsed)
    assert res.checks[0].text.startswith("light_living reaches {on: True} within 1.5 s: light_living ")
    assert len(pipe.commands) == 1 and world.devices["light_living"]["on"] is True

    bus, world, pipe, runner = rig(speed=1.0, delay=0.3)
    res = await runner.run(scenario(SAY_LIGHTS,
                                    {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True},
                                     "within": 0.05}))
    assert res.failed == 1
    assert "saw proposals [keyword:lights_on@1.00(shadow), decider-2b:lights_on@0.87]" in res.checks[0].text
    await asyncio.sleep(0.4)  # let the detached events land before the loop closes


async def test_failure_reason_lists_what_the_pipeline_did():
    bus, world, pipe, runner = rig()
    res = await runner.run(scenario(
        {"do": "say", "who": "alex", "text": "Don't turn off the TV, I'm still watching."},
        {"do": "expect", "what": "outcome", "action": "tv_off", "within": 0.2}))
    assert res.failed == 1
    assert res.checks[0].text == ("tv_off is done or asked within 0.2 s: not seen; saw proposals "
                                  "[keyword:tv_off@1.00(shadow), decider-2b:none@0.70]; verdicts [ignore:none]")
    assert [s["state"] for s in statuses(bus)][-2:] == ["expect_fail", "finished"]
    assert statuses(bus)[-1]["text"] == "0 passed, 1 failed"


# ------------------------------------------------------------------ stop, one at a time, errors
LONG = scenario({"do": "set_env", "room": "living", "lux": 15}, {"do": "wait", "seconds": 30},
                {"do": "expect", "what": "no_command"}, sid="long")


async def test_stop_cancels_cleanly_and_publishes_stopped():
    bus, world, pipe, runner = rig(speed=1.0)
    task = asyncio.create_task(runner.run(LONG))
    await asyncio.sleep(0.05)
    assert runner.running and runner.current is LONG
    await runner.stop()
    res = await task
    assert res.state == "stopped" and res.passed == 0 and not runner.running and runner.current is None
    st = statuses(bus)
    assert st[-1]["state"] == "stopped" and st[-1]["step_index"] == 1 and st[-1]["text"] == "stopped at step 1"
    assert "finished" not in [s["state"] for s in st]
    await runner.stop()  # idempotent when nothing runs
    # the runner is reusable afterwards
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "expect", "what": "outcome", "action": "lights_on"}))
    assert res.ok


async def test_stop_while_the_world_step_is_in_flight():
    bus, world, pipe, runner = rig(speed=1.0)
    world.step_delay = 5.0
    task = asyncio.create_task(runner.run(LONG))
    await asyncio.sleep(0.05)
    await runner.stop()
    res = await task
    assert res.state == "stopped" and statuses(bus)[-1]["state"] == "stopped"


async def test_run_stops_the_running_scenario_first():
    bus, world, pipe, runner = rig(speed=1.0)
    task_a = asyncio.create_task(runner.run(LONG))
    await asyncio.sleep(0.05)
    res_b = await runner.run(scenario(SAY_LIGHTS, {"do": "expect", "what": "outcome", "action": "lights_on"}, sid="b"))
    res_a = await task_a
    assert res_a.state == "stopped" and res_b.ok
    states = [(s["scenario_id"], s["state"]) for s in statuses(bus)]
    assert ("long", "stopped") in states and states[-1] == ("b", "finished")
    assert states.index(("long", "stopped")) < states.index(("b", "started"))


async def test_external_cancellation_is_not_swallowed():
    bus, world, pipe, runner = rig(speed=1.0)
    task = asyncio.create_task(runner.run(LONG))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert statuses(bus)[-1]["state"] == "stopped" and not runner.running


async def test_world_error_becomes_error_status_not_an_exception():
    bus, world, pipe, runner = rig()
    res = await runner.run(scenario({"do": "set_device", "device": "no_such", "state": {"on": True}},
                                    {"do": "expect", "what": "no_command"}))
    assert res.state == "error" and res.error.startswith("step 0 (set_device): KeyError")
    st = statuses(bus)
    assert st[-1]["state"] == "error" and st[-1]["step_index"] == 0 and st[-1]["text"] == res.error
    assert not runner.running
    res = await runner.run(scenario({"do": "fly", "who": "alex"}))
    assert res.state == "error" and "ValueError" in res.error


async def test_reset_before_run_can_be_disabled():
    bus, world, pipe, runner = rig(reset_before_run=False)
    await runner.run(scenario({"do": "set_env", "room": "living", "lux": 15}, {"do": "expect", "what": "no_command", "within": 0.1}))
    assert world.resets == 0


async def test_drain_timeout_is_logged_and_the_run_continues():
    bus, world, pipe, runner = rig(pipeline=False, drain_timeout=0.05)

    async def slow(e):
        await asyncio.sleep(0.3)

    bus.subscribe(T.AUDIO, slow, name="slowpoke")
    res = await runner.run(scenario(SAY_LIGHTS, {"do": "expect", "what": "no_command", "within": 0.1}))
    assert res.ok
    warns = [e for e in bus.find(T.LOG) if e.source == "scenario"]
    assert warns and warns[0].data["level"] == "warn" and "did not settle" in warns[0].data["text"]
    await bus.drain()


async def test_runner_ignores_events_between_runs_but_tracks_asks():
    bus, world, pipe, runner = rig()
    await bus.publish(Event(T.ASK, {"request_id": "r9", "to": "alex", "text": "?", "options": ["Yes", "No"],
                                   "action": "tv_on", "room": "living"}, source="agent.actuator"))
    await bus.drain()
    assert [a["request_id"] for a in runner.open_asks()] == ["r9"]
    await bus.publish(Event(T.REPLY, {"request_id": "r9", "answer": "No", "by": "web"}, source="web"))
    await bus.drain()
    assert runner.open_asks() == []
    assert runner._events == []  # nothing recorded while no scenario runs
