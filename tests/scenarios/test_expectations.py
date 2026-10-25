"""expectations.check(): every expect kind against hand-built events, plus describe() and summarize()."""
import pytest

from homedemo.contracts import Event, T
from homedemo.scenarios import expectations as X
from homedemo.scenarios.homeindex import HomeIndex, load_home

HOME = HomeIndex(load_home())


def ev(topic, **data):
    return Event(topic, data, source="test")


def cmd(device, command, **args):
    return ev(T.COMMAND, device_id=device, command=command, args=args)


def ask(action, room="living", to="alex", text="Did you want it?"):
    return ev(T.ASK, request_id="r1", to=to, text=text, options=["Yes", "No"], action=action, room=room)


# ------------------------------------------------------------------ device
def test_device_passes_on_applied_or_device_event_with_state_subset():
    exp = {"what": "device", "device": "light_kitchen", "state": {"on": True}}
    assert X.check(exp, [], HOME) is None
    partial = [ev(T.APPLIED, device_id="light_kitchen", state={"on": False, "brightness": 0})]
    assert X.check(exp, partial, HOME) is None
    other = [ev(T.APPLIED, device_id="light_living", state={"on": True})]
    assert X.check(exp, other, HOME) is None
    hit = ev(T.DEVICE, device_id="light_kitchen", state={"on": True, "brightness": 100})
    m = X.check(exp, other + [hit], HOME)
    assert m and m.ok and m.event_id == hit.id and "sensor.device" in m.text


def test_device_passes_on_world_snapshot_and_numeric_tolerance():
    exp = {"what": "device", "device": "thermostat_main", "state": {"setpoint_c": 20.5}}
    snap = {"devices": [{"device_id": "thermostat_main", "state": {"setpoint_c": 20.5000001}}]}
    m = X.check(exp, [], HOME, snap)
    assert m and m.ok and "world state" in m.text
    assert X.check(exp, [], HOME, {"devices": [{"device_id": "thermostat_main", "state": {"setpoint_c": 19}}]}) is None
    # booleans never equal numbers
    assert not X.values_equal(True, 1) and not X.values_equal(0, False)


# ------------------------------------------------------------------ no_command
def test_no_command_fails_only_on_a_matching_command():
    events = [cmd("light_living", "on", brightness=100), ev(T.ASK, request_id="x", to="alex", action="tv_on")]
    assert X.check({"what": "no_command"}, [], HOME) is None
    m = X.check({"what": "no_command", "device": "any"}, events, HOME)
    assert m and not m.ok and "on -> light_living" in m.text
    assert X.check({"what": "no_command", "device": "tv_living"}, events, HOME) is None
    assert X.check({"what": "no_command", "device": "light_living", "command": "off"}, events, HOME) is None
    assert X.check({"what": "no_command", "device": "light_living", "command": "on"}, events, HOME).ok is False
    assert X.is_negative({"what": "no_command"}) and not X.is_negative({"what": "device"})


# ------------------------------------------------------------------ ask
def test_ask_matches_fields_and_text_substring():
    events = [ask("tv_on", to="parents", text="Turn on the TV for Maya?")]
    assert X.check({"what": "ask"}, events, HOME).ok
    assert X.check({"what": "ask", "to": "parents", "action": "tv_on", "room": "living"}, events, HOME).ok
    assert X.check({"what": "ask", "to": "alex"}, events, HOME) is None
    assert X.check({"what": "ask", "action": "lights_on"}, events, HOME) is None
    assert X.check({"what": "ask", "text": "for maya"}, events, HOME).ok
    assert X.check({"what": "ask", "text": "unlock"}, events, HOME) is None


# ------------------------------------------------------------------ verdict
def test_verdict_lists_rule_substring_and_who():
    v = ev(T.VERDICT, proposal_id="p", action="unlock_door", room="porch", verdict="deny",
           rule="Only voices inside the house count", who={"id": None, "role": "unknown"})
    v2 = ev(T.VERDICT, proposal_id="p2", action="lights_on", room="living", verdict="allow", rule="confidence 0.9",
            who={"id": "alex", "role": "parent"})
    events = [v, v2]
    assert X.check({"what": "verdict", "verdict": "deny"}, events, HOME).event_id == v.id
    assert X.check({"what": "verdict", "verdict": ["ignore", "deny"], "action": "unlock_door"}, events, HOME).ok
    assert X.check({"what": "verdict", "verdict": "deny", "action": ["lights_on"]}, events, HOME) is None
    assert X.check({"what": "verdict", "verdict": "deny", "rule": "inside the house"}, events, HOME).ok
    assert X.check({"what": "verdict", "verdict": "deny", "rule": "curfew"}, events, HOME) is None
    assert X.check({"what": "verdict", "verdict": "allow", "who": "alex"}, events, HOME).event_id == v2.id
    assert X.check({"what": "verdict", "verdict": "allow", "who": "sam"}, events, HOME) is None
    assert X.check({"what": "verdict", "verdict": "allow", "room": "kitchen"}, events, HOME) is None


# ------------------------------------------------------------------ proposal
def test_proposal_by_wildcard_mode_kind_and_confidence_bounds():
    kw = ev(T.PROPOSAL, by="keyword", mode="shadow", action="lights_on", confidence=1.0, kind="speech", room="kitchen")
    d2 = ev(T.PROPOSAL, by="decider-2b", mode="shadow", action="none", confidence=0.61, kind="speech", room="kitchen")
    d12 = ev(T.PROPOSAL, by="decider-12b", mode="act", action="none", confidence=0.45, kind="speech", room="kitchen")
    events = [kw, d2, d12]
    assert X.check({"what": "proposal", "by": "keyword", "action": "lights_on"}, events, HOME).event_id == kw.id
    assert X.check({"what": "proposal", "by": "keyword", "action": "none"}, events, HOME) is None
    assert X.check({"what": "proposal", "by": "decider", "action": "none"}, events, HOME).event_id == d2.id
    assert X.check({"what": "proposal", "by": "decider", "mode": "act"}, events, HOME).event_id == d12.id
    assert X.check({"what": "proposal", "by": ["reflex", "decider-12b"]}, events, HOME).event_id == d12.id
    assert X.check({"what": "proposal", "by": "decider", "min_confidence": 0.5}, events, HOME).event_id == d2.id
    assert X.check({"what": "proposal", "by": "decider", "max_confidence": 0.5}, events, HOME).event_id == d12.id
    assert X.check({"what": "proposal", "by": "reflex"}, events, HOME) is None
    assert X.check({"what": "proposal", "by": "keyword", "kind": "scene"}, events, HOME) is None
    # Claude on Bedrock is a model decision too: the wildcard and the exact name both match it
    br = ev(T.PROPOSAL, by="bedrock", mode="act", action="lights_on", confidence=0.9, kind="speech", room="kitchen")
    assert X.check({"what": "proposal", "by": "decider", "action": "lights_on"}, [kw, br], HOME).event_id == br.id
    assert X.check({"what": "proposal", "by": "bedrock"}, [kw, d2, br], HOME).event_id == br.id
    assert X.check({"what": "proposal", "by": "decider"}, [kw], HOME) is None
    assert X.check({"what": "proposal", "by": "keyword", "room": "living"}, events, HOME) is None


# ------------------------------------------------------------------ outcome
@pytest.mark.parametrize("action,device,command,args", [
    ("lights_on", "light_living", "on", {"brightness": 100}),
    ("lights_off", "light_living", "off", {}),
    ("lights_off", "light_living", "dim", {"brightness": 20}),   # "lights off or dimmer"
    ("lights_dim", "light_living", "dim", {"brightness": 20}),
    ("tv_on", "tv_living", "on", {}),
    ("tv_off", "tv_living", "off", {}),
    ("warmer", "thermostat_main", "set_setpoint", {"delta_c": 1.5}),
    ("cooler", "thermostat_main", "set_setpoint", {"delta_c": -1.5}),
    ("unlock_door", "lock_front", "unlock", {}),
    ("lock_door", "lock_front", "lock", {}),
    ("alert_parent", "phone_parents", "notify", {"to": "parents", "text": "stove"}),
])
def test_outcome_done_for_every_action(action, device, command, args):
    events = [cmd(device, command, **args)]
    m = X.check({"what": "outcome", "action": action, "accept": ["done"]}, events, HOME)
    assert m and m.ok and "done" in m.text, (action, m)
    # a different action's command never satisfies it
    other = [cmd("light_living", "off")] if action != "lights_off" else [cmd("tv_living", "on")]
    assert X.check({"what": "outcome", "action": action, "accept": ["done"]}, other, HOME) is None


def test_outcome_rooms_and_asked_path():
    living_on = [cmd("light_living", "on")]
    assert X.check({"what": "outcome", "action": "lights_on", "room": "living"}, living_on, HOME).ok
    assert X.check({"what": "outcome", "action": "lights_on", "room": "kitchen"}, living_on, HOME) is None
    # thermostat and lock are house-wide: the room is ignored
    assert X.check({"what": "outcome", "action": "warmer", "room": "bedroom_main"},
                   [cmd("thermostat_main", "set_setpoint", delta_c=1.5)], HOME).ok
    assert X.check({"what": "outcome", "action": "unlock_door", "room": "hallway"}, [cmd("lock_front", "unlock")], HOME).ok
    # the asked path, honoured only when accepted
    asked = [ask("lights_on", room="living")]
    assert X.check({"what": "outcome", "action": "lights_on", "room": "living"}, asked, HOME).text.startswith("lights_on: asked")
    assert X.check({"what": "outcome", "action": "lights_on", "accept": ["done"]}, asked, HOME) is None
    assert X.check({"what": "outcome", "action": "lights_on", "room": "kitchen"}, asked, HOME) is None
    assert X.check({"what": "outcome", "action": "lights_on", "room": "living", "accept": ["asked"]}, living_on, HOME) is None
    # a zero or missing delta is not warmer or cooler
    assert X.command_action({"device_id": "thermostat_main", "command": "set_setpoint", "args": {}}, HOME) is None
    assert X.command_action({"device_id": "thermostat_main", "command": "set_setpoint", "args": {"delta_c": 0}}, HOME) is None
    assert X.command_action({"device_id": "unknown", "command": "on"}, HOME) is None


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        X.check({"what": "teleport"}, [], HOME)


# ------------------------------------------------------------------ texts
def test_describe_and_summarize():
    assert X.describe({"what": "device", "device": "light_kitchen", "state": {"on": True}, "within": 3}, HOME) \
        == "light_kitchen reaches {on: True} within 3 s"
    assert X.describe({"what": "no_command", "device": "any", "within": 5}, HOME) == "no command to any device for 5 s"
    assert X.describe({"what": "no_command", "device": "tv_living", "command": "off"}, HOME) == "no off command to tv_living for 5 s"
    assert X.describe({"what": "ask", "to": "parents", "action": "tv_on", "within": 10}, HOME) \
        == "a phone ask to parents about tv_on within 10 s"
    assert X.describe({"what": "verdict", "verdict": ["deny", "ignore"], "action": "unlock_door"}, HOME) \
        == "verdict deny or ignore for unlock_door within 5 s"
    assert X.describe({"what": "proposal", "by": "keyword", "action": "none"}, HOME) == "keyword proposes none within 5 s"
    assert X.describe({"what": "outcome", "action": "lights_on", "room": "living", "within": 8}, HOME) \
        == "lights_on in the Living room is done or asked within 8 s"
    assert X.describe({"what": "outcome", "action": "alert_parent", "accept": ["done"]}, HOME) \
        == "alert_parent is done within 5 s"
    events = [ev(T.PROPOSAL, by="keyword", mode="shadow", action="lights_on", confidence=1.0),
              ev(T.PROPOSAL, by="decider-2b", action="none", confidence=0.61),
              ev(T.VERDICT, verdict="ignore", action="none"), cmd("light_living", "on"),
              ev(T.ASK, to="alex", action="tv_on"), ev(T.LOG, text="noise")]
    assert X.summarize(events) == ("proposals [keyword:lights_on@1.00(shadow), decider-2b:none@0.61]; "
                                   "verdicts [ignore:none]; commands [on->light_living]; asks [alex:tv_on]")
    assert X.summarize([]) == "no proposals, verdicts, commands or asks on the bus"
