"""validate(): every class of mistake a scenario author can make is reported with a readable message."""
from homedemo.scenarios import load_home, validate

HOME = load_home()


def scenario(*steps, **over):
    base = {"id": "t", "title": "T", "blurb": "b", "tags": ["x"], "steps": list(steps)}
    base.update(over)
    return base


def errors_of(*steps, **over):
    return validate(scenario(*steps, **over), HOME)


def test_valid_minimal():
    assert errors_of({"do": "say", "who": "alex", "text": "hi"},
                     {"do": "expect", "what": "no_command", "within": 1}) == []


def test_shape_errors():
    assert validate([], HOME) == ["scenario must be a JSON object"]
    errs = validate({"id": "Bad Id!", "title": "", "blurb": 3, "tags": "hero", "steps": []}, HOME)
    assert any("id must match" in e for e in errs)
    assert any("title must be" in e for e in errs)
    assert any("blurb must be" in e for e in errs)
    assert any("tags must be" in e for e in errs)
    assert any("steps must be a non-empty list" in e for e in errs)
    assert any("unknown top-level keys" in e for e in errors_of({"do": "note", "text": "n"}, extra=1))
    assert any("no expect step" in e for e in errors_of({"do": "note", "text": "n"}))


def test_unknown_step_and_bad_wait():
    errs = errors_of({"do": "teleport", "who": "alex"}, {"do": "say", "who": "alex", "text": "x", "wait": -1})
    assert any("unknown do 'teleport'" in e for e in errs)
    assert any("wait must be a number >= 0" in e for e in errs)
    assert any("must be an object with a 'do' key" in e for e in errors_of("say"))


def test_expect_before_any_trigger():
    errs = errors_of({"do": "note", "text": "n"}, {"do": "expect", "what": "no_command"})
    assert any("expect before any world or reply step" in e for e in errs)
    assert errors_of({"do": "set_time", "time": "10:00"}, {"do": "expect", "what": "no_command"}) == []


def test_world_step_ids():
    errs = errors_of(
        {"do": "set_time", "time": "25:00", "weekday": "Funday"},
        {"do": "move", "who": "bob", "to": "attic"},
        {"do": "activity", "who": "alex", "text": "", "attrs": []},
        {"do": "idle", "who": "mochi", "minutes": 0},
        {"do": "say", "who": "alex", "text": "x", "lang": "fr"},
        {"do": "tv_say", "text": "x", "device": "light_living"},
        {"do": "outside_say", "text": "x", "door": "back"},
        {"do": "set_env", "room": "kitchen"},
        {"do": "set_device", "device": "tv_living", "state": {"colour": "red"}},
        {"do": "phone", "who": "maya", "home": "yes"},
        {"do": "leave", "who": "nobody"},
        {"do": "arrive", "who": "Guest One", "to": "garden"},
        {"do": "expect", "what": "no_command"},
    )
    joined = "\n".join(errs)
    for needle in ("time must be HH:MM", "'weekday' has unknown value(s) ['Funday']", "unknown person or pet 'bob'",
                   "unknown room 'attic'", "'text' must be a non-empty string", "'attrs' must be an object",
                   "'minutes' must be a number >= 1", "'lang' has unknown value(s) ['fr']", "'light_living' is not a tv",
                   "unknown door 'back'", "needs 'lux' and/or 'celsius'", "has no state key(s) ['colour']",
                   "'maya' has no phone", "'home' must be true or false", "unknown person or pet 'nobody'",
                   "'who' must be an id like 'guest1'", "unknown room 'garden'"):
        assert needle in joined, needle


def test_arrive_introduces_a_guest():
    assert errors_of({"do": "arrive", "who": "guest1", "to": "living"},
                     {"do": "say", "who": "guest1", "text": "hello"},
                     {"do": "expect", "what": "no_command"}) == []
    errs = errors_of({"do": "say", "who": "guest1", "text": "hello"}, {"do": "expect", "what": "no_command"})
    assert any("unknown person or pet 'guest1'" in e for e in errs)
    # a guest has no phone and cannot be a 'phone' step subject
    errs = errors_of({"do": "arrive", "who": "guest1", "to": "living"}, {"do": "phone", "who": "guest1", "home": True},
                     {"do": "expect", "what": "no_command"})
    assert any("unknown resident 'guest1'" in e for e in errs)


def test_runner_step_fields():
    errs = errors_of(
        {"do": "say", "who": "alex", "text": "x"},
        {"do": "wait", "seconds": -2},
        {"do": "note"},
        {"do": "reply", "answer": "", "to": "nobody", "if_asked": "yes"},
        {"do": "expect", "what": "teleported"},
        {"do": "expect", "what": "device", "device": "light_living", "state": {}, "within": 0},
        {"do": "expect", "what": "device", "device": "lock_front", "state": {"on": True}},
        {"do": "expect", "what": "no_command", "device": "nope", "command": 3},
        {"do": "expect", "what": "ask", "to": "mochi", "action": "fly"},
        {"do": "expect", "what": "verdict"},
        {"do": "expect", "what": "verdict", "verdict": ["deny", "maybe"], "action": "fly", "who": "guest1"},
        {"do": "expect", "what": "proposal", "action": "none"},
        {"do": "expect", "what": "proposal", "by": "oracle", "mode": "dry", "kind": "smell", "min_confidence": 2},
        {"do": "expect", "what": "outcome", "action": "none", "accept": ["maybe"], "room": "attic"},
    )
    joined = "\n".join(errs)
    for needle in ("'seconds' must be a number >= 0", "'text' must be a non-empty string", "'answer' must be a non-empty",
                   "'to' has unknown value(s) ['nobody']", "'if_asked' must be true or false",
                   "'what' must be one of", "'state' must be a non-empty object", "'within' must be a number >= 0.1",
                   "'lock_front' has no state key(s) ['on']", "unknown device 'nope'", "'command' must be a string",
                   "'to' has unknown value(s) ['mochi']", "'action' has unknown value(s) ['fly']",
                   "'verdict' is required", "'verdict' has unknown value(s) ['maybe']", "'who' has unknown value(s) ['guest1']",
                   "'by' is required", "'by' has unknown value(s) ['oracle']", "'mode' has unknown value(s) ['dry']",
                   "'kind' has unknown value(s) ['smell']", "'min_confidence' must be between 0 and 1",
                   "'action' must be a real action", "'accept' must be a non-empty subset", "unknown room 'attic'"):
        assert needle in joined, needle


def test_expect_valid_variants():
    assert errors_of(
        {"do": "say", "who": "alex", "text": "x"},
        {"do": "expect", "what": "verdict", "verdict": ["deny", "ignore"], "action": ["unlock_door"], "rule": "voice"},
        {"do": "expect", "what": "proposal", "by": ["decider", "reflex"], "action": ["lights_on", "none"],
         "mode": "act", "kind": "speech", "min_confidence": 0.5, "max_confidence": 1},
        {"do": "expect", "what": "outcome", "action": "warmer", "accept": ["done"]},
        {"do": "expect", "what": "ask", "to": "parents", "action": "tv_on", "text": "tv"},
        {"do": "expect", "what": "no_command", "device": "any", "command": "on"},
        {"do": "reply", "answer": "Yes", "to": "alex", "if_asked": True},
    ) == []
