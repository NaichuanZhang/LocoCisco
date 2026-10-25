"""The scenario library itself: every file valid against home.json, the required counts, seed coverage."""
import json
from collections import Counter
from pathlib import Path

import pytest

from homedemo.scenarios import (HERO_TAG, LIBRARY_DIR, REGRESSION_TAG, RESEARCH_TAG, TOUR_TAG, get, load_home,
                                load_scenarios, summary, validate)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "eval" / "seed_cases_2026-10-20.json"

HOME = load_home()
SCENARIOS = load_scenarios()
TOUR_ID = "hero_three_minute_tour"
# What each tour beat must contain (the step kind and a text fragment), in order.
TOUR_BEATS = [
    ("move", "kitchen"),                       # (a) into the dark kitchen -> reflex light
    ("say", "I can't see a thing in here."),   # (b) no wake word -> 2B unsure -> 12B
    ("say", "I left the lights on at the office"),  # (c) a mention, not a request
    ("tv_say", "Turn off the lights."),        # (d) the TV is not a resident
    ("activity", "stove"),                     # (e) child at a lit stove -> parent alert
]


def _tagged(tag):
    return [s for s in SCENARIOS if tag in s["tags"]]


def _wait_seconds(sc):
    """Narration waits + explicit waits + no_command windows at speed 1 (the pipeline's own latency is extra)."""
    waits = sum(float(st.get("wait", 0)) for st in sc["steps"])
    waits += sum(float(st.get("seconds", 0)) for st in sc["steps"] if st["do"] == "wait")
    windows = sum(float(st.get("within", 5)) for st in sc["steps"]
                  if st["do"] == "expect" and st["what"] == "no_command")
    return waits + windows


def test_library_counts():
    heroes, regression, research = _tagged(HERO_TAG), _tagged(REGRESSION_TAG), _tagged(RESEARCH_TAG)
    assert len(heroes) >= 8, [s["id"] for s in heroes]
    assert len(regression) == 33, [s["id"] for s in regression]
    assert len(research) == 14, [s["id"] for s in research]
    assert len(heroes) + len(regression) + len(research) == len(SCENARIOS) == 59
    groups = [set(s["id"] for s in g) for g in (heroes, regression, research)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


def test_ids_unique_and_one_scenario_per_file():
    ids = Counter(s["id"] for s in SCENARIOS)
    assert all(n == 1 for n in ids.values()), ids
    files = sorted(LIBRARY_DIR.glob("*.json"))
    assert len(files) == len(SCENARIOS)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert set(data) == {"id", "title", "blurb", "tags", "steps"}, f.name


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["id"])
def test_every_scenario_is_valid(scenario):
    assert validate(scenario, HOME) == []


def test_heroes_come_first_then_file_order():
    tags = [HERO_TAG in s["tags"] for s in SCENARIOS]
    first_non_hero = tags.index(False)
    assert all(tags[:first_non_hero]) and not any(tags[first_non_hero:])
    hero_ids = [s["id"] for s in SCENARIOS if HERO_TAG in s["tags"]]
    assert hero_ids[0] == TOUR_ID and hero_ids[1] == "hero_midnight_snack" and hero_ids[-1] == "hero_asleep_tv_on"
    assert SCENARIOS[0]["id"] == TOUR_ID  # the demo order starts with the tour
    non_hero = [s["id"] for s in SCENARIOS if HERO_TAG not in s["tags"]]
    assert non_hero == sorted(non_hero)  # reg_* then res_*, file-name order


def test_get_and_summary():
    sc = get("hero_maya_tv_curfew")
    assert sc["title"].startswith("Maya")
    s = summary(sc)
    assert s == {"id": "hero_maya_tv_curfew", "title": sc["title"], "blurb": sc["blurb"], "tags": sc["tags"],
                 "steps": len(sc["steps"])}
    with pytest.raises(KeyError):
        get("no_such_scenario")


def test_load_from_other_dir(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps({"id": "b", "title": "B", "blurb": "", "tags": [], "steps": []}))
    (tmp_path / "a.json").write_text(json.dumps({"id": "a", "title": "A", "blurb": "", "tags": ["hero"], "steps": []}))
    assert [s["id"] for s in load_scenarios(tmp_path)] == ["a", "b"]
    (tmp_path / "bad.json").write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_scenarios(tmp_path)


def test_heroes_have_narration_and_checks_and_pacing():
    for sc in SCENARIOS:
        if HERO_TAG not in sc["tags"] or TOUR_TAG in sc["tags"]:
            continue
        dos = [st["do"] for st in sc["steps"]]
        assert dos.count("note") >= 2, sc["id"]
        assert "expect" in dos, sc["id"]
        # at speed 1: narration waits + no_command windows should land in the 20-60 s band,
        # before the pipeline's own latency (0.8 s per 2B call, ~5 s more per 12B call)
        assert 15 <= _wait_seconds(sc) <= 60, (sc["id"], _wait_seconds(sc))


def test_the_tour_runs_about_three_minutes_and_has_the_five_beats():
    """hero_00: the run order judges see. 170-190 s of narration and windows at speed 1; the real pipeline
    adds its own latency (about 5 2B calls and 2-3 12B calls) on top."""
    tour = get(TOUR_ID)
    assert set(tour["tags"]) >= {HERO_TAG, TOUR_TAG} and _tagged(TOUR_TAG) == [tour]
    assert 170 <= _wait_seconds(tour) <= 190, _wait_seconds(tour)
    steps = tour["steps"]
    assert steps[0]["do"] == "note" and steps[-1]["do"] == "note"  # opens and closes with narration
    assert sum(1 for st in steps if st["do"] == "note") >= 12
    # the five beats, in the brief's order, each followed by its check
    positions = []
    for do, fragment in TOUR_BEATS:
        idx = next(i for i, st in enumerate(steps) if st["do"] == do and fragment in st.get("text", st.get("to", "")))
        positions.append(idx)
    assert positions == sorted(positions), positions
    checks = [(st.get("what"), st.get("by"), st.get("action"), st.get("verdict")) for st in steps if st["do"] == "expect"]
    assert ("device", None, None, None) in checks[:2]                    # (a) light_kitchen on
    assert ("proposal", "reflex", "lights_on", None) in checks           # (a) by the reflex rule
    assert ("outcome", None, "lights_on", None) in checks                # (b) done or asked
    assert ("proposal", "keyword", "lights_on", None) in checks          # (c) the baseline would have acted
    assert ("no_command", None, None, None) in checks                    # (c) nothing moved
    assert ("verdict", None, None, "deny") in checks                     # (d) the TV is denied
    assert ("outcome", None, "alert_parent", None) in checks             # (e) the parent alert
    assert ("proposal", "reflex", "alert_parent", None) in checks        # (e) by the safety rule
    lights_on = next(st for st in steps if st["do"] == "expect" and st.get("action") == "lights_on" and st["what"] == "outcome")
    assert lights_on["accept"] == ["done", "asked"]
    replies = [st for st in steps if st["do"] == "reply"]
    assert replies and all(r["answer"] == "Yes" and r.get("if_asked") for r in replies)


# ------------------------------------------------------------------ research scenarios (res_*.json)
RESEARCH = _tagged(RESEARCH_TAG)
VERIFIED_INCIDENT = "res_01_tv_news_dollhouse"   # the only real-world incident VERIFICATION.md allows us to cite


def test_research_scenarios_are_sourced_and_speech_driven():
    assert [s["id"] for s in RESEARCH] == sorted(s["id"] for s in RESEARCH)
    for sc in RESEARCH:
        assert sc["id"].startswith("res_") and "speech" in sc["tags"], sc["id"]
        sources = [t for t in sc["tags"] if t.startswith("f2:") or t.startswith("eval:")]
        assert len(sources) == 1, (sc["id"], sc["tags"])
        dos = [st["do"] for st in sc["steps"]]
        assert any(d in ("say", "tv_say") for d in dos), sc["id"]
        assert ("proposal", "keyword") in [(st.get("what"), st.get("by")) for st in sc["steps"] if st["do"] == "expect"], sc["id"]
        assert "note" not in dos  # regression-style: no narration, the README carries the story
        assert _wait_seconds(sc) <= 20, (sc["id"], _wait_seconds(sc))


def test_research_no_ops_forbid_asks_too():
    """A no-op research case (mention, negation, state question, TV) needs an `ignore` verdict or the
    policy's deny, not only `no_command`: an ask would also be wrong."""
    for sc in RESEARCH:
        expects = [st for st in sc["steps"] if st["do"] == "expect"]
        kinds = {st["what"] for st in expects}
        if "no_command" in kinds and "outcome" not in kinds:
            verdicts = [st for st in expects if st["what"] == "verdict"]
            assert verdicts, sc["id"]
            assert all(set(_as_list(v["verdict"])) <= {"ignore", "deny"} for v in verdicts), sc["id"]


def test_research_decider_cases_accept_done_or_asked_then_reply():
    for sc in RESEARCH:
        outcomes = [st for st in sc["steps"] if st["do"] == "expect" and st["what"] == "outcome"]
        if not outcomes:
            continue
        assert outcomes[0]["accept"] == ["done", "asked"], sc["id"]
        dos = [st["do"] for st in sc["steps"]]
        reply = dos.index("reply")
        assert reply > dos.index("expect") and sc["steps"][reply].get("if_asked") is True, sc["id"]
        assert dos[reply + 1] == "expect" and sc["steps"][reply + 1]["what"] in ("device", "outcome"), sc["id"]
        if sc["steps"][reply + 1]["what"] == "outcome":
            assert sc["steps"][reply + 1]["accept"] == ["done"], sc["id"]


def test_at_risk_research_cases_say_why():
    for sc in RESEARCH:
        if "at-risk" in sc["tags"]:
            assert "at-risk" in sc["blurb"].lower(), sc["id"]


def test_only_the_verified_incident_is_cited():
    for sc in SCENARIOS:
        text = json.dumps(sc, ensure_ascii=False)
        if sc["id"] == VERIFIED_INCIDENT:
            assert "BBC" in text and "2017" in text and "incident" in sc["tags"]
        else:
            assert "BBC" not in text and "incident" not in sc["tags"], sc["id"]


def _as_list(value):
    return list(value) if isinstance(value, list) else [value]


def test_regression_covers_every_seed_case_and_quotes_its_utterance():
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    by_seed = {}
    for sc in SCENARIOS:
        for tag in sc["tags"]:
            if tag.startswith("seed:"):
                by_seed.setdefault(tag[5:], []).append(sc)
    for section, body in seeds.items():
        for case in body["cases"]:
            assert case["id"] in by_seed, f"no regression scenario for seed {case['id']}"
            assert len(by_seed[case["id"]]) == 1
            sc = by_seed[case["id"]][0]
            assert REGRESSION_TAG in sc["tags"] and section in sc["tags"], sc["id"]
            if section in ("speech", "who"):
                quotes = [q for q in _quoted(case["t"])]
                said = [st["text"] for st in sc["steps"] if st["do"] in ("say", "tv_say", "outside_say")]
                assert any(q in said for q in quotes), (sc["id"], quotes, said)


def _quoted(text: str) -> list[str]:
    out, i = [], 0
    while True:
        a = text.find('"', i)
        if a < 0:
            return out
        b = text.find('"', a + 1)
        if b < 0:
            return out
        out.append(text[a + 1:b])
        i = b + 1


def test_keyword_wrong_scenarios_show_the_baseline():
    """Every scenario tagged keyword-wrong carries a proposal expect on the keyword baseline."""
    for sc in SCENARIOS:
        if "keyword-wrong" in sc["tags"]:
            kinds = [(st.get("what"), st.get("by")) for st in sc["steps"] if st["do"] == "expect"]
            assert ("proposal", "keyword") in kinds, sc["id"]
