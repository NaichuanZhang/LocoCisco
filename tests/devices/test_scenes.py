"""Scenes map to the right pilots; colour / brightness helpers; pilot <-> state round trips."""
from __future__ import annotations

import pytest

from unspoken.devices.scenes import (DIMMING_MAX, DIMMING_MIN, SCENE_NAMES, SCENES, TEMP_MAX, TEMP_MIN,
                                     clamp_dimming, format_color, merge_pilot, parse_color, pilot_for_set,
                                     pilot_to_state, scene_for_pilot, scene_pilot)


def test_scene_table_has_every_scene_the_room_can_ask_for():
    assert set(SCENE_NAMES) == {"talk", "slides", "break", "privacy", "paused", "off_record", "bright", "dim", "off"}


def test_privacy_scenes_are_the_agreed_colours():
    assert SCENES["privacy"]["g"] > SCENES["privacy"]["r"] and SCENES["privacy"]["g"] > SCENES["privacy"]["b"]  # soft green
    assert SCENES["paused"] == {"state": True, "r": 255, "g": 160, "b": 0, "dimming": 60}                  # amber
    assert SCENES["off_record"]["r"] == 255 and SCENES["off_record"]["g"] == 0 and SCENES["off_record"]["b"] == 0
    assert SCENES["off"] == {"state": False}
    assert SCENES["bright"]["dimming"] == 100 and SCENES["dim"]["dimming"] < SCENES["slides"]["dimming"] < SCENES["talk"]["dimming"]


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_every_scene_pilot_is_something_a_bulb_accepts(name):
    pilot = SCENES[name]
    assert isinstance(pilot["state"], bool)
    if pilot["state"]:
        assert DIMMING_MIN <= pilot["dimming"] <= DIMMING_MAX
        if "temp" in pilot:
            assert TEMP_MIN <= pilot["temp"] <= TEMP_MAX and not {"r", "g", "b"} & pilot.keys()
        else:
            assert all(0 <= pilot[c] <= 255 for c in ("r", "g", "b"))


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_round_trips_through_pilot_to_state(name):
    st = pilot_to_state(scene_pilot(name))
    assert st["scene"] == name
    assert st["on"] is SCENES[name]["state"]
    if name == "privacy":
        assert st["color"] == "#3cc85a" and st["temp"] is None
    if name == "talk":
        assert st["temp"] == 4000 and st["color"] is None


def test_scene_pilot_is_a_copy():
    p = scene_pilot("talk")
    p["dimming"] = 1
    assert SCENES["talk"]["dimming"] == 80


def test_scene_for_pilot_ignores_extra_bulb_fields_and_rejects_near_misses():
    bulb = {"mac": "x", "rssi": -50, "state": True, "sceneId": 0, "r": 255, "g": 0, "b": 0, "c": 0, "w": 0, "dimming": 70}
    assert scene_for_pilot(bulb) == "off_record"
    assert scene_for_pilot({**bulb, "dimming": 71}) is None
    assert scene_for_pilot({"state": True, "dimming": 80, "r": 255, "g": 255, "b": 255}) is None   # white via RGB is not "talk"
    assert scene_for_pilot({"state": False, "r": 255, "g": 0, "b": 0}) == "off"


def test_parse_and_format_color():
    assert parse_color("#ff8800") == (255, 136, 0)
    assert parse_color("FF8800") == (255, 136, 0)
    assert format_color(255, 136, 0) == "#ff8800"
    for bad in ("#fff", "red", "#gg0000", ""):
        with pytest.raises(ValueError):
            parse_color(bad)


def test_clamp_dimming_respects_the_bulb_floor():
    assert clamp_dimming(0) == 10 and clamp_dimming(5) == 10 and clamp_dimming(10) == 10
    assert clamp_dimming(50.4) == 50 and clamp_dimming(100) == 100 and clamp_dimming(250) == 100


def test_pilot_for_set():
    assert pilot_for_set(brightness=40, color="#ff8800") == {"state": True, "r": 255, "g": 136, "b": 0, "dimming": 40}
    assert pilot_for_set(brightness=0) == {"state": False}
    assert pilot_for_set(brightness=0, color="#ff0000") == {"state": False}
    assert pilot_for_set(brightness=3) == {"state": True, "dimming": 10}
    assert pilot_for_set(color="#00ff00") == {"state": True, "r": 0, "g": 255, "b": 0}
    with pytest.raises(ValueError):
        pilot_for_set()
    with pytest.raises(ValueError):
        pilot_for_set(brightness=50, color="blue")


def test_merge_pilot_models_what_the_bulb_keeps():
    cur = {"state": True, "temp": 4000, "dimming": 80}
    assert merge_pilot(cur, {"dimming": 20}) == {"state": True, "temp": 4000, "dimming": 20}      # colour kept
    red = merge_pilot(cur, {"state": True, "r": 255, "g": 0, "b": 0})
    assert "temp" not in red and red["dimming"] == 80                                            # white mode dropped
    assert merge_pilot(red, {"state": False}) == {**red, "state": False}                         # off keeps the colour
    assert "r" not in merge_pilot(red, {"temp": 2700})                                           # back to white mode


def test_pilot_to_state_reports_brightness_even_when_off():
    st = pilot_to_state({"state": False, "temp": 2700, "dimming": 15})
    assert st == {"on": False, "brightness": 15, "color": None, "temp": 2700, "scene": "off"}
