"""Scorer sanity: an oracle transcript scores perfect, swapped seats score zero seat accuracy."""
import json
from pathlib import Path

from eval.transcribe.score import cpwer, der, normalize, score_transcript, wer


def _case(tmp: Path) -> Path:
    words = [{"speaker": "A", "seat": "left", "on_camera": True, "start": 0.0, "end": 0.5, "text": "Okay"},
             {"speaker": "A", "seat": "left", "on_camera": True, "start": 0.5, "end": 1.0, "text": "everybody"},
             {"speaker": "C", "seat": "right", "on_camera": True, "start": 2.0, "end": 2.4, "text": "Twenty"},
             {"speaker": "C", "seat": "right", "on_camera": True, "start": 2.4, "end": 2.9, "text": "five"}]
    seg = [{"session_id": "t", "speaker": "A", "start_time": 0.0, "end_time": 1.0, "words": "Okay everybody"},
           {"session_id": "t", "speaker": "C", "start_time": 2.0, "end_time": 2.9, "words": "Twenty five"}]
    (tmp / "ref.words.json").write_text(json.dumps(words))
    (tmp / "ref.seglst.json").write_text(json.dumps(seg))
    (tmp / "meta.json").write_text(json.dumps({"case": "t"}))
    return tmp


def _tr(left: str, right: str) -> dict:
    return {"speakers": {"person-left": {"seat": left}, "person-right": {"seat": right}},
            "utterances": [{"start": 0.0, "end": 1.0, "speaker": "person-left", "text": "okay, everybody."},
                           {"start": 2.0, "end": 2.9, "speaker": "person-right", "text": "25"}]}


def test_oracle_is_perfect(tmp_path):
    r = score_transcript(_case(tmp_path), _tr("left", "right"))
    assert r["wer"] == 0 and r["cpwer"] == 0 and r["der"] < 0.01 and r["sa_acc"] == 1 and r["seat_acc"] == 1


def test_swapped_seats(tmp_path):
    r = score_transcript(_case(tmp_path), _tr("right", "left"))
    assert r["sa_acc"] == 1 and r["seat_acc"] == 0           # the mapping rescues sa_acc, never seat_acc


def test_parts():
    assert normalize("Uh, Twenty-five!") == "25"
    assert wer("a b c d", "a b x d") == 0.25
    assert cpwer({"A": "one two", "B": "three"}, {"x": "three", "y": "one two"})[0] == 0
    d = der([(0, 1, "A"), (1, 2, "B")], [(0, 1, "y"), (1, 2, "y")], collar=0)
    assert abs(d["confusion"] - 0.5) < 0.02
