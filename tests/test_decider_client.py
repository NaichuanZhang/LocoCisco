from homedemo.decider.questions import build_scene_query, build_speech_query
from homedemo.decider_client import FakeDecider


async def test_fake_scripted_and_interpret():
    d = FakeDecider(script=[("can't see", "intent", ("Lights on", 0.93))])
    q = build_speech_query({"room": "living", "text": "I can't see a thing in here.",
                            "context": "Living room. Lights off, evening."})
    ans = await d.ask(q.state, q.questions)
    it = q.interpret(ans)
    assert it.action == "lights_on" and it.confidence == 0.93
    assert abs(sum(it.probabilities.values()) - 1) < 1e-6


async def test_fake_default_scene():
    d = FakeDecider()
    q = build_scene_query({"room": "kitchen", "text": "Time 23:40. Kitchen camera: 1 adult."})
    it = q.interpret(await d.ask(q.state, q.questions))
    assert it.action in {"lights_on", "tv_on", "warmer", "tv_off", "alert_parent", "none"}
