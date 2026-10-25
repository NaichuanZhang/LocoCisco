import asyncio

from homedemo.bus import Bus, topic_matches
from homedemo.contracts import Event, T


def test_topic_matches():
    assert topic_matches("*", "sensor.camera")
    assert topic_matches("sensor.*", "sensor.camera")
    assert not topic_matches("sensor.*", "voice.utterance")
    assert topic_matches(T.SCENE, T.SCENE)


async def test_publish_chain_and_drain():
    bus = Bus()
    seen = []

    async def a(e):
        await asyncio.sleep(0.01)
        await bus.publish(Event(T.SCENE, {"room": e.data["room"], "text": "x"}, source="agent.perception", cause=e.id))

    async def b(e):
        seen.append(e)

    bus.subscribe(T.MOTION, a, name="perception")
    bus.subscribe(T.SCENE, b, name="sink")
    await bus.publish(Event(T.MOTION, {"room": "kitchen", "motion": True}, source="sim.motion"))
    await bus.drain()
    assert len(seen) == 1 and seen[0].data["room"] == "kitchen"
    assert seen[0].cause == bus.find(T.MOTION)[0].id


async def test_handler_error_becomes_log():
    bus = Bus()

    async def boom(e):
        raise RuntimeError("nope")

    bus.subscribe(T.MOTION, boom, name="boom")
    await bus.publish(Event(T.MOTION, {"room": "kitchen", "motion": True}, source="sim.motion"))
    await bus.drain()
    assert bus.errors and bus.find(T.LOG)[0].data["agent"] == "boom"
