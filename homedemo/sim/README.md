# homedemo.sim: the simulated home and its mocked sensors

Owner: **sim** workstream. Implements `contracts.WorldAPI`. Everything leaves through the bus as
`sensor.*` / `sim.clock` / `action.applied` events with `source="sim.<kind>"` and `ts=World.clock`.

```python
from homedemo.sim import World, SensorHub, load_home

home = load_home()                              # homedemo/home.json (or load_home(path))
world = World(home, bus, settings, seed=7, noise=0.3)
await world.start()      # subscribes to action.command, publishes the boot report (clock + every sensor)
await world.step({"do": "move", "who": "alex", "to": "kitchen"})
await world.tick()       # one step of slow physics (call it from the runtime; nothing runs in the background)
world.snapshot()         # JSON for the UI
await world.reset()      # back to home.json, same seed, boot report again
```

`seed` fixes the standing spots and every score; the same seed and the same steps give the same
events. `noise` (0..1) scales the jitter on scores and on the mic azimuth. At `noise=0` every
score is its base value exactly. Jitter never changes a label (who spoke, what the camera saw,
where a voice came from): labels are computed from the true state before jitter is added.

## Boot report (`start()` and `reset()`)

`sim.clock`, then `sensor.device` for every device, `sensor.door` for every door, and per room
(in `home.json` order) `sensor.motion`, `sensor.lux`, `sensor.temp` and, where there is a camera,
`sensor.camera`; finally `sensor.phone` for each resident who has a phone. After that everything is
published **on change** only.

## Scenario steps -> events

| step | state change | events emitted |
|---|---|---|
| `set_time` `{time?, weekday?}` | clock jumps forward to the time (an earlier time means the next day, the weekday rolls over unless given); `set_env` lux overrides are cleared | `sim.clock`; `sensor.lux` for rooms whose daylight changed |
| `move` `{who, to, x?, y?}` | person or pet placed in `to` (seeded spot, or the given canvas x/y); person's `activity`, `idle_min`, `pose`/`near` attrs cleared. A person who is out comes home (= `arrive`) | `sensor.motion` for a room that became empty (`false`) or occupied (`true`); `sensor.camera` for the rooms left and entered when they have a camera |
| `activity` `{who, text, attrs?}` | `activity` caption and the camera attrs (`age_group`, `pose`, `near`, `wearing`, `holding`, ...) **replaced**; `idle_min` reset; `near: <device id or kind>` (not a TV) moves the person next to that device | `sensor.camera` for the room (nothing in a bedroom: no camera there) |
| `idle` `{who, minutes}` | clock jumps `minutes`; `idle_min += minutes`; indoor rooms drift toward the setpoint (0.5 C per minute) | `sim.clock`; `sensor.camera` with `attrs.idle_min` if the room has a camera; `sensor.lux` / `sensor.temp` on change. Presence stays `true`, so no `sensor.motion` |
| `say` `{who, text, lang?, match?, asr_conf?, wake_word?}` | none (the speaker must be at home) | one `sensor.audio` from the mic in the speaker's room (no mic there: the nearest mic, e.g. a hallway voice reaches `mic_kitchen`) |
| `tv_say` `{text, device?, lang?}` | the TV is turned on if it was off | `sensor.device` if it was turned on; `sensor.audio` from the mic in the TV's room with `doa.source="tv"`, speaker unknown |
| `outside_say` `{text, door?, who?, lang?}` | none | `sensor.audio` from the mic of the outdoor room behind the door (porch), `doa.source="outside"`; speaker unknown, or the named resident's voice (a parent locked out) |
| `set_env` `{room, lux?, celsius?}` | lux pins the room's ambient light (lux reads exactly the given value; lights still add on top) until the next `set_time`/`reset`; celsius sets the room temperature (indoor rooms then drift back on `tick`) | `sensor.lux` and/or `sensor.temp` for the room (always, even if unchanged) |
| `set_device` `{device, state}` | state keys merged (unknown keys raise); lights are normalised (`on` without brightness -> 100, brightness 0 -> off) | `sensor.device` (always); `sensor.door` for a lock whose door changed; `sensor.lux` for a light |
| `phone` `{who, home}` | the resident's phone presence (residents without a phone raise) | `sensor.phone` on change |
| `leave` `{who, door?}` | person out of the house (`home=false`, no room); phone leaves with them | `sensor.door` open then closed; `sensor.motion false` if the room emptied; `sensor.camera` for the room left; `sensor.phone false` |
| `arrive` `{who, to, name?, role?, age_group?, door?, x?, y?}` | unknown `who` creates a guest (`role` default `guest`, voice not enrolled, no phone, no camera identity); a resident who was out comes home with their phone; someone already home is just moved | `sensor.door` open then closed; `sensor.motion true` if the room was empty; `sensor.camera`; `sensor.phone true` for a returning resident |

Runner steps (`wait`, `reply`, `expect`, `note`) and anything not in `contracts.SCENARIO_STEPS`
raise `ValueError`, as do unknown people, rooms, devices and malformed arguments.

## Commands (`action.command` -> the sim)

| `command` | device kind | effect | events (all with `cause` = the command event id) |
|---|---|---|---|
| `on` `{brightness?}` | light | on, brightness given / previous / 100 | `action.applied`, `sensor.device`, `sensor.lux` if the room's light changed |
| `off` | light | off, brightness 0 | `action.applied`, `sensor.device`, `sensor.lux` |
| `dim` `{brightness}` | light | brightness 0..100 (0 = off) | `action.applied`, `sensor.device`, `sensor.lux` |
| `on` `{volume?}` / `off` | tv | on/off | `action.applied`, `sensor.device` |
| `set_setpoint` `{delta_c}` or `{setpoint_c}` | thermostat | setpoint clamped to 10..30 | `action.applied`, `sensor.device`; rooms then drift 0.5 C per `tick()` -> `sensor.temp` |
| `lock` / `unlock` | lock | locked true/false | `action.applied`, `sensor.device`, `sensor.door` |
| `notify` `{to, text}` | any `device_id` (`"phone"` if missing) | appended to `world.notifications` (`{to, text, command_id, time}`) | `action.applied` with state `{to, text, delivered: true}` |

`action.applied` is `{"device_id", "state", "command_id"}` and is always published for a valid
command, even when the state did not change. A command the device cannot do (`dim` a TV, an
unknown device) raises `ValueError` inside the handler; the bus records it in `bus.errors` and
publishes a `trace.log` warning. Nothing is applied.

## `tick(seconds=1.0)`

Advances the clock, drifts every indoor room 0.5 C toward the thermostat setpoint, follows the
daylight curve. Publishes `sim.clock` when the minute changed, `sensor.temp` on change (>= 0.05 C)
and `sensor.lux` when it moved by at least max(1 lux, 3 %).

## Sensor model

**Light.** `lux = ambient + 300 * brightness / 100` per light that is on. Ambient is a half sine
between sunrise and sunset (`thresholds.night_to` .. `night_from` in home.json: 06:30..19:30),
10 000 lux outdoors at the peak, 8 % of that indoors (400..900 lux on a sunny afternoon), with a
night floor of 2 lux indoors and 1 lux outdoors. A lit room at night reads about 302 lux.

**Temperature.** Indoor rooms start at the setpoint (19 C), the porch at 12 C and never follows the
thermostat. Drift is 0.5 C per `tick()` and per `idle` minute.

**Motion** is occupancy: `true` while any person *or pet* is in the room, `false` once it empties.
Idle or sleeping people keep it `true`.

**Camera** frames exist only for rooms listed under `sensors.cameras` (not the bedrooms). Each
person is a `person` detection with `conf` 0.80..0.97, a `track_id` stable for the whole run
(`t01`, `t02`, ... in order of first sight), and attrs `age_group` (`child` for role child or age
< 16, else `adult`), plus whatever the last `activity` step set (`pose`, `near`, `wearing`,
`holding`, ...), `idle_min` after an `idle` step, and `identity: {id, match}` for residents only
(guests have no identity). Pets are detections labelled by species (`cat`). `activity` is the
distinct activity captions of the people and pets in the room joined with `; `, or `null`.
A frame is re-published only when what the camera sees changed (not for jitter).

**Audio** (`sensor.audio`): `text`, `lang` (`zh` when the text has CJK characters, else `en`,
or the step's `lang`), `asr_conf`, `wake_word` (false unless the step says so), `speaker` and
`doa`. The mic in the source's room hears it (porch mic for outside, living mic for the TV); a room
with no mic (hallway) is heard by the nearest mic and the event's `room` is that mic's room.

* `speaker`: enrolled residents `{id, name, role, match 0.88..0.97}`; guests `{id: null, role:
  "guest", match < 0.5}`; TV, outside strangers and non-enrolled people `{id: null, role:
  "unknown", match < 0.5}`. `say` may pass `match` to force a weak match.
* `doa.azimuth_deg`: bearing from the mic to the source in canvas degrees (0 = +x/right,
  90 = +y/down, clockwise on screen), jittered by up to +-5 deg at `noise=1`.
  `doa.source`: `outside` if the source is in an outdoor room; `tv` if a TV in the mic's room is
  **on** and the true bearing is within 15 deg of it; else `person`. From `mic_living` the TV sits
  at 180 deg, so a person placed at `x=210, y=160` while the TV plays is heard as the TV.
  Default standing spots avoid the TV cone; a scenario puts someone in it with `move x/y`
  (`activity attrs.near: "tv"` deliberately does not snap the person there).

**Scores at noise 0** (base, +-amplitude at noise 1): camera conf 0.885 +-0.085, face match 0.92
+-0.05, enrolled voice 0.925 +-0.045, guest voice 0.30 +-0.15, unknown voice 0.15 +-0.10,
ASR 0.92 +-0.05, DOA 0.90 +-0.06.

## Snapshot

`{"time","weekday","school_night","clock","rooms":[{id,name,x,y,w,h,indoor,camera,env:{lux,celsius}}],
"devices":[{device_id,kind,room,x,y,state}],"people":[{id,name,role,room,x,y,activity,attrs,idle_min,
home,phone_home,guest}],"pets":[{id,name,species,room,x,y,activity}],"doors":[{door_id,name,between,
x,y,open,locked}],"mics":[{mic_id,room,x,y}],"notifications":[{to,text,command_id,time}]}`.
People who are out have `room`, `x`, `y` = `null`. `school_night` is true Sunday..Thursday from
18:00 and in the small hours that follow (until sunrise).
