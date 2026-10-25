# MODELS workstream report (ws/models)

Goal: every model swappable by config. `unspoken.models.load_registry()` implements the `Registry` protocol
from `unspoken/contracts.py`; `config/models.toml` maps roles to providers per profile; the home demo runs on
it. Branch `ws/models`, 3 commits + 2 merges of `hack/main` (sensors, screen, deploy), nothing pushed.

## 1. What was built

| Path | One line |
|---|---|
| `unspoken/models/__init__.py` | `load_registry(profile, path, overrides)`; profile arg > `UNSPOKEN_PROFILE` > `dev`, path arg > `UNSPOKEN_MODELS` > `config/models.toml` |
| `unspoken/models/config.py` | TOML -> one `RoleSpec` per role: profiles, `extends`, `local_only`, env overrides `UNSPOKEN_<ROLE>_<KEY>`, dict overrides (`llm.url`, `llm.options.max_tokens`); unknown role/key -> `ConfigError` naming file, profile, key |
| `unspoken/models/registry.py` | `ModelRegistry`: lazy build + per-role cache, `spec/enabled/label`, typed getters that check the protocol, `check()`, `aclose()` |
| `unspoken/models/providers.py` | `(role, provider) -> builder` table, `python:module:Factory` loader (`Factory(spec)`), `none`; every heavy import lives inside its builder |
| `unspoken/models/openai_compat.py` | strands `OpenAIModel` builder (Chat Completions tools + streamed tool calls, `extra_body` routing), `BedrockModel` builder, `ScriptedModel` from options; `OpenAIAudioASR`, `OpenAISpeechTTS`, `OpenAIEmbedder` over httpx with `health` (`/ready`) and `timeout_s` options |
| `unspoken/models/netpolicy.py` | `local_only`: refuses `bedrock` and any host that is not loopback / RFC 1918 / fc00::/7 / `*.local` / `inference.local` (`RemoteNotAllowed`) |
| `unspoken/models/fakes.py` | deterministic fakes for asr, tts, vad (energy gate), diarization (session-stable labels + reset), faces, asd, embed (hashed, normalised), pii |
| `unspoken/models/pii.py` | `RegexPII`: e-mails, phones, keys/tokens (sk-, AKIA, ghp_, xox, Bearer, long hex, `password=`), money amounts |
| `unspoken/models/probes.py` | `check()` probes: one noul question, 1-token chat (uncached `max_tokens=1` model), one embed string, 0.5 s silence to ASR, "ok" to TTS, one sentence to PII, build + one session for the in-process roles; sync or async `healthy()` |
| `unspoken/models/__main__.py` | `python -m unspoken.models show|check [--profile] [--path] [--roles a,b] [--json] [--timeout]`; exit 1 if an enabled role fails |
| `unspoken/models/audio.py` | PCM16 <-> WAV helpers (stdlib) |
| `config/models.toml` | profiles `dev`, `dev-local`, `dev-sensors`, `gb10`, `fake`, every block commented |
| `docs/MODELS.md` | switch a model in one edit, env overrides, the check command, provider table, how to add a provider |
| `requirements/models.txt` | pins (section 5) |
| `scripts/dev-llm.sh` | start/stop/status of the pinned llama.cpp b11869 + Qwen3-4B-Q4_K_M on 127.0.0.1:8010 |
| `tests/models/` | 117 tests (config, registry, netpolicy, fakes, PII, OpenAI-compatible providers with `httpx.MockTransport` incl. a full strands tool-call loop, CLI, home-demo wiring) + 2 live tests (`UNSPOKEN_LIVE=1`) |
| `homedemo/contracts.py` (Settings only) | `profile: str = "dev"`, `escalate_to` gains `"llm"` |
| `homedemo/runtime.py` | `build_deciders(settings, registry)`: slot `2b` = role `decider`, `12b` = `decider_big`, `llm` = house agent on `registry.strands_model("llm")` with `registry.label("llm")`; `Runtime` owns and closes the registry, rebuilds on `profile` change; `BLURBS["llm"]` |
| `homedemo/agents/house_agent.py` | `HouseAgent(name, label, route, ...)`: works for the `bedrock` and `llm` slots; `HouseResult.by_router`; tool rounds capped at 2 (a small local model otherwise loops) |
| `homedemo/agents/cascade.py` | house-agent cascade for `bedrock` or `llm`; X-ray `model`/`route` from the decider objects (registry labels) with the old constants as fallback; repeated identical tool calls collapse into one proposal |
| `homedemo/cli.py`, `__main__.py`, `play.py`, `web/server.py` | `--profile`, `--escalate llm`, `play` lists `llm` decisions, `SettingsBody.escalate_to` accepts `llm` |

Reusable dev artefacts (not in the repo) under `/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models/`:
`llama.cpp/llama-b11869/` (unpacked `llama-b11869-bin-ubuntu-x64.tar.gz`, sha256 `529ebb30…a6e`) and
`gguf/Qwen3-4B-Q4_K_M.gguf` (2.5 GB, sha256 `7485fe6f…dfd5`, verified against the HF LFS oid).

## 2. How to run it

```bash
cd unspoken-wt/models            # (or hack/main after the merge)
UV_CACHE_DIR=... scripts/dev-venv.sh && uv pip install --python .venv/bin/python -r requirements/models.txt
# optional, for dev-sensors: uv pip install --python .venv/bin/python -r requirements/sensors.txt

.venv/bin/python -m unspoken.models show  --profile gb10            # resolved table, no network
.venv/bin/python -m unspoken.models check --profile dev             # against the running 2B/12B (+ Bedrock)
scripts/dev-llm.sh start                                             # llama.cpp Qwen3-4B on :8010
.venv/bin/python -m unspoken.models check --profile dev-local        # all local
.venv/bin/python -m homedemo.play --profile dev-local --escalate llm --speed 8 hero_cant_see reg_p03_freezing
.venv/bin/python -m homedemo      --profile dev-local --escalate llm   # the web demo on 127.0.0.1:8780
scripts/dev-llm.sh stop
UNSPOKEN_LLM_URL=http://127.0.0.1:8000/v1 UNSPOKEN_LLM_MODEL=... python -m unspoken.models check --profile gb10
```

## 3. Verification (this desktop, x86_64, no GPU)

- Full suite: `.venv/bin/python -m pytest -q` -> **677 passed, 9 skipped** (383 original + sensors/screen tests
  from the merges + 117 models tests; the skips are live tests). Models tests alone: 117 passed in 1.8 s, no servers.
- `check --profile dev-local` (live): decider 2B ok 582 ms (`P(true)=0.96`), decider_big 12B ok 3024 ms, **llm ok 381 ms**
  (`stop=max_tokens, tokens=19+1, reply='ok'` from Qwen3-4B on llama.cpp through strands `OpenAIModel`), 11/11 enabled roles ok.
- `check --profile dev` (live): deciders ok (766 ms / 3933 ms); **llm FAIL `NoCredentialsError`** because this
  sandbox has no AWS credentials for profile `bedrock`; expected here, and exactly what the exit code is for.
- `check --profile dev-sensors` (live, real in-process providers from ws/sensors): Silero VAD ok 227 ms, faster-whisper
  base.en ok 617 ms (`text='You'` for 0.5 s of silence: Whisper's usual silence hallucination, harmless as a probe),
  CAM++ diarizer, YuNet, MotionASD ok; 7/7.
- `check --profile gb10` here: in-process roles ok, `llm/vlm/asr/tts` FAIL with connection errors (those servers
  only exist on the GB10), `decider_big` reported `disabled`; exit 1 as it should.
- `homedemo.play --profile dev-local --escalate llm`: **real tool calls from the local model.** `hero_cant_see`: 2B
  unsure 0.77 -> llm `lights_on` (tool `lights_on`, 2.6 s, 2778 tokens) -> PASS. `hero_mandarin_cold`: 2B 0.42 ->
  llm `warmer` 1.5 s -> PASS. `reg_p14_sisters_heater`: llm `ask_resident(warmer)` -> PASS. `reg_p03_freezing`,
  `reg_p08_wheres_the_remote`, `reg_s05_sweater` PASS after the two fixes below. `hero_office_lights` FAIL: the 4B
  called `lights_off` for the office remark (a model-quality miss, not a wiring one; the scenario expects no
  command). 8 of 9 distinct scenarios passed; every escalation produced a tool call in 1.5-3 s.
  First run found two real problems and fixed them: the small model kept re-calling the same tool after each
  tool result (14 model calls, scenarios stalled past the runner's 30 s window) -> `HouseAgent` now stops after
  two tool rounds via strands' cooperative `agent.cancel(after_current_tools=True)`; and identical repeated
  calls became duplicate proposals -> the cascade collapses them.
- Live tests: `UNSPOKEN_LIVE=1 pytest tests/models/test_live.py` -> 2 passed (dev deciders probe; dev-local
  house agent made a real `lights_on` tool call with the router bypassed).
- Offline proof of the OpenAI tool path: `tests/models/test_openai_compat.py` drives a real strands `Agent` through
  an SSE stream shaped like llama.cpp/vLLM (`tool_calls` deltas split across chunks, usage-only final chunk): the
  tool runs with the parsed arguments, the tool result goes back as a `role: tool` message with the right
  `tool_call_id`, usage sums to 860+24.
- llama-server was stopped at the end (`scripts/dev-llm.sh stop`). The systemd decider services were not touched.

## 4. Not verifiable here; how to verify on the GB10

- vLLM Qwen3.6 with `tool_choice = "auto"` and `reasoning = false` in `extra_body` (vLLM accepts unknown body
  keys; if its OpenAI server rejects `reasoning`, delete that option and rely on `chat_template_kwargs = { enable_thinking = false }`
  instead): `UNSPOKEN_PROFILE=gb10 python -m unspoken.models check --roles llm,vlm`, then
  `python -m homedemo.play --profile gb10 --escalate llm --mode cascade hero_cant_see` and look for `tools=['lights_on']`.
- NeMo-Speech.cpp ASR/TTS field names (`model`, `response_format=verbose_json`, `diarization`, `voice`): `check --roles asr,tts`
  prints the parsed text / WAV size; a 4xx shows up verbatim in the DETAIL column.
- The 12B on :8765 reporting `model decider-12b`; `decider_big` flip to the 2B block if deploy runs it.
- aarch64 wheels: everything in `requirements/models.txt` is pure Python (section 5); the sensors pins were checked
  by that workstream. `llama.cpp` is not needed on the GB10 (vLLM serves the llm), but b11869 ships
  `llama-b11869-bin-ubuntu-arm64.tar.gz` and `-cuda-13.4-arm64` if a fallback server is wanted.

## 5. Contract changes, deps, blocks for other owners

Contract change requests: **none.** Two notes for the integrator:
- `Registry.decider(role)` returns objects that also carry a `label` attribute (set by the registry on
  `HttpDecider`/`FakeDecider`); the home demo's X-ray reads it. If the integrator prefers it in the contract,
  add `label: str` to `Decider` in `unspoken/core/decider.py`.
- `HouseResult` (homedemo) gained `by_router: bool`; `HouseAgent` gained `name/label/route/short/who/max_tool_rounds`.

Deps added (`requirements/models.txt`, all pure-Python wheels, aarch64 fine): `strands-agents[openai]==1.62.0`,
`openai==2.54.0`, `aws-bedrock-token-generator==1.1.0` (pulled by the extra; unused on the GB10), `httpx==0.28.1`,
`boto3==1.43.118` (bedrock provider, dev only). None lack aarch64 wheels.

For the **deploy** owner (docs/HACKDAY.md section 4): the TTS provider name is **`openai-speech`**, not
`openai-audio` (`openai-audio` is the ASR provider); the gb10 blocks in `config/models.toml` are the reconciled
ones: ASR `openai-audio` on :8770 with `options = { response_format = "verbose_json", language = "en",
diarization = false, timeout_s = 20, health = "/ready" }`, TTS `openai-speech` on :8771 with
`options = { voice = "default", response_format = "wav", timeout_s = 10, health = "/ready" }`,
`decider_big` = `none` with the 2B block commented, llm options `{ api_key = "none", max_tokens = 400,
temperature = 0, timeout_s = 120, tool_choice = "auto", reasoning = false }`. Please copy them over the HACKDAY block.

For the **sensors** owner: options pass through as `spec.options` exactly as in your section 5; `gb10.asd` is
`MotionASD`; `dev-sensors` carries your blocks; sync `healthy()` is accepted.

For the **meeting / screen** owners: `reg.label(role)` is the display name (every gb10 role has one), `reg.spec(role).provider`
tells local vs server, `await reg.check()` gives the health table for a preflight card.

## 6. Open risks

- Qwen3-4B on CPU occasionally picks a wrong tool (`hero_office_lights`); the demo model is Qwen3.6 on the GPU, so
  this is a dev-profile quality note, not a blocker. The two-round cap keeps a looping model to one extra call.
- `reasoning = false` / `tool_choice` for vLLM are untested against a live vLLM here (see section 4).
- The llm probe spends one token per `check`; the decider probes hit the live servers (fine, they are shared).
- `Settings.decider_2b_url/decider_12b_url` override the profile only when they differ from the dataclass defaults
  (kept so `eval/` and the live tests keep working); the profile's TOML is the source of truth otherwise.
