# Models: one file to swap any model

Every model LocoCisco runs has a **role**; `config/models.toml` says which **provider** serves each role, per
**profile**. Code asks the registry for a role and never names a vendor, a URL or a model id:

```python
from unspoken.models import load_registry

reg = load_registry()                 # profile: UNSPOKEN_PROFILE, default "dev"
decider = reg.decider()               # unspoken.core.decider.Decider (role "decider")
model = reg.strands_model("llm")      # strands Model: Qwen on vLLM / llama.cpp, or Claude on Bedrock in dev
asr, vad, tts = reg.asr(), reg.vad(), reg.tts()
reg.label("llm")                      # "Qwen3.6 35B-A3B NVFP4 on vLLM (GPU)" for the room screen / X-ray
await reg.check()                     # one RoleHealth per role; await reg.aclose() at shutdown
```

Roles (`unspoken/contracts.py`): `decider`, `decider_big`, `llm`, `vlm`, `asr`, `vad`, `diarization`, `faces`,
`asd`, `tts`, `embed`, `pii`. A role a profile does not name is **disabled**: `reg.enabled(role)` is False and
its getter raises `RoleDisabled`. Objects are built lazily on first use and cached per role.

## Profiles

| Profile | What it is |
|---|---|
| `dev` | this cloud desktop: Strands Decider 2B/12B on CPU, Claude on Bedrock as the llm (remote, dev only), fakes for hearing and vision |
| `dev-local` | `dev` with nothing remote: the llm is a small Qwen3 on a local llama.cpp server (`scripts/dev-llm.sh start`); `local_only = true` |
| `dev-sensors` | `dev-local` plus the real in-process sensors providers on the CPU (Silero VAD, faster-whisper, WeSpeaker, YuNet, MotionASD) |
| `gb10` | the Dell Pro Max: everything local on the box (Decider 12B, Qwen3.6 on vLLM, NeMo-Speech.cpp ASR/TTS, in-process sensors); `local_only = true` |
| `fake` | tests and the simulator: deterministic fakes everywhere, no servers |

Pick one with `UNSPOKEN_PROFILE=gb10`, `load_registry("gb10")`, or `--profile gb10` on the CLIs
(`python -m unspoken.models`, `python -m homedemo`, `python -m homedemo.play`).

## Switch a model in one edit

Edit the role's block in the profile:

```toml
[profiles.gb10.llm]
provider = "openai"                     # any OpenAI-compatible server: vLLM, llama.cpp, Ollama
url = "http://127.0.0.1:8000/v1"
model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
label = "Qwen3.6 35B-A3B NVFP4 on vLLM (GPU)"
options = { max_tokens = 400, temperature = 0, timeout_s = 120 }
```

Keys of a role block: `provider`, `url`, `model`, `label`, `options` (an inline table of provider-specific knobs,
handed to the provider as `RoleSpec.options`). Anything else is a hard error, so a typo cannot silently
disable a model. Profile keys: `local_only` and `extends = "other-profile"` (start from another profile's
blocks; a child block replaces the whole role block).

Or override one key without editing anything:

```bash
UNSPOKEN_LLM_MODEL=Qwen/Qwen3-8B UNSPOKEN_LLM_URL=http://127.0.0.1:8000/v1 python -m unspoken.meeting
UNSPOKEN_DECIDER_BIG_PROVIDER=none ...                       # role names with _ work the same way
```

`UNSPOKEN_<ROLE>_<KEY>` with KEY in `PROVIDER`, `URL`, `MODEL`, `LABEL`. In code: `load_registry(overrides=
{"llm.url": "...", "llm.options.max_tokens": "300"})`. Precedence: overrides dict > env > the profile's TOML >
the profile it extends > disabled. Unknown roles or keys in the dict raise `ConfigError`. `python -m
unspoken.models show` prints the resolved table and the overrides that were applied.

### local_only

`local_only = true` (gb10, dev-local, dev-sensors, fake) makes `load_registry` refuse, with
`RemoteNotAllowed`, the `bedrock` provider and any `url` whose host is not loopback, RFC 1918
(10/8, 172.16/12, 192.168/16, fc00::/7), `*.local` or `inference.local`. Hostnames are not resolved: a public
name is refused even if it points at the LAN. This is the guarantee behind "nothing leaves the box".

## Check every role

```
python -m unspoken.models check [--profile P] [--roles llm,asr] [--json] [--timeout 120]
python -m unspoken.models show  [--profile P] [--roles ...] [--json]
```

`check` runs `healthy()` plus ONE tiny real request per enabled role and prints role, provider, target,
ok, probe latency and a detail line; exit code 1 if any enabled role fails. The probes: decider = one noul
question, llm/vlm = a 1-token chat completion (an uncached model with `max_tokens = 1`), embed = one string,
asr = 0.5 s of silence, tts = "ok", pii = one sentence, vad/diarization/faces/asd = build the provider
(loading the model is the cost) and open one stream/session. Server providers take `health` in options:
`"/ready"` (NeMo-Speech.cpp, resolved against the server root), `"models"` (default, `{url}/models`) or a
full URL.

```
$ python -m unspoken.models check --profile dev-local
ROLE         PROVIDER         TARGET                    OK  LATENCY  DETAIL
decider      strands-decider  http://127.0.0.1:8765     ok  582 ms   P(true)=0.96, model decider-2b
decider_big  strands-decider  http://127.0.0.1:8766     ok  3024 ms  P(true)=0.98, model decider-12b
llm          openai           http://127.0.0.1:8010/v1  ok  381 ms   stop=max_tokens, tokens=19+1, reply='ok'
...
11/11 enabled roles ok
```

## Providers

| Role(s) | Provider | What it builds | Options |
|---|---|---|---|
| decider, decider_big | `strands-decider` | `unspoken.core.decider.HttpDecider` on `POST {url}/v1/systemone` | `model` or `options.name` = client name (`decider-2b`), `timeout`, `max_concurrency` |
| decider, decider_big | `fake` | `FakeDecider` | `script = [["substring", "question", "Option"], ...]`, `latency_ms` |
| llm, vlm | `openai` | strands `OpenAIModel` on any OpenAI-compatible server. Tools go out as Chat Completions `tools`, tool calls come back as streamed `tool_calls` deltas (vLLM `--tool-call-parser`, llama.cpp `--jinja`) | `api_key` ("local"), `timeout`/`timeout_s`, `temperature` (0), `max_tokens`, `stream` (true), standard Chat Completions params (`tool_choice`, `top_p`, `seed`, ...) as themselves, anything else (`reasoning`, `chat_template_kwargs`) into `extra_body` |
| llm, vlm | `bedrock` | strands `BedrockModel` (dev only; refused when `local_only`) | `region`, `profile` (AWS profile), `temperature`, `max_tokens` |
| llm, vlm | `scripted` (alias `fake`) | `unspoken.core.fakes.ScriptedModel` | `tool_calls = [["lights_on", {room = "living"}]]`, `reply`, `fail` |
| asr | `openai-audio` | `POST {url}/audio/transcriptions`, multipart WAV, `model` field | `language`, `response_format` (`verbose_json` -> words), `temperature`, `health`; other keys are sent as extra form fields (`diarization = false`) |
| asr | `fake` | `FakeASR` | `texts = [...]` (cycled) or `text` |
| tts | `openai-speech` | `POST {url}/audio/speech` -> WAV bytes | `voice`, `speed`, `response_format` (`wav`), `health`; other keys as extra JSON fields |
| tts | `fake` | `FakeTTS` (a tone, 60 ms per character) | `sample_rate` |
| embed | `openai` | `POST {url}/embeddings` | `dimensions`, `batch` |
| embed | `fake` | `FakeEmbedder` (hashed bag of words, L2-normalised) | `dim` (64) |
| pii | `regex` | `RegexPII`: e-mails, phones, keys/tokens, money amounts | |
| pii | `fake` | `FakePII` | `words = { "SECRET" = "secret" }` |
| vad / diarization / faces / asd | `fake` | energy VAD, loudness-bucket diarizer, fixed face boxes, first-face ASD | `threshold`, `hang_ms`, `bucket`, `tracks` |
| any | `python:<module>:<Factory>` | `Factory(spec)` where `spec` is the `RoleSpec`; must implement the role's protocol | whatever the factory reads from `spec.options` |
| any | `none` | disabled | |

The sensors workstream's real in-process providers are `python:unspoken.sensors.vad:SileroVAD`,
`python:unspoken.sensors.asr_local:FasterWhisperASR`, `python:unspoken.sensors.diar:EmbeddingDiarizer`,
`python:unspoken.sensors.faces:YuNetFaces`, `python:unspoken.sensors.asd:MotionASD` (see `dev-sensors`
and `gb10`). Heavy imports (openai, strands, boto3, onnxruntime, ctranslate2) happen inside the builder of
the provider that needs them, so a profile that never uses one never imports it.

## Add a provider

1. Write the class. It takes the `RoleSpec` (`spec.url`, `spec.model`, `spec.options`) in its constructor
   and implements the role's protocol from `unspoken/contracts.py` (`ASR.transcribe`, `TTS.synthesize`,
   `VAD.stream`, `Embedder.embed`, ...). Give it `async healthy()` (or sync) and `aclose()` if it opens
   connections; an optional `describe()` string shows up in the `check` table.
2. Fastest route, no registry change: name it in the TOML as `provider = "python:your.module:YourClass"`.
3. To give it a short name, add one line to `BUILDERS` in `unspoken/models/providers.py`:
   `("asr", "whisper-cpp"): _your_builder` where `_your_builder(spec)` imports the class lazily and returns
   `YourClass(spec)`. `providers_for(role)` and the error messages pick it up automatically.
4. If `check` needs a different tiny probe, add it to `PROBES` in `unspoken/models/probes.py` (one per role,
   not per provider; the existing ones work for any implementation of the protocol).
5. Add a test under `tests/models/` with a `httpx.MockTransport` (see `tests/models/test_openai_compat.py`):
   the suite must run with no servers.

## The home demo

`python -m homedemo [--profile dev-local] [--escalate bedrock|12b|llm]` takes its deciders from the
registry: slot `2b` = role `decider`, `12b` = role `decider_big`, and `--escalate llm` runs the Strands house
agent on the registry's `llm` role (Qwen on llama.cpp with `dev-local`, Qwen3.6 on vLLM with `gb10`). The
X-ray's model names and routes come from `reg.label(role)`. `--escalate bedrock` works as before (Claude
from `Settings`).

## Local llama.cpp server for `dev-local`

`scripts/dev-llm.sh start|stop|status` runs the pinned llama.cpp b11869 (`llama-server --jinja`) with
`Qwen3-4B-Q4_K_M.gguf` on `127.0.0.1:8010`; the script header lists the two download URLs and sha256s
(both public). On this desktop a probe takes ~0.4 s and a full house-agent escalation (1.4k-token prompt,
tool call, reply) 1.5-3 s on 16 CPU threads.

## Privacy notes

The registry never writes audio or video anywhere; the ASR provider streams PCM to the server as an
in-memory WAV and drops it. In-process providers keep voices and faces session-only (that is the sensors
workstream's `reset()` / `stop()`). `local_only` profiles cannot name a remote host.
