# LocoCisco: a smart home that understands without a wake word

Smart homes make you talk like a robot: a wake word, then a command. Real people say "I can't see a thing in here," or "客厅有点暗，turn on the lights." They also talk to each other, and the TV talks too. LocoCisco is a home with no wake word. It listens, watches and decides for itself: was that meant for me, and what should I do?

**How it works.** The house is a set of small agents on one event bus. Mocked cameras, microphones and room sensors feed a perception agent and a voice agent. The voice agent works out who spoke: a person, the TV, or someone outside. One Strands house agent decides. Inside it, the Strands Decider 2B sits in the agent's path as a DeciderToolRouter and picks the tool. When the 2B is sure (confidence 0.80 or more), the house acts on the hub: zero tokens, and nothing leaves the home. When it is unsure, Claude Haiku 4.5 on Amazon Bedrock decides with the same tools. Claude only gets text; audio and video stay home. Plain code keeps the safety rules: reflexes like "dark kitchen, lights on", ignoring TV voices, and permissions, so a child cannot unlock the front door.

**Why a decision model fits.** Nobody has labelled data for "was this sentence meant for the house?", and training a classifier per home is not worth it. The model must run on a small hub inside the home. Someone is waiting in the dark, so it must be fast.

**Results.** We measured 93 speech cases and 42 camera scenes. On speech, the cascade picked the right action 95% of the time, against 83% for the 2B alone and 94% for Claude on every case. The 2B handled 37% of speech cases by itself, all correctly. On scenes, the cascade was right 98% of the time, against 69% for the 2B alone. On no-wake-word speech, a keyword rule got 5 of 14 right; the 2B got 12. Mandarin and mixed Mandarin-English work on the hub. On a CPU the 2B answers in about 2.4 seconds; the Strands team reports 0.1 to 0.2 seconds on a GPU. A Claude escalation adds about 1.5 seconds.

**The demo.** A 3D house plays 59 scripted scenarios, from a three-minute tour to hard cases like negation and "it's a sauna in here." A "Decider X-ray" panel shows each decision: what was heard or seen, the typed question sent to the 2B, each option's probability against the threshold, and where it was decided, LOCAL or BEDROCK. A scoreboard counts LLM calls avoided.

**Reuse it.** DeciderToolRouter is a reusable Strands intervention. Five lines put a small decider in front of any Strands agent, so the big model runs only when needed.

**What we learned.** Both models made mistakes no home should make, like turning lights on in a sunny kitchen. So the model never owns safety: rules handle reflexes, and code handles permissions. The decider handles what code cannot: what people mean.
