# LocoCisco — Presentation Outline

> **Status:** content outline only. Slides are not built yet; this file defines
> what each slide should say and why. One `##` heading = one slide.
>
> **Narrative arc:** LocoCisco started as a local-first *meeting* assistant, but
> the same capability — understanding live, multi-speaker conversation entirely
> on local hardware, with explicit control over what leaves the room — is
> exactly what service settings need. The deck widens the aperture from the
> meeting room to hospitality, restaurants, general customer service, government
> counters, and police stations and vehicles; shows the edge-server + IoT-client
> architecture that makes one deployment serve many rooms and many use cases;
> and frames the edge-vs-cloud choice as a deliberate trade between
> confidentiality (edge) and ease of use and management (cloud).

---

## 1. Title

- **LocoCisco — Conversation intelligence that stays where you need it**
- Subtitle: *Local-first understanding for meetings, service counters, and the field*
- One line: listens without a wake word, attributes who said what, captures
  tasks and decisions, and controls what — if anything — leaves the room.
- Presenter / date / audience placeholder.

**Speaker note:** Set the frame immediately: this is not "another transcription
tool." It is conversation understanding you can place on the edge for
confidentiality or in the cloud for convenience — the same product, deployed to fit.

---

## 2. Team

- **The team behind LocoCisco** — who we are and why we can build this.
- Per-member entries (placeholders to fill):
  - Name · role/title · one-line focus (e.g. local inference & edge systems).
  - Name · role/title · one-line focus (e.g. ML / speech & perception).
  - Name · role/title · one-line focus (e.g. product & deployment).
  - Name · role/title · one-line focus (e.g. security & privacy / policy).
- Optional: short credibility line — relevant background, prior shipped work, or
  domain expertise (edge AI, hospitality/public-sector, security).
- Optional: advisors / partners logos.

**Speaker note:** Keep it fast — names, roles, and the one thing each person
makes credible. The goal is to establish that this team can deliver a local-first
system across sensitive verticals, then move straight into the product.

---

## 3. What LocoCisco already is (grounding)

- A local-first meeting assistant built for edge hardware (Dell Pro Max / NVIDIA GB10).
- Runs the full stack locally: voice activity detection → speech recognition →
  speaker attribution → decision cascade → policy checks → actions.
- No wake word; attributes captions to participants; extracts commitments,
  owners, due dates, and decisions.
- Deterministic policy gate runs *before* anything is posted externally or any
  device acts. The room can see what it heard, what it decided, and what it sent out.
- Works with no internet for core inference; external integrations (e.g. Discord)
  are a clearly separated, optional path.

**Speaker note:** Everything later in the deck is built on capabilities that
already exist in the spec — we are extending proven mechanics to new settings,
not inventing a new product per vertical.

---

## 4. The transferable core

- The hard part LocoCisco solves is **live, multi-speaker conversation
  understanding on local hardware, with an auditable egress boundary.**
- Four reusable primitives:
  1. **Who is speaking** — speaker attribution, seats, explicit check-in, honest "unknown."
  2. **What was said and meant** — streaming captions + intent/decision extraction.
  3. **What should happen** — bounded decision model, escalation only when needed.
  4. **What may leave** — deterministic policy + egress metering + privacy states.
- These primitives are domain-agnostic. A hotel desk, a squad room, and a
  permitting counter all need the same four things.

**Speaker note:** This is the pivot slide. Make the point that meetings were the
first instance of a general capability, then move into the verticals.

---

## 5. Why "localization" is the through-line

- Every new use case shares one word with two meanings, both of which LocoCisco serves:
  - **Local processing** — sensitive conversation stays on-premise; nothing is
    shipped to a third party unless policy allows it.
  - **Language/locale localization** — serving people in *their* language:
    guests, residents, and the public rarely speak one language.
- Day-to-day service work is where these two meet: front-line staff handle
  sensitive interactions, often across a language barrier, dozens of times a day.
- LocoCisco's local models can transcribe, attribute, translate, and summarize
  on-site — no cloud round-trip for either the content or the translation.

**Speaker note:** Define "localization" explicitly here so the use-case slides
can lean on it. It is both "keep it local" and "speak the customer's language."

---

## 6. Use case — Hotels & hospitality

- **Setting:** front desk, concierge, guest services, back-of-house coordination.
- **What it does:**
  - Live multilingual conversation support at the desk — captions and translation
    so staff and international guests understand each other.
  - Captures guest requests as tasks with owner + due time (late checkout, extra
    towels, maintenance, wake-up call) and routes them to the right department.
  - Records what was promised to the guest — accountability without manual notes.
- **Why local:** guest names, room numbers, payment context, and complaints are
  sensitive; keep them on-property.
- **Fit:** an edge server per property; IoT clients at desk, concierge, and
  service stations.

**Speaker note:** Hospitality is the most intuitive opener — multilingual guests
make the "localization" double meaning obvious.

---

## 7. Use case — Restaurants & food service

- **Setting:** host stand, order taking, drive-through, kitchen pass, manager floor.
- **What it does:**
  - Multilingual order capture and read-back; fewer errors across a language barrier.
  - Turns spoken orders and special requests (allergies, modifications) into
    structured tickets.
  - Summarizes shift handoffs and flags commitments ("comp the table," "call the vendor").
- **Why local:** low latency matters at the counter; allergy/dietary and staff
  data stay on-site; works even when the venue's internet is flaky.
- **Fit:** compact edge node per location; IoT clients at stations that need it.

**Speaker note:** Emphasize latency and offline resilience — a restaurant can't
wait on a cloud round-trip at the pass, and shouldn't go down when Wi-Fi does.

---

## 8. Use case — General customer service & services

- **Setting:** retail service desks, banks/credit unions, clinics check-in,
  telecom/utility offices, help counters of any kind.
- **What it does:**
  - Real-time captions + translation so every customer is served in their language.
  - Extracts the ask, the commitment, and the follow-up into a trackable item.
  - Produces an honest record of the interaction (what was said, decided, promised).
  - Accessibility benefit: captions help hard-of-hearing customers and staff.
- **Why local:** account details, health context, and financial data are exactly
  what you don't want traversing a third-party cloud by default.
- **Fit:** edge server behind the branch/office; thin IoT clients at each window.

**Speaker note:** This is the broadest bucket — frame it as "any counter where a
person serves another person," then let the audience map it to their own setting.

---

## 9. Use case — Government services

- **Setting:** permitting and licensing counters, benefits/social services,
  DMV-style offices, city hall, public information desks.
- **What it does:**
  - Serves a multilingual public in their language without a human interpreter
    on every visit.
  - Captures requests, required documents, and commitments; produces a consistent,
    auditable record of each interaction.
  - Supports equitable access and public-records expectations.
- **Why local:** citizen PII, immigration/benefits status, and case data carry
  strict residency, retention, and sovereignty requirements — on-premise processing
  keeps the agency in control and simplifies compliance.
- **Fit:** department-level edge server; IoT clients at public-facing windows;
  policy/egress controls map directly to agency records rules.

**Speaker note:** Lead with data sovereignty and auditability — these are the
words that matter to a public-sector buyer.

---

## 10. Use case — Police stations & vehicles

- **Setting:** front desk / intake, interview rooms, dispatch, and in-vehicle
  (mobile data terminal) use in the field.
- **What it does:**
  - Multilingual assistance at intake and in the field so officers and the public
    understand each other without waiting for a remote interpreter.
  - Captures statements, requests, and decisions with speaker attribution and
    honest "unknown speaker" handling — no guessed identities.
  - Clear on/off-record and privacy states; a deterministic boundary on what is
    retained and what, if anything, is shared.
- **Why local — this is the strongest edge case:** interview and field content is
  highly sensitive and evidentiary; chain-of-custody, retention, and
  jurisdiction rules make cloud processing a non-starter in many agencies. The
  vehicle may have intermittent or no connectivity, so **the system must work
  fully offline.**
- **Fit:** edge server at the station; **ruggedized edge node in the vehicle**
  for disconnected operation; IoT clients at desk and interview rooms.

**Speaker note:** This is the sharpest proof point for "edge when confidentiality
is critical" — and for offline operation. Be careful and precise; stress honest
attribution and explicit privacy states, not surveillance claims.

---

## 11. Use-case recap — one capability, many counters

- Single table/visual across the verticals:
  - Column: setting · the conversation · the commitment captured · why local matters.
  - Rows: Hotels · Restaurants · Customer service · Government · Police.
- The common pattern: *a person serves another person, often across a language
  barrier, with sensitive content, many times a day.*
- LocoCisco turns each of those interactions into an understood, attributed,
  actionable, and controllable record — locally.

**Speaker note:** Reinforce that these are configurations of one product, not
five products. This sets up the architecture section: one platform, many clients.

---

## 12. Architecture — edge server + IoT clients

- **Edge server** (the GB10-class box): runs the heavy models — speech
  recognition, speaker attribution, decision/language models, policy and egress
  control. One server serves a whole site.
- **IoT clients**: light endpoints placed where interactions happen — meeting
  rooms, front desks, service windows, interview rooms, vehicle terminals,
  kiosks. They capture audio/video and present results; they do **not** need to
  run the models.
- **Event bus / contracts** connect them: audio and frames in, attributed
  utterances, proposals, policy verdicts, and results out — the same contracts
  the meeting build already defines.
- **Many clients, many use cases, one server:** a single site runs hospitality,
  back-office, and compliance use cases off the same edge box with
  per-client/per-room configuration and policy.

**Speaker note:** Draw the picture: one edge server in the comms closet, thin
clients fanned out to every room and counter. Stress that adding a location is
adding a client, not another AI deployment.

---

## 13. Architecture — how a client is provisioned

- Each client is defined by: its **location/role**, its **use-case profile**
  (which models and prompts), and its **policy** (what may leave, retention,
  on/off-record defaults).
- Clients are thin and swappable; the model stack and sensitive logic stay on
  the server.
- Scales by adding clients to an existing edge server up to its resource budget;
  add servers to scale sites.
- Local device/LAN control (lamps, displays, signage, call routing) handled by
  the server, same as the meeting build's device path.

**Speaker note:** This slide answers "how do I roll it out?" — centralize the
hard parts, make the edge cheap to add. Keep it concrete.

---

## 14. Deploy anywhere — edge or cloud

- The same platform deploys as:
  - **Edge:** models and policy on an on-premise server; data never leaves unless
    policy allows. Best for confidentiality, latency, and offline operation.
  - **Cloud:** models hosted centrally; clients connect over the network. Best
    for easy onboarding, central management, and sites without edge hardware.
  - **Hybrid:** edge for sensitive capture + local decisions; cloud for
    non-sensitive aggregation, cross-site reporting, and management.
- Model selection and placement is **configuration, not a rebuild** — one
  role-based profile per deployment (the spec already treats models as roles to
  resolve, not hardcoded dependencies).

**Speaker note:** The headline is flexibility *without forking the product*. Same
code, same contracts, different placement. The hybrid option is usually the real
answer for multi-site customers.

---

## 15. Edge vs cloud — the deliberate trade-off

- Two-column comparison (edge | cloud):
  - **Confidentiality / data residency:** edge keeps everything on-site ·
    cloud centralizes data off-site.
  - **Offline operation:** edge works with no internet · cloud needs connectivity.
  - **Latency:** edge is local/immediate · cloud depends on the network.
  - **Setup & management:** edge needs on-site hardware and ops · cloud is
    managed centrally, nothing to install locally.
  - **Scaling a new site:** edge = ship a box · cloud = provision an account.
  - **Cost shape:** edge = capital/hardware up front · cloud = operating/subscription.
- **The rule of thumb:**
  - **Edge is best when confidentiality is critical** — police, government,
    health/financial context, anywhere data can't leave the premises or the vehicle.
  - **Cloud is easiest for clients to use and manage** — fast onboarding, no
    local hardware, central updates; good where content is less sensitive and
    connectivity is reliable.

**Speaker note:** This is the single most important slide for the decision-maker.
State the trade-off plainly and own it: we don't pretend one mode wins
everywhere — we let the customer place the system where their risk and ops reality
put it.

---

## 16. Choosing your deployment (decision guide)

- Simple decision flow:
  - Is the content legally/contractually restricted from leaving the premises?
    → **Edge.**
  - Does it need to work with no/intermittent internet (e.g. vehicles, remote
    sites)? → **Edge.**
  - Is onboarding speed and zero local hardware the priority, with reliable
    connectivity and lower sensitivity? → **Cloud.**
  - Multi-site with mixed sensitivity? → **Hybrid:** edge capture + cloud
    management/reporting.
- Map the earlier verticals onto it: Police/Government → edge; Hotels/Restaurants
  → edge or cloud by policy and connectivity; multi-brand/chains → hybrid.

**Speaker note:** Make the audience self-select. Walk one or two of their likely
scenarios through the flow live.

---

## 17. Privacy, security & trust (carries across every mode)

- Deterministic policy gate before any egress or device action — not a model's discretion.
- Explicit on-record / paused / off-record states; honest pending/failed/delivered
  status on anything sent externally.
- Egress metering and auditable records of what left; bearer-authenticated,
  TLS-protected service boundaries.
- Honest attribution: a visible face is not proof of identity; unknown speakers
  stay unknown. No biometric enrollment or cross-session identity by default.
- These guarantees are strongest on edge (nothing leaves by default) and carried
  forward as explicit controls in cloud/hybrid.

**Speaker note:** This slide is the credibility backstop for the regulated
verticals. Everything here is already in the build spec — we are promising what
the architecture actually enforces, not aspirations.

---

## 18. Roadmap / where this goes

- From the meeting build (proven local pipeline) → service-counter profiles →
  field/vehicle edge nodes → multi-site hybrid management.
- Per-vertical configuration packs (prompts, policies, device integrations).
- Pilot path: one edge server, a few clients, one vertical, measured on real
  hardware before scaling.

**Speaker note:** Keep it honest and staged — the spec is deliberately
specification-first and measurement-driven; the roadmap should feel the same.

---

## 19. Close / call to action

- One capability — local conversation understanding with a controllable egress
  boundary — deployed where each customer needs it.
- **Edge when confidentiality is critical. Cloud when ease of use and management
  win. Hybrid when you need both.**
- Ask: pick a pilot site and a vertical; stand up one edge server and a handful
  of clients; measure on real hardware.

**Speaker note:** End on the trade-off sentence — it's the thesis of the whole deck.

---

## 20. Contact

- **Let's talk — pilot LocoCisco at your site.**
- Primary contact: name · title · email · phone (placeholders to fill).
- Company / team name · website · location.
- Optional: QR code to a one-pager or demo booking link.
- Social / repo handles if relevant.
- Closing line: *Edge when confidentiality is critical. Cloud when ease of use wins.*

**Speaker note:** Leave this slide up during Q&A. Make the single next step
obvious — one email or one booking link — rather than a wall of channels.

---

## Appendix / backup slides (optional)

- A1. Model & perception stack (from the build spec) — for technical audiences.
- A2. Offline-first guarantees and the "no automatic download / no cloud fallback"
  posture.
- A3. Resource budget and sizing on a single edge server (how many clients per box).
- A4. Example event flow for one service interaction, end to end.
- A5. Compliance mapping notes per vertical (data residency, retention, records).
