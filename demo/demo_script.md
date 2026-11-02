# 3-minute demo

**Setup (before you go on stage)**
- `.venv/bin/uvicorn app.backend.main:app --port 8031`
- Clean slate: `curl -X POST localhost:8031/demo/reset` → open http://localhost:8031/pawtown → **Seed demo Den** (1 real photo triage, ≈$0.001) → **+ Walker's 2nd Den** (no model call).
- Tabs ready: `/pawtown`, `/pawtown#lab`, `/#replay`, `/#compare`; the live scanner `?key=` link on your phone.
- Live calls in this script: ≈4 × $0.001. Needs a real browser (WebGL).

| Time | Show | Say |
|---|---|---|
| 0:00 | PawTown, the Den over the Mission | "SF has more dogs than kids. Most pets have several caregivers: partners, roommates, a dog walker. Nobody knows who gave the meds, or when 'watch it' becomes 'go now'." |
| 0:15 | `/pawtown#lab` | "We fine-tuned GLM 5.3 Flash with a vision LoRA on Fireworks serverless training. On a locked 712-row test set, split by scenario: photos **93% vs 26%** for the base model, p ≈ 7e-61. With safety rules, **87% vs 43%** overall, CI [+32, +58] pp. And against a frontier model, GPT-6.1 Terra Pro: **87% vs 69%**, photos **93% vs 53%**, at under half the cost and a fifth of the latency. Honest part: on text it's a tie, and the frontier model under-triages less." |
| 0:45 | `/#replay` → **Show a fine-tune win** | "These are the LoRA's real recorded answers. The base model often can't even produce a valid answer on photos; 63% of its photo outputs failed our schema." Click **Show a fine-tune miss** once: "and here's where it's wrong." |
| 1:05 | Phone: live scanner, pet name "Miso", point at the ringworm photo | "The voice agent and camera never change. Only the model behind `/scan` does." Yellow report appears. |
| 1:25 | Back to PawTown | Within 5 s: "📷 Live scan of Miso" in the log; contagion quests for Biscuit, Pepper and the humans. The ⚡ chip shows **$0.0010 · 1.6 s**. |
| 1:45 | View as **Jordan (helper)** | Diagnoses and photos disappear; Jordan's route covers **both** Dens. Mark Biscuit's carprofen **Done** → 🛑 double dose blocked (Jordan already gave it). |
| 2:10 | **Demo: Pepper ate chocolate** | Real call + toxin rule → `emergency_now`. The city turns red, the game pauses, route + phone for **Sage Veterinary Center** (a real OSM 24h vet), poison-control line. "The game never rewards waiting." Click **We're on our way**. |
| 2:40 | Pet card → **Vet packet PDF** | Timeline, care adherence %, caregiver log. "The vet gets this before you walk in." |
| 2:55 | — | "A tenth of a cent per triage. Fine-tuned on Fireworks." |

**Q&A honesty notes**
- The LoRA isn't served live: it needs a 4×B300 deployment (~$60/h). Live calls use base + rubric prompt (83.3% on the test set). The LoRA numbers were measured through that deployment before it was deleted.
- Test labels are rubric-derived and the set was locked without human review. The messages are synthetic.
- The app repairs off-vocabulary model output (drops unknown words, logs them); the **scorer does not**, so the eval numbers stay strict.
- Vets come from OpenStreetMap; hours can be stale, so the UI says call ahead.

## Backup clip (record once, before 3:30)
In case venue Wi-Fi or the Cloudflare tunnel fails:
1. Screen-record your phone (Control Center → Screen Recording, mic **on**).
2. Open the scanner link, enter pet name **Miso**, species cat.
3. Point at `demo/media/ringworm_cat.jpg` shown on your laptop screen. Say: "There's a bald patch on her face, is it serious?"
4. Let the agent box the spot and finish the report (~30–40 s). Stop recording.
5. Then record 10 s of PawTown showing the new "📷 Live scan of Miso" log line.
Put both clips in `demo/media/` (gitignored) and keep them open in QuickTime.
