# LangGraph for Developers and Architects — Progress Tracker

> **Learner Profile:** Backend/middleware developer (GraphQL, Node.js, ReactJS, Python). Completed "Claude Certified Architect – Foundations" and "Generative AI Essentials With LangChain" (all 8 modules — LCEL, prompt templates, guardrails, memory, RAG, ReAct agents). Prefers developer-centric explanations, practical code, and fast onboarding.
>
> **Assume known, do not re-teach:** LCEL and the `|` pipe operator, Runnables (`RunnableLambda`/`RunnableMap`), `ChatPromptTemplate` / `MessagesPlaceholder`, `StrOutputParser`, HuggingFace `ChatHuggingFace` + `HuggingFaceEndpoint` setup, embeddings + FAISS/Chroma retrieval, the `@tool` decorator, and `create_react_agent` from `langgraph.prebuilt`.
>
> **The pivot to teach:** the LangChain course ended at the *prebuilt* agent. This course opens that box — explicit state, graphs, edges, and control flow.

**Course stats (from syllabus):** 15h 30m · 8 Modules · 1 Other Item

---

## Syllabus Roadmap

| # | Module | Duration | Status | Lesson File |
|---|--------|----------|--------|-------------|
| — | Prelude | — | ⏭️ Skipped by choice | — |
| 1 | The Rise of Agentic Workflows and the Emergence of LangGraph | 2h 15m | ✅ Done | 01-module1-rise-of-agentic-workflows.md |
| 2 | LangGraph Architecture and Ecosystem | 45m | ⬜ Not started | See sub-topics below |
| 3 | Getting Started with LangGraph: Building Your First AI Workflow Graph | 30m | ⬜ Not started | See sub-topics below |
| 4 | Prebuilt Agents in LangGraph | 1h | ⬜ Not started | See sub-topics below |
| 5 | Designing Custom Workflows with LangGraph | 4h 20m | ⬜ Not started | See sub-topics below |
| 6 | Dynamic AI Graphs: Combining Subgraphs and Streaming | 1h 30m | ⬜ Not started | See sub-topics below |

**Legend:** ⬜ Not started · 🔄 In progress · ✅ Done · ⏭️ Skipped by choice

**Teaching content tracked: 10h 20m.**

> **On the duration delta:** the syllabus header reads 15h 30m / 8 modules. Everything not tracked above is **intentionally out of scope** — Prelude, Exercises, and all Quizzes. The learner opted out to avoid burning time on non-teaching content. Deliberate scope decision, not a gap in the syllabus capture. Do not chase it, do not re-add quiz rows, and do not generate quiz questions unless explicitly asked.

---

### Module 1 Sub-Topics — The Rise of Agentic Workflows and the Emergence of LangGraph
*2h 15m · 4 Web Modules — all 4 covered in a single merged lesson (they restate one idea)*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| Industry Demands: Beyond Traditional LLM Pipelines | 1h | ✅ Done |
| Agents as LLM-Defined Control Flow | 15m | ✅ Done |
| Fixed vs Dynamic Control Flow | 30m | ✅ Done |
| LangGraph vs Other Orchestration Tools | 30m | ✅ Done |

### Module 2 Sub-Topics — LangGraph Architecture and Ecosystem
*45m · 2 Web Modules*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| LangGraph in the LangChain Ecosystem | 15m | ⬜ Not started |
| LangGraph Architecture | 30m | ⬜ Not started |

### Module 3 Sub-Topics — Getting Started with LangGraph: Building Your First AI Workflow Graph
*30m · 1 Web Module*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| What are Agents? | 10m | ⬜ Not started |

### Module 4 Sub-Topics — Prebuilt Agents in LangGraph
*1h · 3 Web Modules*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| Hooks | 30m | ⬜ Not started |
| langgraph-prebuilt | 15m | ⬜ Not started |
| langgraph-supervisor | 15m | ⬜ Not started |

### Module 5 Sub-Topics — Designing Custom Workflows with LangGraph
*4h 20m · 7 Web Modules — the core module of the course*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| Building the Workflow | 40m | ⬜ Not started |
| Persistence | 40m | ⬜ Not started |
| Time Travel | 30m | ⬜ Not started |
| Tool | 45m | ⬜ Not started |
| Interrupts | 45m | ⬜ Not started |
| Memory | 50m | ⬜ Not started |
| LangGraph APIs — Functional and Graph API | 10m | ⬜ Not started |

### Module 6 Sub-Topics — Dynamic AI Graphs: Combining Subgraphs and Streaming
*1h 30m · 3 Web Modules*

| Sub-Topic | Duration | Status |
|-----------|----------|--------|
| Overview | 30m | ⬜ Not started |
| SubGraphs | 30m | ⬜ Not started |
| Streaming | 30m | ⬜ Not started |

---

## Current Topic
**Module 1 complete.** Next: **Module 2 — LangGraph Architecture and Ecosystem** (45m). Expect heavy overlap with Module 1's "vs LangChain" material — compress against the Concepts log in `_context.md` instead of re-teaching.

---

## Completed Topics

### Module 1 — The Rise of Agentic Workflows and the Emergence of LangGraph
All 4 sub-topics merged into `01-module1-rise-of-agentic-workflows.md` (the source material restates one idea four times).
- **Control flow = who decides what runs next** — the single organizing idea of the module
- Chain = fixed sequence, identical every invocation (predictable but rigid)
- **Agent ≈ control flow defined by the LLM** — path differs per invocation; costs unpredictability, debuggability, guardrails
- **Agency ↔ reliability tradeoff curve** across Code → Chain → Fully Autonomous
- **LangGraph = partial control flow definition** — dev defines critical steps, LLM decides non-critical transitions ("bends the reliability curve"); the module's one real innovation
- Six mechanisms mapped to concrete chain failures: stateful execution, conditional routing, human-in-the-loop, subgraphs, checkpointing, streaming — these are the rest of the course
- Graphs can loop back; pipelines structurally cannot
- LangGraph vs LangChain (orchestrates, doesn't replace) and vs CrewAI/AutoGen/Haystack (flagged as low-value/stale-prone)

---

## Running Project: TBD — named and scaffolded at Module 3

Fresh project, built graph-first (the LangChain course's Developer Documentation Assistant is **not** carried over — LangGraph's state/graph model is different enough that porting it would fight the material).

Modules 1–2 are conceptual/architectural, so the first real code lands in **Module 3 — Building Your First AI Workflow Graph**. Until then lessons carry illustrative snippets only, and `_context.md` keeps an empty project block.

The project grows one version per topic (v0.1 → v0.2 → …), and its full current source lives in `start learning/_context.md`. Each lesson ends with the complete updated file so any single lesson is self-contained.

---

## Workflow Reference
1. User drops screenshots into `/screenshots` and says `"Let's start the next topic: [Topic Name]"`.
2. Tutor reads `start learning/_context.md` (single file — concepts log + latest project code).
3. Tutor reads screenshots, deduces core concepts, rewrites lesson in developer-friendly style.
4. Lesson saved as `start learning/{module}-{slug}.md` (multi-topic modules use `01a-`, `01b-`, … prefixes). Lesson opens with "Where we left off" and ends with full updated project code.
5. Tutor updates `_context.md`: appends concept to log, replaces project code block, updates "Next Up".
6. This file (CLAUDE.md) is updated: sub-topic status → ✅ Done, module status rolled up, current topic updated.

---

## Notes & Decisions
- **⚠️ THE JOB IS: rewrite the course, in easy-to-learn form.** Standing instruction from the learner, clarified at Module 1. This means **teach the material properly and completely** — not summarize it, not produce an executive digest, and **not** write meta-commentary about what was trimmed. Write it the way a good teacher would if they were re-authoring the course from scratch for a developer.
  - **Full coverage.** Every real concept from the screenshots gets taught. Nothing important is dropped just because the source stated it badly.
  - **Build up, don't dump.** Problem first → concept → why it matters → how it's solved. Ideas arrive in an order where each one is ready to be understood.
  - **Simple language, short sentences, concrete examples.** Prefer one running example carried through the lesson over many disconnected ones.
  - **Remove repetition, not content.** The source restates the same point many times — say it once, well, in the right place. That is the *only* kind of cutting to do.
  - **No "what I cut and why" sections.** A course doesn't talk about its own editing.
  - End with a short self-check list so the learner can test understanding.
- Persona dialogue scenes from the source (Ravi, Riya/Karan, etc.) are **useful teaching devices** — keep them as concrete worked examples rather than discarding them as filler.
- Prefer tables and scannable blocks over prose walls; text diagrams over long descriptions. Flag genuinely low-value content honestly (e.g. stale competitor matrices) rather than drilling it.
- Lesson files use practical Python snippets, architectural diagrams (text-based), and real-world analogies drawn from the learner's existing Node.js/GraphQL/Claude/LangChain background.
- Course material is rewritten from scratch — original confusing phrasing is not preserved.
- Where a LangGraph concept has a LangChain equivalent already learned, lead with the contrast ("this is `RunnableLambda`, but the node can loop back") instead of teaching it cold.
- Module 4 (`langgraph-prebuilt`, `langgraph-supervisor`) revisits `create_react_agent`, which was already used as a black box in LangChain Module 8 — teach it as "here's what was actually running", not as a new tool.
- `.env` is **gitignored** in this repo (unlike the LangChain course repo, where the HuggingFace token got committed). Secrets go in `start learning/.env`; the tracked template is `start learning/.env.example`.
