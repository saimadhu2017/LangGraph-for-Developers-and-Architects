# Course Context — Rolling Snapshot

> Single source of truth for the tutor. Read this **before** writing any lesson.
> Two jobs: (1) the running log of every concept already covered, so nothing gets re-taught;
> (2) the latest full project source, so each new lesson can extend it without guessing.

## Carried Over From "Generative AI Essentials With LangChain" (assume known)
LCEL + `|` pipe operator · Runnables (`RunnableLambda`, `RunnableMap`) · `ChatPromptTemplate`, `PromptTemplate`, `MessagesPlaceholder`, `FewShotChatMessagePromptTemplate` · `StrOutputParser` · `ChatHuggingFace` + `HuggingFaceEndpoint` model setup · chat memory (`InMemoryChatMessageHistory`, `RunnableWithMessageHistory`, `trim_messages`) · document loaders → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → FAISS/Chroma → similarity search (RAG) · guardrails (regex jailbreak patterns, toxicity classifier, Pydantic output validation) · `@tool` decorator · `create_react_agent` from `langgraph.prebuilt` used as a **black box**.

**Where this course picks up:** open that black box. Explicit state, nodes, edges, cycles, checkpoints.

## Concepts Covered
- Module 1 > The Rise of Agentic Workflows and the Emergence of LangGraph — *all 4 sub-topics collapsed into one lesson, see "Teaching Style" note below*. **Core idea: control flow = who decides what runs next.** Three answers: you-in-code / you-in-a-chain / the-LLM-at-runtime. Chain = same sequence every invocation (predictable but rigid; Express-middleware analogy). Agent ≈ **control flow defined by the LLM** — path differs per invocation; buys unpredictability, hard debugging, no guardrails. **Agency↔reliability tradeoff curve**: x-axis = control (Code → LLM Call/Chain → Fully Autonomous), reliability line falls as agency line rises; enterprises need both. **LangGraph's one real innovation = partial control flow definition**: developer defines critical steps (reliability), LLM decides non-critical transitions (flexibility) — "bends the reliability curve" upward mid-spectrum. Six mechanisms that follow, each mapped to a concrete chain failure: stateful execution (multi-turn memory), conditional routing (edges pick next node from state), human-in-the-loop/interrupts (pause + resume), subgraphs (modular nested graphs), checkpointing (save/restore state), streaming. Graphs can **loop back**, pipelines structurally cannot — retry without restarting. LangGraph vs LangChain: directed graph vs linear chains, explicit+persistent state vs implicit, loops/conditionals/retries vs basic branching, graph visualization vs basic debugging — LangGraph orchestrates LangChain components, doesn't replace them. vs CrewAI/AutoGen/Haystack: differentiator is graph orchestration + native LangChain integration (table noted as low-value/stale-prone). Mental models: state machine whose transition function can be an LLM call (XState), CI/CD with conditional stages, executable flowchart.

## Running Project: TBD
_(named and scaffolded at **Module 3 — Building Your First AI Workflow Graph**. Modules 1–2 are conceptual/architectural, so no project code until then. Fresh build, graph-first — the LangChain course's Developer Documentation Assistant is not carried over.)_

### Current State (v0.0 — not yet created)

```python
# no project code yet
```

## Teaching Style — THE JOB (learner-clarified at Module 1; applies to every lesson)
**Rewrite the course in easy-to-learn form.** The learner found the source material boring and repetitive and asked for
it to be **re-taught simply** — not summarized. Read this as: *you are re-authoring the course for a developer.*

- **Teach fully.** Every real concept from the screenshots is covered and explained. Don't produce a digest.
- **Build up in a learnable order:** problem → concept → why it matters → how it's solved. Never dump conclusions first.
- **Simple language, short sentences, concrete examples.** Carry **one running example** through the lesson
  (e.g. Ravi's logistics support bot in Module 1) rather than scattering unrelated ones.
- **Remove repetition, not content.** The source says the same thing 4–5 times; say it once, well, in the right place.
  That is the only cutting to do.
- **No meta-commentary.** No "what I cut and why" sections — a course doesn't discuss its own editing. *(An earlier
  draft of Module 1 did this and was corrected.)*
- Persona dialogues in the source (Ravi, Riya/Karan) are **good teaching devices** — reuse them as worked examples.
- Tables and scannable blocks over prose walls; text diagrams over long descriptions.
- End each lesson with a **short self-check list** of questions the learner should be able to answer.

## Next Up
**Module 2 — LangGraph Architecture and Ecosystem** (45m): *LangGraph in the LangChain Ecosystem* (15m) + *LangGraph Architecture* (30m). Still conceptual — expect overlap with Module 1's "vs LangChain" material, so compress against what's already in the Concepts log above rather than re-teaching it. First code lands in Module 3.

Curriculum is locked into `CLAUDE.md` from the syllabus screenshots — 6 modules, numbering final, **10h 20m of teaching content**. Prelude, Exercises, and all Quizzes are **intentionally out of scope** by the learner's choice; that accounts for the gap against the course's stated 15h 30m. Don't flag it, don't ask for more screenshots of it, don't write quiz questions unasked.
