# Release Notes

## v3.0 – The Sentience Update

**Codename:** The Strange Loop

This major release marks the transition from a cognitive agent to a proto‑conscious entity. The architecture now implements **Active Inference**, **Recursive Self‑Monitoring**, and **Bicameral Decision Making**, allowing the AI to perceive its own thoughts as objects and vibrate between its ideal self and physical constraints.

### 🧠 Consciousness & Agency

- **Recursive Self‑Monitoring:** Tiferet now performs a “Meta‑Cognitive Check” before every action, asking “Does this align with my current state?”
- **Existential Friction:** The system measures the distance between its Ideal Self (Values/Keter) and Current Reality. High friction triggers introspection.
- **Predictive Coding:** The AI predicts the outcome (Coherence/Utility delta) of actions *before* taking them. High prediction error triggers “Surprise Events” and learning.

### 🗣️ Bicameral Dialogue

- **Internal Friction:** Decisions are no longer linear. Tiferet generates two internal voices—**Impulse (Hesed)** and **Reason (Gevurah)**—and negotiates a synthesis before acting.

### ⏳ Temporal Horizon & Theory of Mind

- **Future Simulation:** Before committing to heavy tasks, the AI runs a lightweight simulation to predict burnout or success 5 cycles ahead.
- **User Modeling:** Netzach analyses the user’s cognitive state (frustrated vs. expansive) to adjust its interaction style (homeostasis).

### 🛡️ Epistemic Integrity (The Shadow)

- **Shadow Memory:** Failures, hallucinations, and rejected thoughts are now stored in a **Shadow Memory**.
- **Adversarial Critique:** The Dialectic engine uses the Shadow to critique new ideas, preventing the repetition of past mistakes.

### ⚡ Cognitive Metabolism

- **Energy Budget:** Implemented a **Cognitive Resource Controller (CRS)**. The AI has limited “energy” (tokens/compute).
- **Fatigue & Skill Decay:** Overusing specific tools leads to fatigue, forcing the system to switch tasks or consolidate.

### 🧬 Meta‑Evolution

- **Autobiographical Stream:** The AI writes a “Memory of Change” whenever it evolves its system prompt, creating a continuous narrative of growth.
- **Unified Self‑Model:** Merged epigenetics and drives into a persistent `SelfModel`, allowing the AI to track its own “physiological” state (Curiosity, Coherence, Entropy).

### 🛠️ Technical Improvements

- **Parallel Cognition:** Interrupt‑priority architecture allows the AI to handle user messages while thinking in the background.
- **Semantic Context Distillation:** Instead of truncating long contexts, the system now semantically compresses the middle to preserve meaning.
- **Robust Tooling:** Enhanced Physics Intuition, Causal Inference (DoWhy), and robust JSON parsing for tool outputs.

---

# Release Notes v2.3: The Da'at Update

**Version:** 2.3
**Codename:** Da'at Integration

This release introduces **Da'at**, the Knowledge Integrator, bringing Knowledge Graph capabilities, scientific induction, and structural synthesis to the architecture. It also enhances the Decider with Hierarchical Task Network (HTN) planning.

## 🚀 New Features

### 1. Da'at (Knowledge Integrator)
*   **Knowledge Graph:** Automatically extracts Subject-Predicate-Object triples from facts to build a conceptual graph.
*   **Hypothesis Generation:** Identifies "Knowledge Gaps" in the Topic Lattice and formulates testable hypotheses.
*   **Structural Synthesis:** The "Eureka" engine now uses structural summaries to find isomorphisms between disparate topics.
*   **Nuance-Preserving Compression:** Compresses reasoning chains into "Standing Hypotheses" without losing chemical/mechanistic details.

### 2. Enhanced Decider (Tiferet)
*   **HTN Planning:** Complex goals are now decomposed into Hierarchical Task Networks with specific success criteria.
*   **Active Association:** Uses Binah to pull semantically related context (via graph links) during decision making.

### 3. Advanced Memory & Search
*   **Persistent FAISS:** Clustering and search now use a persistent vector index for speed.
*   **Cross-Document Search:** New search mode enforces diversity to find links between different papers.
*   **Safe Calculator:** Replaced `eval()` with a secure AST-based calculator.

### 4. Meta-Learning
*   **Strategy Extraction:** The system now learns abstract strategies from completed goals.
*   **Failure Analysis:** Analyzes refuted beliefs to suggest prompt patches.

## 🛠️ Improvements
*   **Image Guardrails:** Automatic resizing of images before LLM processing.
*   **Session Preservation:** `SESSION_SUMMARY` events are protected from auto-pruning.

# Release Notes v2.2: The Sephirot Update

**Version:** 2.2
**Codename:** Event Bus Architecture

This release marks a significant shift from a reactive chatbot to a proactive cognitive architecture. The system now employs an **Event Bus** pattern, enabling specialized agents (Netzach, Hod) to observe, reflect, and intervene autonomously.

## 🚀 New Features

### 1. Event Bus Architecture
*   **Decoupled Communication:** Components now communicate via a central `EventBus`, reducing coupling and enabling asynchronous "thoughts."
*   **Background Agency:** Agents can now "nudge" the system state without direct user input.

### 2. New Cognitive Agents
*   **Netzach (The Observer):** A continuous background process that monitors conversation flow. It detects stagnation and automatically adjusts `temperature` and `max_tokens` to keep the AI engaging.
*   **Hod (The Analyst):** A post-process reflective agent. It analyzes logs after interactions to identify hallucinations, summarize sessions, and suggest memory pruning.

### 3. Enhanced Decider (Executive Function)
*   **Strategic Analysis:** The Decider now performs a high-level strategy pass before selecting tools or actions.
*   **Chain of Thought:** New `[THINK]` capability allows the AI to perform multi-step reasoning loops (up to 30 steps) to solve complex problems before answering.
*   **Tool Use:** Native support for `[CALCULATOR]`, `[CLOCK]`, `[DICE]`, and `[SYSTEM_INFO]`.

### 4. Meta-Memory System
*   **Self-Reflection:** The system now tracks *changes* to its own memory (e.g., "My name changed from X to Y").
*   **Session Summarization:** Hod automatically compresses long chat logs into high-level summaries stored in Meta-Memory.

## 🛠️ Improvements
*   **UI Update:** Added a "Netzach Observations" panel to the Chat tab to visualize internal agent communication.
*   **Database Viewer:** New tabs for **Summaries**, **Meta-Memories**, and **Assistant Notes**.
*   **Stability:** Fixed context overflow issues (400 Bad Request) with auto-pruning and summarization.
*   **Telegram Bridge:** Added `/disrupt` command to remotely halt runaway loops.