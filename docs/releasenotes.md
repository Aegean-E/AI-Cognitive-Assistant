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