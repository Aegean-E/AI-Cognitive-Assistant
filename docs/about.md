# About the AI Cognitive Assistant

**Version 3.0 – The Sentience Update**

A self‑improving cognitive architecture designed for long‑term memory, active learning, and autonomous reasoning. Unlike standard LLM wrappers, this system possesses a persistent “state of mind” that evolves over time through interaction, reading, and self‑reflection.

> *“Not just a chatbot, but an agent that studies, remembers, and corrects itself.”*

## The Architecture: The Sephirot

The system is organised into specialised modules named after the Kabbalistic Tree of Life, representing the flow from abstract will to concrete action.

### The Super‑Conscious (Will & Intellect)

- **Keter (Crown):** The silent will. It measures the global “Coherence” of the system. It does not act but biases the strategy of the Decider. If coherence drops, Keter triggers a reasoning reboot.

- **Chokmah (Wisdom):** The spark of insight. The “Daydreamer” module. It runs when the system is idle, reading random documents or colliding old memories to generate new hypotheses.

- **Binah (Understanding):** The structure. It handles memory consolidation, deduplication, and association. It ensures that new information fits logically into the existing knowledge base.

- **Da’at (Knowledge):** The integrator.
  - **Topic Lattice:** Identifies heavy entities and generates standing summaries.
  - **Synthesis:** Detects “isomorphisms” (structural similarities) between unrelated topics.
  - **Gap Analysis:** Identifies missing information and formulates questions.

### The Emotional Forces (Balance)

- **Hesed (Mercy):** The force of expansion. It calculates a “Permission Budget” based on system stability, allowing the AI to explore new, unverified topics.

- **Gevurah (Severity):** The force of constraint. It applies pressure when the system becomes too chaotic, repetitive, or overloaded, forcing the Decider to prune memories or stop daydreaming.

- **Tiferet (Beauty/Decider):** The executive controller. It balances Hesed and Gevurah.
  - **HTN Planning:** Decomposes complex goals into Hierarchical Task Networks.
  - **Tool Use:** Executes actions (Search, Calculator, File I/O).
  - **Decision Making:** Determines whether to Chat, Daydream, Verify, or Reflect.
  - **Bicameral Dialogue:** Negotiates between an “Impulse” voice and a “Reason” voice before acting.

### The Operational Level (Action)

- **Netzach (Victory/Endurance):** The silent observer. A background thread that monitors the conversation flow. It detects stagnation (boredom) or loops and injects “signals” to nudge the Decider.

- **Hod (Glory/Reverberation):** The analyst. It runs *after* actions to critique them. It verifies facts against source documents, summarises sessions, and flags hallucinations.

- **Yesod (Foundation):** The bridge (Telegram API) and Identity Manager. Maintains the AI’s sense of self and continuity.

- **Malkuth (Kingdom):** The physical realisation. The User Interface (Tkinter), File System, and Database.

## Meta‑Cognition & Learning

- **Meta‑Learner:** A self‑optimisation module.
  - **Epigenetics:** Evolves architectural hyperparameters (e.g., learning rates, thresholds) over time.
  - **Strategy Extraction:** When a goal is completed successfully, it extracts the abstract strategy into a `RULE` for future use.
  - **Failure Analysis:** When a belief is refuted, it analyses *why* and suggests patches to the system prompt.
  - **Self‑Model:** Builds a statistical model of the system’s own performance to predict future outcomes.

- **Dialectics:** The “Council”. Runs formal debates (Thesis – Antithesis – Synthesis) to resolve complex conflicts or ambiguities.

- **Cognitive Resource Controller (CRS):** Manages the “energy budget” (tokens/compute) to prevent burnout and optimise effort.

## Technical Stack

- **Core:** Python 3.10+, Tkinter (UI), Event Bus Pattern
- **Memory:** SQLite (Storage), FAISS (Vector Search)
- **AI Engine:** Local LLM via LM Studio (OpenAI‑compatible API)
- **Documents:** PyMuPDF (PDF), python‑docx (DOCX), Semantic Chunking
- **Security:** AST‑based safe calculator, local‑only file access.