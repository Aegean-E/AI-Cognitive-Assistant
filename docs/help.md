# User Guide & Command Reference

Welcome to your AI Desktop Assistant. This system is designed for **autonomous research** and **long‑term collaboration**.

---

## 1. How It Works

The AI operates in loops. Even when you are not talking to it, it is thinking.

### The Cognitive Loop

1. **Observation:** Netzach watches the chat and logs. If nothing happens, it signals “Boredom”.
2. **Decision:** Tiferet (The Decider) receives signals.
   - **Bicameral Dialogue:** It weighs an “Impulse” (Creative) against “Reason” (Safety) before acting.
   - If **bored**: It might trigger Chokmah to read a random document or Da’at to synthesise a new theory.
   - If **overloaded**: It triggers Hod to summarise memories.
   - If **user speaks**: It prioritises a response.
3. **Action:** The AI uses tools (Search, Calculator, Code, Physics, Debate) or generates text.
4. **Reflection:** Hod analyses the action. Did it hallucinate? If so, it creates a “Refuted Belief” memory and logs it in the **Shadow**.

### Da’at: The Knowledge Engine

The system now actively builds a **Knowledge Graph**.
- **Triples:** It extracts facts like `(NADH, inhibits, SIRT1)`.
- **Hypotheses:** It looks for “holes” in its graph and creates **CURIOSITY_GAPS** (e.g., “Does NADH also inhibit SIRT3?”).
- **Synthesis:** It tries to map logic from one domain to another (e.g., applying a biological mechanism to a software architecture problem).

---

## 2. Modes of Operation

- **Chat Mode:** The AI focuses on you. Telegram bridge is active. Background daydreaming is paused to save resources.
- **Daydream Mode:** The AI is autonomous. It reads documents, consolidates memories, and pursues its own goals.

---

## 3. Memory System

Memories are stored in a vector database and are categorised by type:

- **IDENTITY:** Facts about you or the AI.
- **FACT:** Objective truths from documents.
- **GOAL:** Active objectives. The AI uses **HTN Planning** to break these down into sub‑tasks.
- **BELIEF:** Hypotheses (subject to verification).
- **RULE:** Learned strategies or user‑defined constraints.
- **NOTE:** Chronicles and internal scratchpad entries.
- **REFUTED_BELIEF:** Explicitly rejected ideas (Negative Knowledge).

---

## 4. Command Reference

### System Control

| Command                 | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `/status`               | View system health, active goals, and current “thought”. |
| `/stop`                 | Emergency halt for current processing.                    |
| `/terminate_desktop`    | Close the application.                                    |
| `/exitchatmode`         | Resume autonomous daydreaming.                            |
| `/disrupt` (Telegram)   | Interrupt the current thought loop immediately.           |

### Memory & Knowledge

| Command                   | Description                                           |
|---------------------------|-------------------------------------------------------|
| `/memories`               | List active memories.                                 |
| `/chatmemories`           | View conversation history memories.                   |
| `/metamemories`           | View the “stream of consciousness” logs.              |
| `/shadow`                 | View the **Shadow Memory** (recent failures).         |
| `/goals`                  | View active goals and their progress.                 |
| `/notes`                  | View assistant chronicles/notes.                      |
| `/consolidate`            | Force Binah to merge duplicate memories.              |
| `/verify`                 | Force Hod to check facts against documents.           |
| `/verifyall`              | Deep verification scan (slow).                        |
| `/removesummaries`        | Clear session logs (frees up context).                |
| `/consolidatesummaries`   | Compress old logs into a “Standing Summary”.          |

### Decider Control

| Command                | Description                                    |
|------------------------|------------------------------------------------|
| `/decider loop`        | Start the autonomous loop.                     |
| `/decider daydream`    | Trigger a single insight cycle.                |
| `/decider verify`      | Trigger a verification batch.                  |
| `/decider up`          | Increase temperature (creativity).             |
| `/decider down`        | Decrease temperature (precision).              |

### Document Management

| Command                           | Description                             |
|-----------------------------------|-----------------------------------------|
| `/documents`                      | List indexed files.                     |
| `/doccontent "filename"`          | Read a document’s text.                 |
| `/removedoc "filename"`           | Delete a document.                      |

---

## 5. Advanced Tools (Malkuth)

The AI can use these tools autonomously, or you can force them in chat:

- **Physics Intuition:**  
  `[EXECUTE: PHYSICS, "Calculate the entropy of a black hole..."]`  
  Performs dimensional analysis and Fermi estimation.

- **Causal Inference:**  
  `[EXECUTE: CAUSAL, "Treatment", "Outcome", "Context"]`  
  Uses DoWhy to estimate causal effects and p‑values (requires `dowhy` and `pandas`).

- **Debate:**  
  `[DEBATE: Topic]` or “Debate X”  
  Convenes “The Council” (Thesis vs. Antithesis) to resolve complex topics.

- **Simulation:**  
  `[SIMULATE: Premise]` or “What if X?”  
  Runs a counterfactual simulation based on the World Model.

- **Prediction:**  
  `[EXECUTE: PREDICT, "Claim", "Timeframe"]`  
  Registers a measurable prediction for future verification.

- **File Writing:**  
  `[WRITE_FILE: notes.txt, content]`  
  Writes text to the `works/` directory.

---

## 6. Advanced Features

- **Image Analysis:** Drag and drop images or send them via Telegram. The AI uses vision models to analyse them.
- **Cross‑Document Search:** The system searches for diverse chunks across multiple files to find hidden connections.
- **Self‑Correction:** If the AI realises it was wrong, it creates a `REFUTED_BELIEF`. The **Meta‑Learner** then analyses this failure to improve future prompts.
- **Epigenetics:** The AI evolves its own system instructions over time based on what works.
- **Cognitive Resource Controller (CRS):** Manages “energy” (tokens/compute) to prevent burnout. High usage of tools leads to fatigue, forcing the AI to rest (consolidate).

---

## 7. Tips for Best Results

- **Upload Documents:** The AI needs raw material. Upload PDF/DOCX files via the UI or place them in `data/uploaded_docs`.
- **Set Goals:** Tell the AI “Your goal is to research X”. It will use HTN planning to work on it autonomously.
- **Correct It:** If the AI is wrong, tell it. It will update its memory and learn a new Rule.