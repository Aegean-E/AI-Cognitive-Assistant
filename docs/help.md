# User Guide & Command Reference

Welcome to your AI Desktop Assistant. This system is designed for **autonomous research** and **long-term collaboration**.

## 1. How it Works
The AI operates in loops. Even when you are not talking to it, it is thinking.

### The Cognitive Loop
1.  **Observation:** Netzach watches the chat and logs. If nothing happens, it signals "Boredom".
2.  **Decision:** Tiferet (The Decider) receives signals.
    *   If **Bored**: It might trigger Chokmah to read a random document or Da'at to synthesize a new theory.
    *   If **Overloaded**: It triggers Hod to summarize memories.
    *   If **User Speaks**: It prioritizes a response.
3.  **Action:** The AI uses tools (Search, Calculator, Code, Physics) or generates text.
4.  **Reflection:** Hod analyzes the action. Did it hallucinate? If so, it creates a "Refuted Belief" memory.

### Da'at: The Knowledge Engine
The system now actively builds a **Knowledge Graph**.
*   **Triples:** It extracts facts like `(NADH, inhibits, SIRT1)`.
*   **Hypotheses:** It looks for "holes" in its graph and creates **CURIOSITY_GAPS** (e.g., "Does NADH also inhibit SIRT3?").
*   **Synthesis:** It tries to map logic from one domain to another (e.g., applying a biological mechanism to a software architecture problem).

## 2. Modes of Operation
*   **Chat Mode:** The AI focuses on you. Telegram bridge is active. Background daydreaming is paused to save resources.
*   **Daydream Mode:** The AI is autonomous. It reads documents, consolidates memories, and pursues its own goals.

## 3. Memory System
Memories are stored in a vector database.
*   **IDENTITY:** Facts about you or the AI.
*   **FACT:** Objective truths from documents.
*   **GOAL:** Active objectives. The AI uses **HTN Planning** to break these down into sub-tasks.
*   **BELIEF:** Hypotheses (subject to verification).
*   **RULE:** Learned strategies or user-defined constraints.

## 4. Command Reference

### System Control
*   `/status` - View system health, active goals, and current "Thought".
*   `/stop` - Emergency halt for current processing.
*   `/terminate_desktop` - Close the application.
*   `/exitchatmode` - Resume autonomous daydreaming.

### Memory & Knowledge
*   `/memories` - List active memories.
*   `/metamemories` - View the "Stream of Consciousness" logs.
*   `/consolidate` - Force Binah to merge duplicate memories.
*   `/verify` - Force Hod to check facts against documents.
*   `/verifyall` - Deep verification scan (slow).
*   `/removesummaries` - Clear session logs (frees up context).
*   `/consolidatesummaries` - Compress old logs into a "Standing Summary".

### Decider Control
*   `/decider loop` - Start the autonomous loop.
*   `/decider daydream` - Trigger a single insight cycle.
*   `/decider verify` - Trigger a verification batch.

### Document Management
*   `/documents` - List indexed files.
*   `/doccontent "filename"` - Read a document's text.
*   `/removedoc "filename"` - Delete a document.

## 5. Advanced Tools (Malkuth)
The AI can use these tools autonomously, or you can force them in chat:

*   **Python Code:** `[CODE: x=5; print(x*2)]`
    *   Executes Python code in a persistent session. Variables are remembered.
*   **Physics Intuition:** `[EXECUTE: PHYSICS, "Calculate the entropy of..."]`
    *   Performs dimensional analysis and Fermi estimation before calculating.
*   **Causal Inference:** `[EXECUTE: CAUSAL, "Insulin", "Weight", "Metabolic Syndrome"]`
    *   Uses DoWhy to estimate causal effects and p-values (requires `dowhy` and `pandas`).
*   **File Writing:** `[WRITE_FILE: notes.txt, content]`
    *   Writes text to the `works/` directory.

## 6. Advanced Features
*   **Image Analysis:** Drag and drop images or send them via Telegram. The AI uses Vision models to analyze them.
*   **Cross-Document Search:** The system searches for diverse chunks across multiple files to find hidden connections.
*   **Self-Correction:** If the AI realizes it was wrong, it creates a `REFUTED_BELIEF`. The **Meta-Learner** then analyzes this failure to improve future prompts.
*   **Epigenetics:** The AI evolves its own system instructions over time based on what works.

## 7. Tips for Best Results
*   **Upload Documents:** The AI needs raw material. Upload PDFs/DOCX files to the `data/uploaded_docs` folder or via the UI.
*   **Set Goals:** Tell the AI "Your goal is to research X". It will use HTN planning to work on it autonomously.
*   **Correct It:** If the AI is wrong, tell it. It will update its memory and learn a new Rule.