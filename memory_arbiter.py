from typing import Optional, Callable, Any

from memory import MemoryStore
from meta_memory import MetaMemoryStore


TYPE_PRECEDENCE = {
    "PERMISSION": 0,  # Highest priority: only user can grant
    "RULE": 1,        # Rules from/for assistant
    "IDENTITY": 2,    # Who is user/assistant
    "REFUTED_BELIEF": 3, # Explicitly rejected ideas (Negative Memory)
    "COMPLETED_GOAL": 3, # Finished objectives (Prevents re-generation)
    "PREFERENCE": 4,  # Likes/dislikes
    "GOAL": 5,        # Aims/desires
    "FACT": 6,        # Objective truths
    "BELIEF": 7,      # Opinions/convictions (lowest priority)
    "NOTE": 1,        # Assistant Notes (High priority, internal)
    "CURIOSITY_GAP": 1, # High priority: Questions for the user
}

CONFIDENCE_MIN = {
    "PERMISSION": 0.85,  # Very high: explicit user permission (only user can grant)
    "RULE": 0.9,         # Very high: guidelines for assistant behavior
    "IDENTITY": 0.8,     # High: identity claims (who are you)
    "REFUTED_BELIEF": 0.9, # Very high: explicit rejection
    "COMPLETED_GOAL": 0.9, # Very high: explicit completion
    "PREFERENCE": 0.6,   # Medium-high: preferences/likes/dislikes
    "GOAL": 0.7,         # High: goals/desires
    "FACT": 0.7,         # High: factual assertions
    "BELIEF": 0.5,       # Medium: beliefs/opinions/convictions
    "NOTE": 0.9,         # Very high: manual notes
    "CURIOSITY_GAP": 0.9, # Very high: system generated
}


class MemoryArbiter:
    """
    Autonomous bridge between ReasoningStore and MemoryStore.

    - Does NOT reason
    - Does NOT decide truth
    - Only enforces admission rules
    """

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, embed_fn: Optional[Callable] = None, log_fn: Callable[[str], None] = print, event_bus: Any = None):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.embed_fn = embed_fn
        self.log = log_fn
        self.event_bus = event_bus

    # --------------------------
    # Public API
    # --------------------------
    def consider(
        self,
        text: str,
        mem_type: str,
        confidence: float,
        subject: str = "User",
        source: str = "reasoning"
    ) -> Optional[int]:
        """
        Decide whether to promote reasoning into memory.

        Returns memory_id if stored, None otherwise.
        """

        mem_type = mem_type.upper()
        self.log(f"🔍 [Arbiter] Processing {mem_type} for {subject} (Conf: {confidence}):\n    \"{text}\"")

        if mem_type not in TYPE_PRECEDENCE:
            self.log(f"❌ [Arbiter] Type '{mem_type}' not in precedence table")
            return None

        # 0️⃣ Filter out meta-actions/refutations saved as text
        if text.strip().startswith("[Refuting Belief") or text.strip().startswith("Refuting Belief"):
            self.log(f"⛔ [Arbiter] BLOCKING meta-action text: \"{text[:50]}...\"")
            return None

        # 1️⃣ Confidence gate
        min_conf = CONFIDENCE_MIN[mem_type]
        if confidence < min_conf:
            self.log(f"❌ [Arbiter] Confidence gate failed: {confidence} < {min_conf} (required for {mem_type})")
            return None

        self.log(f"✅ [Arbiter] Passed confidence gate: {confidence} >= {min_conf}")

        # 2️⃣ Identity + version chaining
        identity = self.memory_store.compute_identity(text, mem_type=mem_type)
        previous_versions = self.memory_store.get_by_identity(identity)
        parent_id = previous_versions[-1]["id"] if previous_versions else None

        # 2.5️⃣ Exact duplicate guard (same text as latest version)
        if previous_versions:
            last_text = previous_versions[-1]["text"].strip()
            if last_text == text.strip():
                self.log(f"❌ [Arbiter] Exact duplicate detected (ID: {previous_versions[-1]['id']})")
                return None

        self.log(f"✅ [Arbiter] No duplicates. Previous versions: {len(previous_versions)}, parent_id: {parent_id}")

        # 2.7️⃣ Negative Knowledge Enforcement (Invariant 7)
        # Generate embedding early for checks
        embedding = None
        if self.embed_fn:
            embedding = self.embed_fn(text)

        if mem_type not in ("REFUTED_BELIEF", "NOTE"):
            # Check via Embedding
            if embedding is not None:
                similar_mems = self.memory_store.search(embedding, limit=5)
                for mid, mtype, msubj, mtext, sim in similar_mems:
                    if mtype == "REFUTED_BELIEF" and sim > 0.85:
                         self.log(f"⛔ [Arbiter] BLOCKING memory: Contradicts REFUTED_BELIEF (Sim: {sim:.2f})\n    New: \"{text}\"\n    Refuted: \"{mtext}\"")
                         return None
            
            # Check via Text (Fallback/Exact)
            refuted_list = self.memory_store.get_active_by_type("REFUTED_BELIEF")
            text_lower = text.strip().lower()
            for rid, rsubj, rtext, rsource in refuted_list:
                if "[REFUTED:" in rtext:
                    claim = rtext.split("[REFUTED:")[0].strip().lower()
                    if text_lower == claim or (len(claim) > 10 and claim in text_lower):
                        self.log(f"⛔ [Arbiter] BLOCKING memory: Text match with REFUTED_BELIEF.\n    New: \"{text}\"\n    Refuted: \"{rtext}\"")
                        return None

        # 3️⃣ Conflict detection (exact, conservative)
        
        # 2.6️⃣ Cross-Subject Identity Conflict Guard
        # Prevent Assistant from claiming User's name/identity and vice versa
        # Check this for IDENTITY type OR if the text looks like an identity claim
        extracted_val = self._extract_value(text)
        if extracted_val and (mem_type == "IDENTITY" or "name is" in text.lower()):
            # Check against all active identities (and FACTs that act like identities)
            active_identities = self.memory_store.get_active_by_type("IDENTITY")
            active_facts = self.memory_store.get_active_by_type("FACT")
            
            for _, subj, txt, _ in (active_identities + active_facts):
                # If subject is different (e.g. User vs Assistant)
                if subj.lower() != subject.lower():
                    existing_val = self._extract_value(txt)
                    if existing_val and existing_val.lower() == extracted_val.lower():
                        self.log(f"❌ [Arbiter] Identity conflict: '{extracted_val}' is already assigned to {subj}")
                        return None

        conflicts = self.memory_store.find_conflicts_exact(text)
        self.log(f"🔍 [Arbiter] Found {len(conflicts)} conflicts")

        # 4️⃣ Precedence dampening
        adjusted_confidence = confidence
        for c in conflicts:
            if TYPE_PRECEDENCE[c["type"]] < TYPE_PRECEDENCE[mem_type]:
                self.log(f"⛔ [Arbiter] BLOCKING memory due to conflict with higher precedence {c['type']} (ID: {c['id']}):\n    Conflict: \"{c['text']}\"")
                return None

        # 5️⃣ Append-only write (versioned)
        self.log(f"✅ [Arbiter] Saving memory with adjusted_confidence={adjusted_confidence}")
        
        # Use consistent timestamp for both memory and meta-memory

        import time
        created_at = int(time.time())
        from datetime import datetime
        timestamp = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")

        memory_id = self.memory_store.add_entry(
            text=text,
            mem_type=mem_type,
            subject=subject,
            confidence=adjusted_confidence,
            source=source,
            identity=identity,
            parent_id=parent_id,
            conflicts=[c["id"] for c in conflicts],
            created_at=created_at,
            embedding=embedding,
        )

        # If we just learned a HIGH CONFIDENCE Fact or Insight (Epiphany)
        # and it wasn't just explicitly told to us by the User (Source != User)
        if memory_id is not None and confidence > 0.92 and source != "User":
            
            # 1. Check importance (heuristic)
            is_exciting = any(w in text.lower() for w in ["breakthrough", "critical", "proven", "refuted", "connection found"])
            
            if is_exciting and self.event_bus:
                self.log(f"⚡ Epiphany detected! Signaling to user.")
                self.event_bus.publish(
                    "AI_SPEAK", 
                    f"💡 Insight: I just realized that {text} (Confidence: {confidence})"
                )

        # 6️⃣ Create meta-memory about this new memory creation
        if self.meta_memory_store and memory_id:
            # Extract the current value
            value = self._extract_value(text)
            
            # Check for previous version to show change
            old_value = None
            if previous_versions:
                old_value = self._extract_value(previous_versions[-1]["text"])

            # Create human-readable meta-memory
            if mem_type == 'IDENTITY':
                if 'name is' in text.lower():
                    if old_value:
                        meta_text = f"{subject} name changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} name set to {value} on {timestamp}"
                elif 'lives in' in text.lower():
                    if old_value:
                        meta_text = f"{subject} location changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} location set to {value} on {timestamp}"
                else:
                    if old_value:
                        meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
                    else:
                        meta_text = f"{subject} {mem_type.lower()} recorded: '{value}' on {timestamp}"
            elif old_value:
                meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
            else:
                # Default for new recordings
                type_labels = {
                    'PREFERENCE': 'preference',
                    'GOAL': 'goal',
                    'FACT': 'fact',
                    'RULE': 'rule',
                    'PERMISSION': 'permission',
                    'BELIEF': 'belief'
                }
                label = type_labels.get(mem_type, mem_type.lower())
                meta_text = f"{subject} {label} recorded: '{value}' on {timestamp}"

            self.meta_memory_store.add_meta_memory(
                event_type="VERSION_UPDATE" if old_value else "MEMORY_CREATED",
                memory_type=mem_type,
                subject=subject,
                text=meta_text,
                old_id=previous_versions[-1]["id"] if previous_versions else None,
                new_id=memory_id,
                old_value=old_value,
                new_value=value,
                metadata={"confidence": adjusted_confidence, "source": source}
            )
            self.log(f"      🧠 Meta-memory: {meta_text}")

        return memory_id

    @staticmethod
    def _extract_value(text: str) -> str:
        """Extract the value from memory text."""
        text = text.strip()
        patterns = [" is ", " lives in ", " works at ", " wants to ", " prefers ", " loves ", " allowed ", " granted "]
        for pattern in patterns:
            if pattern in text.lower():
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return text
