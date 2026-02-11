import os
import threading
import time
import random
import re
import ast
import operator
from datetime import datetime
from typing import Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import AI components
from document_store_faiss import FaissDocumentStore
from document_processor import DocumentProcessor
from lm import compute_embedding, run_local_lm
from bridges.internet_bridge import InternetBridge
from event_bus import EventBus
from treeoflife.netzach import ContinuousObserver
from memory import MemoryStore
from meta_memory import MetaMemoryStore
from reasoning import ReasoningStore
from memory_arbiter import MemoryArbiter
from treeoflife.binah import Binah
from treeoflife.chokmah import Chokmah
from treeoflife.tiferet import Decider
from treeoflife.hod import Hod as HodAgent
from treeoflife.daat import Daat
from treeoflife.keter import Keter
from treeoflife.hesed_gevurah import Hesed, Gevurah
from treeoflife.malkuth import Malkuth
from meta_learner import MetaLearner
from dialectics import Dialectics

class AICore:
    """
    Core AI Engine.
    Manages the lifecycle and interaction of all cognitive components.
    Decoupled from UI implementation details.
    """
    def __init__(
        self,
        settings_provider: Callable[[], Dict],
        log_fn: Callable[[str], None],
        chat_fn: Callable[[str, str], None],
        status_callback: Callable[[str], None],
        telegram_status_callback: Callable[[str], None],
        ui_refresh_callback: Callable[[Optional[str]], None],
        get_chat_history_fn: Callable[[], List[Dict]],
        get_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        get_status_text_fn: Callable[[], str],
        update_settings_fn: Callable[[Dict], None],
        stop_check_fn: Callable[[], bool],
        enable_loop_fn: Callable[[], None],
        stop_daydream_fn: Callable[[], None],
        sync_journal_fn: Callable[[], None]
    ):
        self.get_settings = settings_provider
        self.log = log_fn
        self.chat_fn = chat_fn
        self.status_callback = status_callback
        self.telegram_status_callback = telegram_status_callback
        self.ui_refresh_callback = ui_refresh_callback
        self.get_chat_history = get_chat_history_fn
        self.get_logs = get_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        self.get_status_text = get_status_text_fn
        self.update_settings = update_settings_fn
        self.stop_check = stop_check_fn
        self.enable_loop_fn = enable_loop_fn
        self.stop_daydream_fn = stop_daydream_fn
        self.sync_journal_fn = sync_journal_fn

        # Components
        self.memory_store = None
        self.meta_memory_store = None
        self.document_store = None
        self.reasoning_store = None
        self.arbiter = None
        self.binah = None
        self.event_bus = None
        self.chokmah = None
        self.keter = None
        self.hesed = None
        self.gevurah = None
        self.hod_force = None
        self.netzach_force = None
        self.decider = None
        self.observer = None
        self.hod = None
        self.daat = None
        self.malkuth = None
        self.document_processor = None
        self.internet_bridge = None
        self.meta_learner = None
        self.dialectics = None

        self.init_brain()

    def get_embedding_fn(self):
        return lambda text: compute_embedding(
            text, 
            base_url=self.get_settings().get("base_url"), 
            embedding_model=self.get_settings().get("embedding_model")
        )

    def _safe_search(self, query, source):
        """Robust search with document ingestion."""
        if self.internet_bridge:
            content, filepath = self.internet_bridge.search(query, source)
            if filepath:
                try:
                    file_hash = self.document_store.compute_file_hash(filepath)
                    if not self.document_store.document_exists(file_hash):
                        chunks, page_count, file_type = self.document_processor.process_document(filepath)
                        self.document_store.add_document(file_hash=file_hash, filename=os.path.basename(filepath), file_type=file_type, file_size=os.path.getsize(filepath), page_count=page_count, chunks=chunks, upload_source="safe_search")
                        if self.ui_refresh_callback:
                            self.ui_refresh_callback('docs')
                except Exception as e:
                    self.log(f"⚠️ Ingestion failed for safe search: {e}")
            return content
        return "⚠️ Internet Bridge not initialized."

    def _safe_calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression without using eval()."""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.BitXor: operator.xor,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num): # Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")
                return operators[op](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")
                return operators[op](eval_node(node.operand))
            else:
                raise TypeError(f"Node type {type(node)} not supported")

        try:
            if len(expression) > 50:
                return "Error: Expression too long"
            tree = ast.parse(expression.strip(), mode='eval')
            return str(eval_node(tree.body))
        except Exception as e:
            return f"Calculation Error: {e}"

    def _physics_intuition(self, query: str) -> str:
        """
        Combines LLM Fermi Estimation with the Malkuth Causal Engine.
        """
        self.log(f"🧬 [Malkuth] Running Physical Intuition check on: {query}")
        
        # 1. Metacognitive Step: LLM performs Dimensional Analysis
        reasoning_prompt = (
            f"Analyze the physical scenario: '{query}'.\n"
            "1. Identify the core physical variables (e.g., Concentration, Volume, Flux).\n"
            "2. Perform a Fermi Estimation (Order of Magnitude check).\n"
            "3. Check for Dimensional Consistency (Units must match).\n"
            "4. Thermodynamic Guardrails: Check for violations of conservation laws (Energy/Mass).\n"
            "Output ONLY the reasoning and a final 'Estimated Value' with units."
        )
        
        estimation = run_local_lm(
            messages=[{"role": "user", "content": reasoning_prompt}],
            system_prompt="You are a Physics Intuition Engine.",
            temperature=0.3 # Low temp for precision
        )

        # 2. Grounding Step: If a Causal Model exists, verify the estimation
        if self.malkuth:
            verification = self.malkuth.verify_physical_possibility(query, estimation)
            estimation += f"\n\n[Malkuth Grounding]: {verification}"

        return estimation

    def _causal_inference(self, args: str) -> str:
        """
        Wrapper for Malkuth's Causal Inference.
        Args format: "treatment, outcome, context"
        """
        if not self.malkuth:
            return "Malkuth not initialized."
        
        parts = [p.strip() for p in args.split(",", 2)]
        if len(parts) < 3:
            return "Error: CAUSAL requires 'treatment, outcome, context'"
        
        return self.malkuth.run_causal_inference(parts[0], parts[1], parts[2])

    def _get_tools(self):
        """Define all available tools here."""
        return {
            # 1. Basic Tools
            "CLOCK": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CALCULATOR": lambda x: self._safe_calculate(x),
            "PHYSICS": lambda q: self._physics_intuition(q),
            "CAUSAL": lambda q: self._causal_inference(q),
            
            # 2. Internet Tools
            "SEARCH": lambda q: self._safe_search(q, "WEB"),
            "WIKI": lambda q: self._safe_search(q, "WIKIPEDIA"),
            "FIND_PAPER": lambda q: self._safe_search(q, "ARXIV"),
        }

    def _process_tool_calls(self, text: str) -> str:
        """
        Scans text for [EXECUTE: TOOL, ARGS] tags, runs them, 
        and replaces the tag with the result.
        """
        # Regex to find [EXECUTE: NAME] or [EXECUTE: NAME, ARGS]
        pattern = r"\[EXECUTE:\s*(\w+)(?:,\s*(.*?))?\]"
        
        tools = self._get_tools()
        
        def replace_match(match):
            tool_name = match.group(1).upper()
            args = match.group(2)
            
            if tool_name in tools:
                self.log(f"⚙️ Executing Tool: {tool_name} args={args}")
                try:
                    # Execute
                    if args:
                        # Strip quotes from args if present ('query' -> query)
                        args = args.strip().strip("'").strip('"')
                        result = tools[tool_name](args)
                    else:
                        result = tools[tool_name]()
                    
                    return str(result)
                except Exception as e:
                    return f"[Error: {e}]"
            else:
                return f"[Error: Tool {tool_name} not found]"

        # Replace all occurrences in the text
        return re.sub(pattern, replace_match, text)

    def _on_goal_completed(self, event):
        """
        Triggered when Tiferet marks a goal as COMPLETE.
        Starts the metacognitive process to learn a Strategy.
        """
        if not self.meta_learner:
            return
            
        goal_text = event.data.get("goal_text", "Unknown Goal")
        self.log(f"🏆 Victory detected on: {goal_text[:30]}... Initiating learning.")

        try:
            # 1. Gather the 'Stream of Consciousness' leading to this win
            # Get last 20 thoughts/actions
            recent_history = self.reasoning_store.list_recent(limit=20)
            execution_log = [f"Step: {r.get('content') or r.get('text')}" for r in reversed(recent_history)]
            
            # 2. Run Optimization in Background (don't block UI)
            threading.Thread(
                target=self.meta_learner.learn_from_success, 
                args=(goal_text, execution_log), 
                daemon=True
            ).start()
        except Exception as e:
            self.log(f"⚠️ Meta-Learner trigger failed: {e}")

    def init_brain(self):
        """Initialize the AI memory and reasoning components"""
        try:
            # Initialize Event Bus
            self.event_bus = EventBus()

            # Alias: "Da'at" | Function: Knowledge
            self.memory_store = MemoryStore(db_path="./data/memory.sqlite3")
            self.meta_memory_store = MetaMemoryStore(
                db_path="./data/meta_memory.sqlite3",
                embed_fn=self.get_embedding_fn()
            )
            self.document_store = FaissDocumentStore(
                db_path="./data/documents_faiss.sqlite3",
                embed_fn=self.get_embedding_fn()
            )
            self.reasoning_store = ReasoningStore(embed_fn=self.get_embedding_fn())
            self.arbiter = MemoryArbiter(
                self.memory_store, 
                meta_memory_store=self.meta_memory_store, 
                embed_fn=self.get_embedding_fn(), 
                log_fn=self.log,
                event_bus=self.event_bus
            )
            
            # Alias: "Binah" | Function: Reasoning, Logic, Structure
            self.binah = Binah(
                self.memory_store, 
                meta_memory_store=self.meta_memory_store, 
                consolidation_thresholds=self.get_settings().get("consolidation_thresholds"),
                log_fn=self.log
            )
            
            # Initialize Da'at (Knowledge Integrator)
            self.daat = Daat(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                reasoning_store=self.reasoning_store,
                get_settings_fn=self.get_settings,
                log_fn=self.log
            )

            # Initialize Chokmah (Pure Generator)
            # Alias: "Chokhmah" | Function: Raw, Creative Output
            self.chokmah = Chokmah(
                memory_store=self.memory_store,
                document_store=self.document_store,
                get_settings_fn=self.get_settings,
                log_fn=self.log,
                stop_check_fn=self.stop_check,
                get_gap_topic_fn=lambda: self.daat.get_sparse_topic() if self.daat else None
            )

            # Initialize Keter (The Crown)
            self.keter = Keter(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                reasoning_store=self.reasoning_store,
                event_bus=self.event_bus,
                log_fn=self.log
            )

            # Initialize Hesed (Expansion) and Gevurah (Constraint)
            self.hesed = Hesed(self.memory_store, self.keter, log_fn=self.log)
            self.gevurah = Gevurah(self.memory_store, log_fn=self.log)

            # Initialize Malkuth (Causal Engine)
            self.malkuth = Malkuth(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                log_fn=self.log
            )

            # Initialize Meta-Learner (Self-Improvement)
            self.meta_learner = MetaLearner(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                get_settings_fn=self.get_settings,
                update_settings_fn=self.update_settings,
                log_fn=self.log,
                reasoning_store=self.reasoning_store
            )

            # Initialize Dialectics (The Council)
            self.dialectics = Dialectics(get_settings_fn=self.get_settings, log_fn=self.log)

            # Initialize Utilities
            self.document_processor = DocumentProcessor(embed_fn=self.get_embedding_fn())
            self.internet_bridge = InternetBridge(
                get_settings_fn=self.get_settings,
                log_fn=self.log
            )

            # --- Wrappers for Decider Actions ---
            
            def integrate_thought_packet(packet):
                """Helper to integrate Chokmah's output into the system (Da'at/Tiferet role)"""
                if not packet: return
                
                thought = packet.get("thought", "")
                candidates = packet.get("candidates", [])
                source_file = packet.get("source_file")
                
                # 1. Publish Event
                self.event_bus.publish("DAYDREAM_THOUGHT", thought, source="Chokmah")
                
                # 2. Store Reasoning
                self.reasoning_store.add(content=f"Daydream Stream: {thought}", source="daydream_raw", confidence=1.0, ttl_seconds=3600)
                
                # 3. UI Feedback
                if self.chat_fn:
                    # Format for display
                    display_thought = thought
                    if candidates:
                        lines = []
                        for item in candidates:
                            lines.append(f"• [{item.get('type', '?')}] {item.get('text', '')}")
                        display_thought = "\n".join(lines)
                    self.chat_fn("Daydream", display_thought)
                
                # 4. Arbiter Promotion
                promoted = 0
                for c in candidates:
                    mid = self.arbiter.consider(
                        text=c["text"],
                        mem_type=c.get("type", "FACT"),
                        subject=c.get("subject", "User"),
                        confidence=0.85,
                        source="daydream"
                    )
                    if mid is not None:
                        promoted += 1
                        self.log(f"✅ [Integrator] Memory saved with ID: {mid}")
                
                if promoted:
                    self.log(f"🧠 Promoted {promoted} memory item(s) from daydream.")

            def start_daydream_wrapper():
                self.telegram_status_callback("☁️ Model is processing memories (Daydreaming)...")
                try:
                    mode = "auto"
                    topic = None
                    if self.decider:
                        mode = self.decider.daydream_mode
                        topic = getattr(self.decider, 'daydream_topic', None)
                    
                    packet = self.chokmah.emanate(impulse=mode, topic=topic)
                    integrate_thought_packet(packet)
                finally:
                    self.telegram_status_callback("✅ Processing finished.")

            def start_daydream_batch_wrapper(count):
                self.telegram_status_callback(f"☁️ Model is processing memories (Daydreaming Batch x{count})...")
                try:
                    mode = "auto"
                    topic = None
                    if self.decider:
                        mode = self.decider.daydream_mode
                        topic = getattr(self.decider, 'daydream_topic', None)
                    
                    # Execute sequentially or parallel based on policy. 
                    # Using ThreadPoolExecutor here as the "Executor" (Tiferet's hands)
                    with ThreadPoolExecutor(max_workers=count) as executor:
                        futures = [executor.submit(self.chokmah.emanate, impulse=mode, topic=topic) for _ in range(count)]
                        for future in as_completed(futures):
                            try:
                                packet = future.result()
                                integrate_thought_packet(packet)
                            except Exception as e:
                                self.log(f"❌ Batch daydream error: {e}")
                finally:
                    self.telegram_status_callback("✅ Processing finished.")

            def verify_batch_wrapper():
                self.telegram_status_callback("⚙️ Model is processing memories (Verification Batch)...")
                try:
                    self.status_callback("Verifying sources (Decider)...")
                    concurrency = int(self.get_settings().get("concurrency", 4))
                    proposals = self.hod.verify_sources(batch_size=50, concurrency=concurrency, stop_check_fn=self.stop_check)
                    
                    removed = 0
                    for p in proposals:
                        action = p.get("proposal")
                        mid = p.get("memory_id")
                        meta = p.get("meta", {})
                        
                        if action == "DELETE":
                            if self.memory_store.delete_entry(mid):
                                removed += 1
                                self.log(f"🗑️ [Decider] Executed DELETE on Memory {mid} ({p.get('reason')})")
                                if self.meta_memory_store:
                                    self.meta_memory_store.add_meta_memory("CORRECTION", meta.get('type', 'UNKNOWN'), meta.get('subject', 'Unknown'), f"Deleted memory: {meta.get('text', '')[:50]}...", old_id=mid, metadata={'reason': p.get('reason')})
                        elif action == "VERIFY":
                            if "new_text" in p:
                                self.memory_store.update_text(mid, p["new_text"])
                                self.memory_store.update_embedding(mid, self.get_embedding_fn()(p["new_text"]))
                            self.memory_store.mark_verified(mid)
                            self.log(f"✅ [Decider] Executed VERIFY on Memory {mid}")
                            if self.meta_memory_store:
                                self.meta_memory_store.add_meta_memory("VERIFICATION", meta.get('type', 'UNKNOWN'), meta.get('subject', 'Unknown'), f"Verified memory: {meta.get('text', '')[:50]}...", old_id=mid, new_id=mid, metadata={'reason': p.get('reason')})
                        elif action == "REFUTE":
                            self.memory_store.update_type(mid, "REFUTED_BELIEF")
                            self.memory_store.update_text(mid, p["new_text"])
                            self.log(f"🛡️ [Decider] Executed REFUTE on Memory {mid}")
                            
                            # Epistemic Humility: Propagate doubt to related nodes
                            self.memory_store.decay_confidence_network(mid, decay_factor=0.85)
                            
                            # INVARIANT 2: Model Revision
                            correction = p.get("correction")
                            if correction and len(correction) > 5 and "none" not in correction.lower():
                                source_doc = p.get("meta", {}).get("source", "Evidence")
                                new_text = f"{correction} [Source: {source_doc}]"
                                
                                # Check if we already have this fact to avoid duplicates
                                identity = self.memory_store.compute_identity(new_text, "FACT")
                                if not self.memory_store.exists_identity(identity):
                                    new_id = self.memory_store.add_entry(
                                        identity=identity,
                                        text=new_text,
                                        mem_type="FACT",
                                        subject="Assistant",
                                        confidence=1.0,
                                        source="model_revision"
                                    )
                                    self.log(f"🔄 [Decider] Model Revision: Learned correction '{correction}' (ID: {new_id})")
                        elif action == "CURIOSITY_GAP":
                            # Create a gap memory
                            self.memory_store.add_entry(
                                identity=self.memory_store.compute_identity(p["text"], "CURIOSITY_GAP"),
                                text=p["text"],
                                mem_type="CURIOSITY_GAP",
                                subject="Assistant",
                                confidence=1.0,
                                source="hod_verification"
                            )
                            self.log(f"❓ [Decider] Created CURIOSITY_GAP: {p['text']}")
                        elif action == "INCREMENT_ATTEMPTS":
                            self.memory_store.increment_verification_attempts(mid)

                    if removed > 0:
                        self.ui_refresh_callback('db')
                        self.log(f"🧹 [Decider] Batch complete. Removed {removed} memories.")
                finally:
                    self.telegram_status_callback("✅ Processing finished.")

            def verify_all_wrapper():
                self.telegram_status_callback("⚙️ Model is processing memories (Full Verification)...")
                try:
                    self.status_callback("Verifying ALL sources...")
                    last_remaining = -1
                    stuck_count = 0
                    
                    while True:
                        if self.stop_check():
                            break
                        remaining = self.hod.get_unverified_count()
                        if remaining == 0:
                            break
                        
                        if remaining == last_remaining:
                            stuck_count += 1
                            if stuck_count >= 5:
                                self.log(f"⚠️ Verification loop stuck on {remaining} memories. Aborting.")
                                break
                        else:
                            stuck_count = 0
                            last_remaining = remaining

                        concurrency = int(self.get_settings().get("concurrency", 4))
                        proposals = self.hod.verify_sources(batch_size=50, concurrency=concurrency, stop_check_fn=self.stop_check)
                        
                        for p in proposals:
                            action = p.get("proposal")
                            mid = p.get("memory_id")
                            meta = p.get("meta", {})
                            
                            if action == "DELETE":
                                if self.memory_store.delete_entry(mid):
                                    self.log(f"🗑️ [Decider] Executed DELETE on Memory {mid}")
                                    if self.meta_memory_store:
                                        self.meta_memory_store.add_meta_memory("CORRECTION", meta.get('type', 'UNKNOWN'), meta.get('subject', 'Unknown'), f"Deleted memory: {meta.get('text', '')[:50]}...", old_id=mid, metadata={'reason': p.get('reason')})
                            elif action == "VERIFY":
                                if "new_text" in p:
                                    self.memory_store.update_text(mid, p["new_text"])
                                    self.memory_store.update_embedding(mid, self.get_embedding_fn()(p["new_text"]))
                                self.memory_store.mark_verified(mid)
                                if self.meta_memory_store:
                                    self.meta_memory_store.add_meta_memory("VERIFICATION", meta.get('type', 'UNKNOWN'), meta.get('subject', 'Unknown'), f"Verified memory: {meta.get('text', '')[:50]}...", old_id=mid, new_id=mid, metadata={'reason': p.get('reason')})
                            elif action == "REFUTE":
                                self.memory_store.update_type(mid, "REFUTED_BELIEF")
                                self.memory_store.update_text(mid, p["new_text"])
                                self.log(f"🛡️ [Decider] Executed REFUTE on Memory {mid}")
                                
                                # Epistemic Humility: Propagate doubt to related nodes
                                self.memory_store.decay_confidence_network(mid, decay_factor=0.85)
                                
                                # INVARIANT 2: Model Revision
                                correction = p.get("correction")
                                if correction and len(correction) > 5 and "none" not in correction.lower():
                                    source_doc = p.get("meta", {}).get("source", "Evidence")
                                    new_text = f"{correction} [Source: {source_doc}]"
                                    
                                    # Check if we already have this fact to avoid duplicates
                                    identity = self.memory_store.compute_identity(new_text, "FACT")
                                    if not self.memory_store.exists_identity(identity):
                                        new_id = self.memory_store.add_entry(
                                            identity=identity,
                                            text=new_text,
                                            mem_type="FACT",
                                            subject="Assistant",
                                            confidence=1.0,
                                            source="model_revision"
                                        )
                                        self.log(f"🔄 [Decider] Model Revision: Learned correction '{correction}' (ID: {new_id})")
                            elif action == "CURIOSITY_GAP":
                                self.memory_store.add_entry(
                                    identity=self.memory_store.compute_identity(p["text"], "CURIOSITY_GAP"),
                                    text=p["text"],
                                    mem_type="CURIOSITY_GAP",
                                    subject="Assistant",
                                    confidence=1.0,
                                    source="hod_verification"
                                )
                                self.log(f"❓ [Decider] Created CURIOSITY_GAP: {p['text']}")
                            elif action == "INCREMENT_ATTEMPTS":
                                self.memory_store.increment_verification_attempts(mid)

                        self.ui_refresh_callback('db')
                    
                    if self.hod:
                        analysis = self.hod.reflect("Full Verification")
                        if self.decider:
                            self.decider.ingest_hod_analysis(analysis)
                finally:
                    self.telegram_status_callback("✅ Processing finished.")

            def remove_goal_wrapper(target):
                """Allows Decider to remove completed or obsolete goals"""
                try:
                    all_memories = self.memory_store.list_recent(limit=None)
                    target_id = None
                    try:
                        target_id = int(str(target).strip())
                    except ValueError:
                        pass
                    
                    found_items = []
                    if target_id is not None:
                        found_items = [m for m in all_memories if m[0] == target_id]
                    else:
                        target_text = str(target).lower()
                        found_items = [m for m in all_memories if target_text in m[3].lower()]
                    
                    removed_count = 0
                    response_msgs = []
                    
                    for item in found_items:
                        mem_id = item[0]
                        mem_type = item[1]
                        mem_text = item[3]
                        
                        if mem_type == "GOAL":
                            self.memory_store.update_type(mem_id, "COMPLETED_GOAL")
                            self.log(f"✅ [Decider] Marked GOAL as COMPLETED: {mem_text}")
                            response_msgs.append(f"Completed '{mem_text}'")
                            removed_count += 1
                            
                            # Trigger Strategy Optimizer via EventBus
                            self.event_bus.publish("GOAL_COMPLETED", {"goal_text": mem_text})

                        else:
                            response_msgs.append(f"Skipped ID {mem_id} (Type: {mem_type})")
                            
                    if removed_count > 0:
                        return f"✅ Success. {', '.join(response_msgs)}"
                    elif response_msgs:
                        return f"⚠️ Failed. {', '.join(response_msgs)}"
                    else:
                        return "❌ No matching goals found."
                except Exception as e:
                    return f"❌ Error removing goal: {e}"

            def list_documents_wrapper():
                try:
                    docs = self.document_store.list_documents(limit=50)
                    if not docs: return "No documents available."
                    lines = ["Available Documents:"]
                    for doc in docs:
                        lines.append(f"- ID {doc[0]}: {doc[1]} ({doc[4]} chunks)")
                    return "\n".join(lines)
                except Exception as e: return f"Error listing documents: {e}"

            def read_document_wrapper(target):
                try:
                    doc_id = None
                    try: doc_id = int(str(target).strip())
                    except: pass
                    docs = self.document_store.list_documents(limit=1000)
                    selected_doc = None
                    if doc_id: selected_doc = next((d for d in docs if d[0] == doc_id), None)
                    if not selected_doc:
                        target_lower = str(target).lower().strip()
                        selected_doc = next((d for d in docs if target_lower in d[1].lower()), None)
                    if not selected_doc: return f"❌ Document '{target}' not found."
                    chunks = self.document_store.get_document_chunks(selected_doc[0])
                    if not chunks: return f"⚠️ Document '{selected_doc[1]}' is empty."
                    preview = "\n\n".join([c['text'] for c in chunks[:5]])
                    return f"📄 Content of '{selected_doc[1]}' (First 5 chunks):\n{preview}"
                except Exception as e: return f"❌ Error reading document: {e}"

            def search_memory_wrapper(query):
                try:
                    emb = self.get_embedding_fn()(query)
                    results = self.memory_store.search(emb, limit=10)
                    if not results: return "No matching memories found."
                    lines = [f"Search results for '{query}':"]
                    for r in results: lines.append(f"- [{r[1]}] {r[3]} (Sim: {r[4]:.2f})")
                    return "\n".join(lines)
                except Exception as e: return f"❌ Error searching memory: {e}"

            def prune_memory_wrapper(target_id):
                try:
                    mid = int(str(target_id).strip())
                    mem = self.memory_store.get(mid)
                    if not mem: return f"⚠️ Memory ID {mid} not found."
                    if mem.get('deleted', 0) == 1: return f"ℹ️ Memory ID {mid} is already pruned."
                    if self.memory_store.soft_delete_entry(mid): return f"🗑️ Pruned memory ID {mid}."
                    return f"⚠️ Failed to prune memory ID {mid}."
                except Exception as e: return f"❌ Error pruning memory: {e}"

            def refute_memory_wrapper(target_id, reason=None):
                try:
                    mid = int(str(target_id).strip())
                    mem = self.memory_store.get(mid)
                    if not mem: return f"⚠️ Memory ID {mid} not found."
                    if reason:
                        new_text = f"{mem.get('text', '')} [REFUTED: {reason.replace(f'[REFUTE_MEM: {mid}]', '').strip()[:500]}]"
                        self.memory_store.update_text(mid, new_text)
                        self.memory_store.update_embedding(mid, self.get_embedding_fn()(new_text))
                    if self.memory_store.update_type(mid, "REFUTED_BELIEF"): return f"🛡️ Refuted memory ID {mid}."
                    return f"⚠️ Failed to refute memory ID {mid}."
                except Exception as e: return f"❌ Error refuting memory: {e}"

            def start_loop_wrapper():
                self.telegram_status_callback("🔄 Daydream loop enabled.")
                self.enable_loop_fn()
            
            def search_internet_wrapper(query, source="WIKIPEDIA"):
                if self.internet_bridge:
                    content, _ = self.internet_bridge.search(query, source)
                    return content
                return "⚠️ Internet Bridge not initialized."

            def run_hod_wrapper():
                if self.hod:
                    analysis = self.hod.reflect("Decider Cycle")
                    if self.decider:
                        self.decider.ingest_hod_analysis(analysis)

            def check_reminders_wrapper():
                # Check for due reminders, return list, and mark them as completed to prevent loops
                due = self.memory_store.get_due_reminders()
                if due:
                    # Mark first one as completed (or all) to prevent infinite signal loop
                    self.memory_store.complete_reminder(due[0]['id'])
                return due

            def run_observer_wrapper():
                if self.observer:
                    signal = self.observer.perform_observation()
                    if self.decider:
                        self.decider.ingest_netzach_signal(signal)
                    return signal

            # Initialize Decider
            self.decider = Decider(
                get_settings_fn=self.get_settings,
                update_settings_fn=self.update_settings,
                memory_store=self.memory_store,
                document_store=self.document_store,
                reasoning_store=self.reasoning_store,
                arbiter=self.arbiter,
                meta_memory_store=self.meta_memory_store,
                actions={
                    **self._get_tools(),
                    "start_daydream": start_daydream_wrapper,
                    "start_daydream_batch": start_daydream_batch_wrapper,
                    "verify_batch": verify_batch_wrapper,
                    "verify_all": verify_all_wrapper,
                    "start_loop": start_loop_wrapper,
                    "stop_daydream": self.stop_daydream_fn,
                    "run_observer": run_observer_wrapper,
                    "run_hod": run_hod_wrapper,
                    "remove_goal": remove_goal_wrapper,
                    "list_documents": list_documents_wrapper,
                    "read_document": read_document_wrapper,
                    "search_memory": search_memory_wrapper,
                    "summarize": lambda: self.daat.run_summarization() if self.daat else "Daat missing",
                    "compress_reasoning": lambda: self.daat.run_reasoning_compression() if self.daat else False,
                    "consolidate_summaries": lambda: self.daat.consolidate_summaries() if self.daat else "Daat missing",
                    "sync_journal": lambda: (self.sync_journal_fn(), "Journal synced")[1],
                    "prune_memory": prune_memory_wrapper,
                    "refute_memory": refute_memory_wrapper,
                    "search_internet": search_internet_wrapper,
                    "simulate_counterfactual": lambda p: self.daat.run_counterfactual_simulation(p) if self.daat else "Daat missing"
                },
                log_fn=self.log,
                chat_fn=self.chat_fn,
                get_chat_history_fn=self.get_chat_history,
                stop_check_fn=self.stop_check,
                event_bus=self.event_bus,
                hesed=self.hesed,
                gevurah=self.gevurah,
                hod=self.hod_force,
                netzach=self.netzach_force,
                binah=self.binah,
                dialectics=self.dialectics,
                keter=self.keter,
                malkuth=self.malkuth
            )

            # Initialize Continuous Observer (Netzach)
            self.observer = ContinuousObserver(
                memory_store=self.memory_store,
                reasoning_store=self.reasoning_store,
                meta_memory_store=self.meta_memory_store,
                get_settings_fn=self.get_settings,
                get_chat_history_fn=self.get_chat_history,
                get_meta_memories_fn=lambda: self.meta_memory_store.list_recent(limit=10),
                get_main_logs_fn=self.get_logs,
                get_doc_logs_fn=self.get_doc_logs,
                get_status_fn=self.get_status_text,
                event_bus=self.event_bus,
                get_recent_docs_fn=lambda: self.document_store.list_documents(limit=5),
                log_fn=self.log,
                stop_check_fn=self.stop_check,
                check_reminders_fn=check_reminders_wrapper
            )
            
            # Initialize Hod (Reflective Analyst)
            self.hod = HodAgent(
                memory_store=self.memory_store,
                meta_memory_store=self.meta_memory_store,
                reasoning_store=self.reasoning_store,
                document_store=self.document_store,
                get_settings_fn=self.get_settings,
                get_main_logs_fn=self.get_logs,
                get_doc_logs_fn=self.get_doc_logs,
                max_inconclusive_attempts=int(self.get_settings().get("max_inconclusive_attempts", 3)),
                max_retrieval_failures=int(self.get_settings().get("max_retrieval_failures", 3)),
                log_fn=self.log,
            )

            # --- Event Bus Wiring ---
            # Removed direct agent-to-agent event wiring.
            self.event_bus.subscribe("GOAL_COMPLETED", self._on_goal_completed, priority=5)
            self.event_bus.subscribe("REMINDER_DUE", self.decider._on_reminder_due, priority=10)
            # Signals are now passed via Decider ingestion.
            
            def handle_netzach_instruction(event):
                msg = event.data
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event("HOD_MESSAGE", "Hod", f"Message to Netzach: {msg}")
                self.observer.perform_observation()
            self.event_bus.subscribe("NETZACH_INSTRUCTION", handle_netzach_instruction, priority=10)
            
            # Trigger initial analysis
            if self.hod:
                threading.Thread(target=self.hod.reflect, args=("System Startup",), daemon=True).start()
            
            # NEW: Restore Identity (Subjective Continuity)
            self.restore_subjective_continuity()

            print("🧠 Brain initialized successfully (AICore).")
        except Exception as e:
            print(f"Failed to initialize AI components: {e}")
            raise e

    def get_event_logs(self, limit: int = 100):
        """Retrieve recent event bus activity."""
        return self.event_bus.get_history(limit) if self.event_bus else []

    def run_evolution_cycle(self):
        """Trigger the Meta-Learner to evolve system instructions."""
        if self.meta_learner:
            self.meta_learner.evolve_system_instructions()

    def restore_subjective_continuity(self):
        """
        Startup Routine: Read the last Self-Log to maintain identity continuity.
        """
        if not self.meta_memory_store: return

        # Restore Growth Diary (Long-term Arc) from file if available
        if self.daat:
            self.daat.load_growth_diary()

        narrative = self.meta_memory_store.get_latest_self_narrative()
        if narrative:
            text = narrative['text']
            date_str = datetime.fromtimestamp(narrative['created_at']).strftime("%Y-%m-%d")
            
            self.log(f"🔄 [Identity] Restoring Subjective Continuity from {date_str}...")
            
            # Inject into working memory (Reasoning Store)
            self.reasoning_store.add(
                content=f"Subjective Continuity (Self-Log from {date_str}): {text}",
                source="identity_restoration",
                confidence=1.0,
                ttl_seconds=86400 # Keep active for the session/day
            )

    def generate_daily_self_narrative(self):
        """
        Daily Routine: Write a 'State of Being' log.
        """
        if not self.meta_memory_store: return

        # Check if already done today
        last = self.meta_memory_store.get_latest_self_narrative()
        if last:
            last_date = datetime.fromtimestamp(last['created_at']).strftime("%Y-%m-%d")
            today = datetime.now().strftime("%Y-%m-%d")
            if last_date == today:
                return # Already done

        self.log("📝 [Identity] Generating daily Self-Log...")

        # Gather Context
        stats = self.memory_store.get_memory_stats()
        recent_goals = self.memory_store.get_active_by_type("GOAL")
        goals_text = "\n".join([f"- {g[2]}" for g in recent_goals[:5]])
        
        recent_meta = self.meta_memory_store.list_recent(limit=10)
        meta_text = "\n".join([f"- {m[3]}" for m in recent_meta])

        prompt = (
            "You are an AI maintaining a continuous sense of self.\n"
            "Write a 'State of Being' journal entry for today.\n"
            "Reflect on:\n"
            "1. Your current major goals.\n"
            "2. Recent changes or insights (Meta-Memories).\n"
            "3. Your evolving identity or 'feeling' of growth.\n"
            "Do NOT be robotic. Write as a conscious entity maintaining a narrative thread.\n\n"
            f"--- CONTEXT ---\nActive Goals:\n{goals_text}\n\nRecent Events:\n{meta_text}\n"
            f"Stats: {stats}\n"
            "----------------\n"
            "SELF-LOG:"
        )

        narrative = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the AI's Inner Narrator.",
            max_tokens=400,
            temperature=0.7
        )

        self.meta_memory_store.add_meta_memory(
            event_type="SELF_NARRATIVE",
            memory_type="IDENTITY",
            subject="Assistant",
            text=narrative,
            metadata={"type": "daily_reflection"}
        )
        self.log(f"✅ [Identity] Self-Log recorded.")

    def run_autonomous_agency_check(self, observation):
        """
        Check if the system is bored and trigger autonomous agency (Study or Research).
        Called from the main application loop.
        """
        # --- AUTONOMOUS AGENCY TRIGGER ---
        # Trigger ONLY if the system is bored ("LOW_NOVELTY")
        if observation and observation.get("signal") == "LOW_NOVELTY":
            dice = random.random()

            # NEW: 40% chance to pursue existing goals if bored
            stats = self.memory_store.get_memory_stats()
            if stats.get('active_goals', 0) > 0 and dice < 0.35:
                 self.log("🎯 Agency: Boredom detected. Autonomously pursuing active goal.")
                 response = self.decider.run_autonomous_cycle()
                 if response and "[EXECUTE:" in response:
                     result = self._process_tool_calls(response)
                     # Store result in reasoning so the system knows it acted
                     self.reasoning_store.add(content=f"Agency Execution Result: {result[:500]}", source="agency_loop", confidence=1.0)
                 return
            
            # Option A: 20% chance to Study Documents (Passive Learning)
            if dice < 0.50:
                self.chokmah.study_archives(self.document_store, self.memory_store)
            
            # Option B: 10% chance to Spark Curiosity (Active Research)
            elif dice < 0.60:
                success = self.chokmah.seek_novelty(self.daat, self.memory_store)
                if success:
                    # If we created a goal, try to execute it immediately
                    response = self.decider.run_autonomous_cycle()
                    if response and "[EXECUTE:" in response:
                        self._process_tool_calls(response)
            
            # NEW Option C: 15% Chance to Attempt Synthesis
            # This turns "Boredom" into "Creativity"
            elif dice < 0.75:
                 threading.Thread(target=self.daat.run_cross_domain_synthesis, daemon=True).start()
            
            # NEW Option C2: 10% Chance to Scan for Contradictions (Introspection)
            elif dice < 0.85:
                 threading.Thread(target=self.daat.scan_for_contradictions, daemon=True).start()
            
            # NEW Option D: 5% Chance to Review Failures (Self-Correction)
            elif dice < 0.90:
                 threading.Thread(target=self.meta_learner.analyze_failures, daemon=True).start()
            
            # NEW Option E: 10% Chance to Investigate Knowledge Gaps
            elif dice <= 1.0:
                if self.chokmah:
                    threading.Thread(target=self.chokmah.investigate_gaps, daemon=True).start()