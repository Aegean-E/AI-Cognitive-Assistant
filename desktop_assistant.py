"""
AI Desktop Assistant
A standalone desktop application with integrated chat, document management, and Telegram bridge
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
import json
import os
import re
from datetime import datetime
import shutil
from typing import Dict, List, Optional

# Import AI Core
from ai_core import AICore

from bridges.telegram_api import TelegramBridge

from ui import DesktopAssistantUI
from docs.commands import handle_command as process_command_logic, NON_LOCKING_COMMANDS

class DesktopAssistantApp(DesktopAssistantUI):

    def __init__(self, root):
        self.root = root
        self.root.title("AI Desktop Assistant")
        self.root.geometry("1200x800")

        # Track the settings file path first
        self.settings_file_path = "./settings.json"

        # State
        self.settings = self.load_settings()
        self.telegram_bridge = None
        self.observer = None
        self.connected = False
        self.is_showing_placeholder = False  # Track placeholder state
        self.stop_processing_flag = False
        self.is_processing = False
        self.daydreamer = None
        self.decider = None
        self.hod = None
        self.keter = None
        self.start_time = time.time()
        self.processing_lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.telegram_status_sent = False  # Track if status has been sent to avoid spam
        
        # Initialize chat mode based on settings
        initial_mode = self.settings.get("ai_mode", "Daydream")
        self.chat_mode_var = tk.BooleanVar(value=(initial_mode == "Chat"))
        self.daydream_cycle_count = 0
        self.pending_confirmation_command = None

        # Initialize ttkbootstrap style with loaded theme
        theme_map = {
            "Cosmo": "cosmo",
            "Cyborg": "cyborg",
            "Darkly": "darkly"
        }
        theme_to_apply = theme_map.get(self.settings.get("theme", "Darkly"), self.settings.get("theme", "darkly"))
        self.style = ttk.Style(theme=theme_to_apply)

        # Initialize bridge toggle
        self.telegram_bridge_enabled = tk.BooleanVar()

        self.setup_ui()
        self.load_settings_into_ui()
        
        # Redirect stdout/stderr to logs tab
        self.redirect_logging()

        # Initialize Brain (Memory & Documents) - Moved after UI setup to capture logs
        self.chat_memory = {}
        self.init_ai_core()
        
        # Refresh documents list now that DB is initialized
        self.refresh_documents()
        self.refresh_database_view()

        # Start background processes (Consolidation)
        self.start_background_processes()

        # Initialize connection state based on settings
        if (self.settings.get("telegram_bridge_enabled", False) and
            self.settings.get("bot_token") and
            self.settings.get("chat_id")):
            self.telegram_bridge_enabled.set(True)
            # Attempt to connect if credentials are provided and bridge is enabled
            self.bot_token_var.set(self.settings.get("bot_token"))
            self.chat_id_var.set(self.settings.get("chat_id"))
            # Connect automatically if settings are valid
            self.connect()
        else:
            self.telegram_bridge_enabled.set(False)
            # Ensure we're disconnected
            self.disconnect()

    def handle_ui_refresh(self, target=None):
        """Callback for AICore to request UI updates"""
        if target == 'db':
            self.root.after(0, self.refresh_database_view)
        elif target == 'docs':
            self.root.after(0, self.refresh_documents)
        else:
            self.root.after(0, self.refresh_database_view)
            self.root.after(0, self.refresh_documents)

    def init_ai_core(self):
        """Initialize the AI Core and alias components"""
        try:
            # Wrapper to ensure newlines for internal AI logs (fixes log truncation/merging)
            def log_with_newline(msg):
                if msg:
                    self.log_to_main(f"{msg}\n")

            self.ai_core = AICore(
                settings_provider=lambda: self.settings,
                log_fn=log_with_newline,
                chat_fn=self.on_proactive_message,
                status_callback=lambda msg: self.root.after(0, lambda: self.status_var.set(msg)),
                telegram_status_callback=self.send_telegram_status,
                ui_refresh_callback=self.handle_ui_refresh,
                get_chat_history_fn=self.get_current_chat_history,
                get_logs_fn=self.get_recent_main_logs,
                get_doc_logs_fn=self.get_recent_doc_logs,
                get_status_text_fn=self.get_current_status_text,
                update_settings_fn=self.update_settings_from_decider,
                stop_check_fn=lambda: self.stop_processing_flag,
                enable_loop_fn=self.enable_daydream_loop,
                stop_daydream_fn=self.stop_processing,
                sync_journal_fn=self.sync_journal
            )
            
            # Alias components for compatibility
            self.memory_store = self.ai_core.memory_store
            self.meta_memory_store = self.ai_core.meta_memory_store
            self.document_store = self.ai_core.document_store
            self.reasoning_store = self.ai_core.reasoning_store
            self.arbiter = self.ai_core.arbiter
            self.binah = self.ai_core.binah
            self.event_bus = self.ai_core.event_bus
            self.keter = self.ai_core.keter
            self.hesed = self.ai_core.hesed
            self.gevurah = self.ai_core.gevurah
            self.hod_force = self.ai_core.hod_force
            self.netzach_force = self.ai_core.netzach_force
            self.decider = self.ai_core.decider
            self.observer = self.ai_core.observer
            self.hod = self.ai_core.hod
            self.daat = self.ai_core.daat
            self.document_processor = self.ai_core.document_processor
            self.internet_bridge = self.ai_core.internet_bridge
            
            # Subscribe to AI_SPEAK
            self.ai_core.event_bus.subscribe("AI_SPEAK", self.on_ai_speak, priority=10)
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize AI Core:\n{e}")

    def update_settings_from_decider(self, new_settings: Dict):
        """Callback for Decider to update settings and UI"""
        self.settings.update(new_settings)
        self.save_settings()
        # Update UI on main thread
        if hasattr(self, 'temperature_var'):
            self.root.after(0, lambda: self.temperature_var.set(new_settings.get("temperature", 0.7)))
        if hasattr(self, 'max_tokens_var'):
            self.root.after(0, lambda: self.max_tokens_var.set(new_settings.get("max_tokens", 800)))

    def enable_daydream_loop(self):
        """Allow the daydream loop to run by clearing stop flags and chat mode"""
        self.stop_processing_flag = False
        if self.chat_mode_var.get():
            self.chat_mode_var.set(False)
            self.on_chat_mode_toggle()

    def get_recent_main_logs(self) -> str:
        """Get last 15 lines of main logs for Netzach"""
        if hasattr(self, 'log_buffer'):
            full_text = "".join([str(x) for x in self.log_buffer if x is not None])
            lines = full_text.splitlines()
            return "\n".join(lines[-15:])
        return ""

    def get_recent_doc_logs(self) -> str:
        """Get last 10 lines of document logs for Netzach"""
        if hasattr(self, 'debug_log_buffer'):
            return "".join(self.debug_log_buffer[-10:])
        return ""

    def get_current_status_text(self) -> str:
        """Get current status bar text for Netzach"""
        return self.status_var.get()

    def get_current_chat_history(self) -> List[Dict]:
        """Helper for Daydreamer/Observer to see the current conversation"""
        chat_id = int(self.settings.get("chat_id", 0) or 0)
        return self.chat_memory.get(chat_id, [])

    def load_settings(self) -> Dict:
        """Load settings from file"""
        settings_file = self.settings_file_path
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                
                # Auto-repair: Fix memory extractor prompt if it contains the old "echoes" rule
                current_prompt = settings.get("memory_extractor_prompt", "")
                if "echoes of the other party" in current_prompt or "SELF-CONTAINED" not in current_prompt or "Assistant's suggestions" not in current_prompt:
                    print("🔧 Auto-repairing memory extractor prompt in settings...")
                    # Hardcoded correct prompt to ensure fix
                    settings["memory_extractor_prompt"] = "Extract durable memories from the conversation. Return ONLY a valid JSON array.\n\nMemory Types (Apply to BOTH User and Assistant):\n- IDENTITY: Names, roles, locations, RELIGION (e.g., 'Assistant religion is Hinduism', 'Assistant is a Buddhist')\n- PREFERENCE: Likes/dislikes (e.g., 'User loves coffee', 'Assistant prefers inner peace')\n- GOAL: Specific, actionable objectives (e.g., 'User wants to learn Python', 'Assistant plans to analyze X'). Do NOT extract general statements like 'Future research should...' as GOALs.\n- FACT: Objective truths (e.g., 'User is an engineer', 'Assistant can process data')\n- BELIEF: Opinions/convictions (e.g., 'User believes AI is good', 'Assistant believes in meditation')\n- PERMISSION: Explicit user grants (e.g., 'User allowed Assistant to hold opinions')\n- RULE: Behavior guidelines (e.g., 'Assistant should not use emojis')\n\nRules:\n1. Extract from BOTH User AND Assistant.\n2. Each object MUST have: \"type\", \"subject\" (User or Assistant), \"text\".\n3. Use DOUBLE QUOTES for all keys and string values.\n4. Max 5 memories, max 240 chars each.\n5. EXCLUDE: Pure greetings (e.g., 'Hi'), questions, and filler. DO NOT exclude facts stated during introductions (e.g., 'Hi, I'm X').\n6. EXCLUDE generic assistant politeness (e.g., 'Assistant goal is to help', 'I'm here to help', 'feel free to ask').\n7. EXCLUDE contextual/situational goals (e.g., 'help with X topic' where X is current conversation topic).\n8. ONLY extract ASSISTANT GOALS if they represent true self-chosen objectives or explicit commitments.\n9. DO NOT extract facts from the Assistant's text if it is merely recalling known info. ALWAYS extract new facts from the User's text.\n10. ATTRIBUTION RULE: If User says 'I am X', subject is User. If Assistant says 'I am X', subject is Assistant. NEVER attribute User statements to Assistant.\n11. CRITICAL: DO NOT attribute Assistant's suggestions, lists, or hypothetical topics to the User. Only record User interests if the USER explicitly stated them.\n12. MAKE MEMORIES SELF-CONTAINED: Replace pronouns like 'This', 'These', 'It' with specific nouns. Ensure the text makes sense without the surrounding context.\n13. If no new memories, return [].\n"
                    self.save_settings_to_file(settings)
                
                # Ensure defaults exist for Decider baselines to prevent drift
                if "default_temperature" not in settings:
                    settings["default_temperature"] = 0.7
                if "default_max_tokens" not in settings:
                    settings["default_max_tokens"] = 800
                
                return settings
        else:
            return {}

    def save_settings(self):
        """Save settings to file"""
        settings_file = self.settings_file_path
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
    def save_settings_to_file(self, settings_dict):
        """Helper to write settings dict to disk"""
        with open(self.settings_file_path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)

    def on_proactive_message(self, sender, msg):
        """Handle proactive messages from Daydreamer or Observer (Netzach)"""
        # 1. Always log to AI Interactions (Netzach) window for transparency
        self.root.after(0, lambda: self.add_netzach_message(f"{sender}: {msg}"))

        # 2. Determine if it should appear in the Main Chat
        # Only explicit messages (SPEAK) should appear in main chat.
        # Thoughts, decisions, and daydreaming are internal.
        should_show_in_chat = False
        
        if sender == "Decider":
            # Filter out internal logs/thoughts. If it's a [SPEAK] message, it won't have these markers.
            internal_markers = ["🤔", "💭", "🤖", "🛠️", "📩", "⚠️", "✅", "Decision:", "Thought:"]
            if not any(marker in msg for marker in internal_markers):
                should_show_in_chat = True
        
        if should_show_in_chat:
            # Show in local UI
            self.root.after(0, lambda: self.add_chat_message("Assistant", msg, "incoming"))
            
            # Forward to Telegram
            if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                 self.telegram_bridge.send_message(msg)

            # Update Chat Memory so the AI remembers its own proactive statement
            chat_id = int(self.settings.get("chat_id", 0) or 0)
            history = self.chat_memory.get(chat_id, [])
            history.append({"role": "assistant", "content": msg})
            if len(history) > 20:
                history = history[-20:]
            self.chat_memory[chat_id] = history

    def on_ai_speak(self, event):
        """Handle proactive AI speech events"""
        message = event.data
        self.root.after(0, lambda: self.on_proactive_message("Assistant (Insight)", message))

    def stop_processing(self):
        """Stop current AI generation"""
        print("🛑 Stop button clicked.")
        if self.is_processing:
            self.stop_processing_flag = True
            self.status_var.set("Stopping...")
            print("⏳ Sending stop signal to background process...")
        else:
            print("ℹ️ AI is currently idle.")

    def stop_daydream(self):
        """Stop daydreaming specifically"""
        print("🛑 Stop Daydream triggered.")
        self.stop_processing_flag = True
        
        if self.decider:
            self.decider.report_forced_stop()
            
        # Reset flag after a moment to allow Decider to pick up the "forced stop" state
        def reset_flag():
            time.sleep(1.5) 
            self.stop_processing_flag = False
            print("▶️ Decider ready for next turn (Cooldown active).")
            
        threading.Thread(target=reset_flag, daemon=True).start()

    def on_chat_mode_toggle(self):
        """Handle chat mode toggle"""
        if self.chat_mode_var.get():
            print("🔒 Chat Mode enabled. Telegram Bridge active.")
        else:
            print("🔓 Chat Mode disabled. Telegram Bridge paused.")

    def start_daydream(self):
        """Manually trigger a daydream cycle"""
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy processing a task.")
            return
            
        if self.decider:
            self.decider.start_daydream()
            # The background loop will pick this up
        else:
            messagebox.showerror("Error", "Decider not initialized.")

    def verify_memory_sources(self):
        """Manually trigger memory source verification"""
        # Alias: "Binah" | Function: Reasoning and Logic
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy (e.g. Daydreaming). Please click 'Stop' or enable 'Chat Mode' first.")
            return
            
        if not hasattr(self, 'hod'):
            return

        def verify_thread():
            print("🧹 [Manual Verifier] Starting quick batch verification...")
            with self.processing_lock:
                self.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying memory sources..."))
                
                try:
                    # Use a smaller batch for quick verification
                    if self.stop_processing_flag:
                        return

                    concurrency = int(self.settings.get("concurrency", 4))
                    removed = self.hod.verify_sources(batch_size=50, concurrency=concurrency, stop_check_fn=lambda: self.stop_processing_flag)
                    msg = f"Verification complete. Removed {removed} hallucinated memories."
                    print(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    self.root.after(0, self.refresh_database_view)
                    if self.hod:
                        self.hod.reflect("Verification Batch")
                except Exception as e:
                    print(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=verify_thread, daemon=True).start()

    def verify_all_memory_sources(self):
        """Loop verification until all memories are verified"""
        if self.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy. Please click 'Stop' first.")
            return
            
        if not hasattr(self, 'hod'):
            return

        def verify_all_thread():
            print("🧹 [Manual Verifier] Starting FULL verification loop...")
            with self.processing_lock:
                self.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying ALL sources..."))
                
                total_removed = 0
                last_remaining = -1
                stuck_count = 0
                
                try:
                    while True:
                        if self.stop_processing_flag:
                            print("🛑 Verification loop stopped by user.")
                            break
                        
                        # Check if anything left to verify
                        remaining = self.hod.get_unverified_count()
                        if remaining == 0:
                            print("✅ All cited memories verified.")
                            break
                        
                        # Loop protection
                        if remaining == last_remaining:
                            stuck_count += 1
                            if stuck_count >= 5:
                                print(f"⚠️ Verification loop stuck on {remaining} memories. Aborting.")
                                break
                        else:
                            stuck_count = 0
                            last_remaining = remaining
                        
                        self.root.after(0, lambda: self.status_var.set(f"Verifying... ({remaining} left)"))
                        
                        # Verify a batch
                        concurrency = int(self.settings.get("concurrency", 4))
                        removed = self.hod.verify_sources(batch_size=10000, concurrency=concurrency, stop_check_fn=lambda: self.stop_processing_flag)
                        total_removed += removed
                        
                        # Refresh UI to show progress
                        self.root.after(0, self.refresh_database_view)
                        
                    msg = f"Full verification complete. Removed {total_removed} memories."
                    print(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    if self.hod:
                        self.hod.reflect("Full Verification")
                except Exception as e:
                    print(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=verify_all_thread, daemon=True).start()

    def toggle_connection(self):
        """Toggle connection to Telegram"""
        # Toggle the setting
        new_state = not self.telegram_bridge_enabled.get()
        self.telegram_bridge_enabled.set(new_state)
        # Save the new state
        self.settings["telegram_bridge_enabled"] = new_state
        self.save_settings()

        # Update connection based on new state
        if new_state:
            # Only connect if both credentials are provided
            bot_token = self.bot_token_var.get().strip()
            chat_id_str = self.chat_id_var.get().strip()
            if bot_token and chat_id_str:
                try:
                    int(chat_id_str)  # Validate chat ID is numeric
                    self.connect()
                except ValueError:
                    messagebox.showerror("Connection Error", "Chat ID must be a valid number")
                    self.telegram_bridge_enabled.set(False)
                    self.settings["telegram_bridge_enabled"] = False
                    self.save_settings()
            else:
                messagebox.showerror("Connection Error", "Please enter both Bot Token and Chat ID in Settings")
                self.telegram_bridge_enabled.set(False)
                self.settings["telegram_bridge_enabled"] = False
                self.save_settings()
        else:
            self.disconnect()

    def connect(self):
        """Connect to Telegram"""
        if self.is_connected():
            return  # Already connected

        bot_token = self.bot_token_var.get().strip()
        chat_id_str = self.chat_id_var.get().strip()

        if not bot_token or not chat_id_str:
            # Don't show error if called internally - just return
            return

        try:
            chat_id = int(chat_id_str)
            # Alias: "Yesod" | Function: Foundation, Transmission
            self.telegram_bridge = TelegramBridge(bot_token, chat_id)

            # Test connection
            if self.telegram_bridge.send_message("✅ Connected to Desktop Assistant"):
                self.connected = True
                self.connect_button.config(text="Connected", bootstyle="success")
                self.status_var.set("Connected to Telegram")

                # Start message polling
                threading.Thread(
                    target=self.telegram_bridge.listen,
                    kwargs={
                        "on_text": self.handle_telegram_text,
                        "on_document": lambda m: threading.Thread(target=self.handle_telegram_document, args=(m,), daemon=True).start(),
                        "on_photo": self.handle_telegram_photo,
                        "running_check": lambda: self.is_connected() and self.settings.get("telegram_bridge_enabled", False),
                        "start_timestamp": self.start_time
                    },
                    daemon=True
                ).start()

            else:
                raise Exception("Failed to send test message")

        except Exception as e:
            # Only show error if this was a direct user action
            if not self.settings.get("telegram_bridge_enabled", False):
                # If bridge is disabled, don't show error
                pass
            else:
                messagebox.showerror("Connection Error", f"Failed to connect: {e}")
            self.disconnect()

    def disconnect(self):
        """Disconnect from Telegram"""
        self.connected = False
        self.telegram_bridge = None
        self.connect_button.config(text="Connect", bootstyle="secondary")
        self.status_var.set("Disconnected from Telegram")

    def send_telegram_status(self, message: str):
        """Send a status update to Telegram if connected"""
        if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
             # Suppress repetitive status messages until user interacts
             if self.telegram_status_sent:
                 return

             if self.telegram_bridge.send_message(message):
                 if "finished" in message.lower():
                     self.telegram_status_sent = True

    def is_connected(self):
        """Check if connected to Telegram"""
        return self.connected and self.telegram_bridge is not None

    def handle_command(self, text: str, chat_id: int) -> Optional[str]:
        """Process slash commands and return response if matched"""
        return process_command_logic(self, text, chat_id)

    def send_message(self, event=None):
        """Send message to both local chat and Telegram"""
        message = self.message_entry.get().strip()
        if not message:
            return

        # Add to local chat UI immediately
        self.add_chat_message("You", message, "outgoing")
        self.message_entry.delete(0, tk.END)

    def send_image(self):
        """Select and send an image to the AI"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return

        # Get optional caption from entry
        caption = self.message_entry.get().strip()
        if not caption:
            caption = "Analyze this image."
        
        # Clear entry
        self.message_entry.delete(0, tk.END)

        # Create a temp copy to avoid deleting the user's original file
        temp_dir = "./data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{filename}")
        shutil.copy2(file_path, temp_path)

        # Add to UI
        self.add_chat_message("You", f"{filename}\n{caption}", "outgoing", image_path=temp_path, trigger_processing=False)

        # Process in background
        threading.Thread(
            target=self.process_message_thread,
            args=(caption, True, None, temp_path),
            daemon=True
        ).start()

    def process_message_thread(self, user_text: str, is_local: bool, telegram_chat_id=None, image_path: str = None):
        """
        Core AI Logic: RAG -> LLM -> Memory Extraction -> Response
        Runs in a separate thread.
        """
        # Check for non-locking commands (read-only) to avoid waiting for processing lock
        cmd = user_text.strip().split()[0].lower() if user_text.strip() else ""
        
        if cmd in NON_LOCKING_COMMANDS:
            try:
                chat_id = telegram_chat_id if telegram_chat_id else int(self.settings.get("chat_id", 0) or 0)
                response = self.handle_command(user_text.strip(), chat_id)
                if response:
                    self.root.after(0, lambda: self.add_chat_message("System", response, "incoming"))
                    if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                        if telegram_chat_id:
                            self.telegram_bridge.send_message(response)
                        elif is_local:
                            self.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                            self.telegram_bridge.send_message(response)
            except Exception as e:
                print(f"Error executing non-locking command: {e}")
            return

        # Use chat_lock to serialize chat messages, but allow concurrency with Daydream (processing_lock)
        with self.chat_lock:
            self.stop_processing_flag = False
            
            try:
                # Determine Chat ID (Local uses 0 or configured ID, Telegram uses actual ID)
                chat_id = telegram_chat_id if telegram_chat_id else int(self.settings.get("chat_id", 0) or 0)
                
                if self.stop_processing_flag:
                    return

                # Check for commands
                if user_text.strip().startswith("/"):
                    response = self.handle_command(user_text.strip(), chat_id)
                    if response:
                        # Send response to UI
                        self.root.after(0, lambda: self.add_chat_message("System", response, "incoming"))
                        
                        # Send to Telegram if applicable
                        if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                            if telegram_chat_id:
                                self.telegram_bridge.send_message(response)
                            elif is_local:
                                self.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                                self.telegram_bridge.send_message(response)
                        return

                if self.stop_processing_flag:
                    return

                # 1. Prepare Context (Chat History)
                history = self.chat_memory.get(chat_id, [])
                history.append({"role": "user", "content": user_text})
                
                # Limit history
                if len(history) > 20: 
                    history = history[-20:]

                if self.stop_processing_flag:
                    return

                # Delegate core logic to Decider
                reply = self.decider.process_chat_message(
                    user_text=user_text,
                    history=history,
                    status_callback=lambda msg: self.root.after(0, lambda: self.status_var.set(msg)),
                    image_path=image_path
                )

                if self.stop_processing_flag:
                    return

                # Execute tools inside the response
                reply = self.ai_core._process_tool_calls(reply)

                # Update History
                history.append({"role": "assistant", "content": reply})
                self.chat_memory[chat_id] = history

                # Update UI (Thread-safe)
                self.root.after(0, lambda: self.add_chat_message("Assistant", reply, "incoming"))

                # Now that the user has their reply, let the Decider think about what's next.
                if self.decider:
                    # Run in background thread to avoid blocking the chat lock for the next message
                    threading.Thread(target=self.decider.run_post_chat_decision_cycle, daemon=True).start()

                # Send to Telegram if applicable
                if self.is_connected() and self.settings.get("telegram_bridge_enabled", False) and self.chat_mode_var.get():
                    # If local user typed it, send to Telegram
                    if is_local:
                        self.telegram_bridge.send_message(f"Desktop: {user_text}") # Optional: echo user text
                        self.telegram_bridge.send_message(reply)
                    # If it came from Telegram, just send the reply
                    elif telegram_chat_id:
                        self.telegram_bridge.send_message(reply)

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing message: {error_msg}")
                self.root.after(0, lambda: self.add_chat_message("System", f"Error: {error_msg}", "incoming"))
            finally:
                # Do not delete image_path here, as UI needs it for display/click
                self.stop_processing_flag = False
                if not self.is_processing:
                    self.root.after(0, lambda: self.status_var.set("Ready"))

    def handle_telegram_document(self, msg: Dict):
        """Handle document upload from Telegram"""
        try:
            file_info = msg["document"]
            file_id = file_info["file_id"]
            file_name = file_info.get("file_name", "unknown_file")
            file_size = file_info.get("file_size", 0)
            chat_id = msg["chat_id"]

            # Check supported types
            if not file_name.lower().endswith(('.pdf', '.docx')):
                self.telegram_bridge.send_message(f"⚠️ Unsupported file type: {file_name}. Please send PDF or DOCX.")
                return

            self.telegram_bridge.send_message(f"📄 Received {file_name}, processing...")

            # Get file path from Telegram
            file_data = self.telegram_bridge.get_file_info(file_id)
            telegram_file_path = file_data["file_path"]

            # Download
            local_dir = "./data/uploaded_docs"
            os.makedirs(local_dir, exist_ok=True)
            local_file_path = os.path.join(local_dir, file_name)
            
            self.telegram_bridge.download_file(telegram_file_path, local_file_path)

            # Check duplicates
            file_hash = self.document_store.compute_file_hash(local_file_path)
            if self.document_store.document_exists(file_hash):
                self.telegram_bridge.send_message(f"⚠️ Document '{file_name}' already exists in database. Skipping...")
                os.remove(local_file_path)
                return

            # Process
            chunks, page_count, file_type = self.document_processor.process_document(local_file_path)

            # Add to store
            self.document_store.add_document(
                file_hash=file_hash,
                filename=file_name,
                file_type=file_type,
                file_size=file_size,
                page_count=page_count,
                chunks=chunks,
                upload_source="telegram"
            )

            self.telegram_bridge.send_message(f"✅ Successfully added '{file_name}' to database ({len(chunks)} chunks).")
            
            # Cleanup
            os.remove(local_file_path)
            
            # Refresh GUI if needed
            self.root.after(0, self.refresh_documents)

        except Exception as e:
            print(f"Error handling Telegram document: {e}")
            if self.telegram_bridge:
                self.telegram_bridge.send_message(f"❌ Error processing document: {str(e)}")

    def handle_disrupt_command(self, chat_id):
        """Handle /disrupt command from Telegram to stop processing immediately"""
        print("🛑 Disrupt command received from Telegram.")
        if self.telegram_bridge:
            self.telegram_bridge.send_message("🛑 Disrupting current process...")
        
        self.stop_processing_flag = True
        
        if self.decider:
            self.decider.report_forced_stop()
            
        def reset_flag():
            time.sleep(1.5) 
            self.stop_processing_flag = False
            print("▶️ Decider ready for next turn (Cooldown active).")
            if self.telegram_bridge:
                self.telegram_bridge.send_message("▶️ Process disrupted. Decider is in cooldown.")
            
        threading.Thread(target=reset_flag, daemon=True).start()

    def handle_telegram_text(self, msg: Dict):
        """Handle text message from Telegram"""
        # Reset status suppression on interaction
        self.telegram_status_sent = False

        # Check for disrupt command OR implicit disrupt on any message
        text_content = msg.get("text", "").strip().lower()
        is_explicit_disrupt = text_content == "/disrupt"
        
        if is_explicit_disrupt:
            self.handle_disrupt_command(msg["chat_id"])
            return

        # Show in UI
        self.root.after(0, lambda m=msg: self.add_chat_message(m["from"], m["text"], "incoming"))
        # Process logic
        threading.Thread(
            target=self.process_message_thread, 
            args=(msg["text"], False, msg["chat_id"]), # Use actual chat_id from msg
            daemon=True
        ).start()

    def handle_telegram_photo(self, msg: Dict):
        """Handle photo from Telegram"""
        try:
            file_id = msg["photo"]["file_id"]
            caption = msg.get("caption", "") or "Analyze this image."
            
            # Download to temp
            temp_path = f"./data/temp_img_{file_id}.jpg"
            file_data = self.telegram_bridge.get_file_info(file_id)
            self.telegram_bridge.download_file(file_data["file_path"], temp_path)
            
            self.root.after(0, lambda m=msg, c=caption, p=temp_path: self.add_chat_message(m["from"], c, "incoming", image_path=p))
            
            threading.Thread(
                target=self.process_message_thread,
                args=(caption, False, msg["chat_id"], temp_path),
                daemon=True
            ).start()
        except Exception as e:
            print(f"Error handling photo: {e}")

    def upload_documents(self):
        """Upload documents via GUI"""
        file_paths = filedialog.askopenfilenames(
            title="Select PDF or DOCX files",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("DOCX files", "*.docx"),
                ("All supported", "*.pdf *.docx")
            ]
        )

        if not file_paths:
            return

        def upload_thread():
            success_count = 0
            total_files = len(file_paths)

            # Only log if debug_log has been initialized
            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Starting upload of {total_files} document(s)")

            for i, file_path in enumerate(file_paths):
                try:
                    filename = os.path.basename(file_path)
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Processing ({i+1}/{total_files}): {filename}")

                    # Check for duplicates
                    file_hash = self.document_store.compute_file_hash(file_path)
                    if self.document_store.document_exists(file_hash):
                        if hasattr(self, 'debug_log'):
                            self.log_debug_message(f"Skipping duplicate: {filename}")
                        continue

                    # Process document
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Extracting text from: {filename}")
                    chunks, page_count, file_type = self.document_processor.process_document(file_path)
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Successfully extracted {len(chunks)} chunks from {filename} ({page_count} pages)")

                    # Add to store
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Adding document to store: {filename}")
                    self.document_store.add_document(
                        file_hash=file_hash,
                        filename=filename,
                        file_type=file_type,
                        file_size=os.path.getsize(file_path),
                        page_count=page_count,
                        chunks=chunks,
                        upload_source="desktop_gui"
                    )
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Successfully added: {filename}")

                    success_count += 1

                except Exception as e:
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                    print(f"Error processing {file_path}: {e}")

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Upload complete: {success_count}/{total_files} documents processed successfully")

            # Update UI in main thread
            self.root.after(0, lambda: self.refresh_documents())  # This will update original_docs
            self.root.after(0, lambda: self.status_var.set(f"Uploaded {success_count} documents"))

        threading.Thread(target=upload_thread, daemon=True).start()

    def start_background_processes(self):
        """Start background processes"""
        # Start memory consolidation loop
        threading.Thread(target=self.consolidation_loop, daemon=True).start()
        # Start daydreaming loop
        threading.Thread(target=self.daydream_loop, daemon=True).start()

    def consolidation_loop(self):
        """Periodic memory consolidation"""
        # Initial delay to ensure startup logs are visible
        time.sleep(1)
        
        while True:
            try:
                if hasattr(self, 'memory_store'):
                     self.memory_store.sanitize_sources()

                if hasattr(self, 'binah'):
                    stats = self.binah.consolidate(time_window_hours=None)
                    if stats['processed'] > 0:
                        print(f"🧠 [Consolidator] Processed: {stats['processed']}, Consolidated: {stats['consolidated']}, Skipped: {stats['skipped']}")
                        
                        # Learn associations after consolidation
                        if hasattr(self.binah, 'learn_associations'):
                            self.binah.learn_associations(self.reasoning_store)
                        
                        if self.hod:
                            self.hod.reflect("Consolidation")
                
                # Prune old operational meta-memories (keep last 3 days of logs)
                if hasattr(self, 'meta_memory_store'):
                    # 3 days = 259200 seconds
                    pruned_count = self.meta_memory_store.prune_events(max_age_seconds=259200, prune_all=False)
                    if pruned_count > 0:
                        print(f"🧹 [Meta-Memory] Pruned {pruned_count} old events (ALL types).")

                # Auto-vacuum to reclaim space
                if hasattr(self, 'memory_store'):
                    self.memory_store.vacuum()

                # Da'at Topic Lattice (Entity Summarization)
                if hasattr(self, 'daat') and self.daat:
                    self.daat.run_topic_lattice()
                    self.daat.monitor_model_tension()

                # Evaluate Keter (System Coherence)
                if hasattr(self, 'keter') and self.keter:
                    self.keter.evaluate()

                # Run Evolution Cycle (Liquid Prompts)
                if hasattr(self, 'ai_core'):
                    self.ai_core.run_evolution_cycle()
                    self.ai_core.generate_daily_self_narrative()

            except Exception as e:
                print(f"Consolidation/Cleanup error: {e}")
            
            time.sleep(600) # 10 minutes

    def daydream_loop(self):
        """Continuous daydreaming loop"""
        time.sleep(2)  # Initial buffer
        last_watchdog_check = time.time()
        
        while True:
            try:
                # Check stop flag
                if self.stop_processing_flag:
                    time.sleep(0.01)
                    continue

                # Watchdog: Check for coma state every 60 seconds
                if time.time() - last_watchdog_check > 60:
                    last_watchdog_check = time.time()
                    if self.decider and self.decider.current_task == "wait":
                        # If waiting for > 5 minutes (300s)
                        if self.decider.wait_start_time > 0 and (time.time() - self.decider.wait_start_time > 300):
                            print("⏰ Watchdog: System dormant for >5m. Forcing Pulse.")
                            self.decider.wake_up("Watchdog Pulse")

                # Check if Decider has work to do
                has_work = False
                if self.decider and self.decider.current_task != "wait":
                    has_work = True
                
                # Manage UI state based on work
                # Ensure chat input is enabled (allow chatting while daydreaming)
                self.root.after(0, lambda: self.toggle_chat_input(True))

                # If has work, try to acquire lock and run
                if not self.is_processing:
                    if has_work:
                        if self.processing_lock.acquire(blocking=False):
                            try:
                                # Double check inside lock
                                if self.is_processing: continue
                                
                                if self.decider:
                                    self.is_processing = True
                                    
                                    # Update status for UI
                                    task = self.decider.current_task.capitalize()
                                    remaining = self.decider.cycles_remaining
                                    status_msg = f"Active: {task} ({remaining} left)"
                                    self.root.after(0, lambda: self.status_var.set(status_msg))
                                    
                                    # Decider rules the loop
                                    self.decider.run_cycle()
                            finally:
                                self.is_processing = False
                                self.processing_lock.release()
                                self.root.after(0, lambda: self.status_var.set("Ready"))
                        else:
                            time.sleep(0.1)
                    else:
                        # No work, just sleep
                        # Run observer if idle to maintain "always working" state
                        if self.observer and self.decider:
                            signal = self.observer.perform_observation()
                            # Feed signal to Decider (which may wake up if pressure is high)
                            self.decider.ingest_netzach_signal(signal)
                            # Check for autonomous agency (Curiosity/Study)
                            self.ai_core.run_autonomous_agency_check(signal)
                        time.sleep(1.0)
                else:
                    time.sleep(0.01)
                        
            except Exception as e:
                print(f"Daydream loop error: {e}")
                time.sleep(1)

    def sync_journal(self):
        """
        Compile all Assistant Notes into a journal file and ingest it into the Document Store.
        This allows the AI to 'read' its own journal via RAG.
        """
        try:
            # 1. Fetch Notes
            items = self.memory_store.list_recent(limit=None)
            # Filter for NOTE type
            notes = [item for item in items if item[1] == "NOTE"]
            # Sort chronologically (list_recent is DESC, so reverse)
            notes.reverse()

            if not notes:
                messagebox.showinfo("Journal", "No journal entries (Notes) to sync.")
                return

            # 2. Create File Content
            content = "ASSISTANT JOURNAL\n=================\n\n"
            for note in notes:
                # note: (id, type, subject, text, ...)
                content += f"Entry [ID:{note[0]}]:\n{note[3]}\n\n" + ("-"*30) + "\n\n"

            # 3. Write to Docs Folder
            docs_dir = "./data/uploaded_docs"
            os.makedirs(docs_dir, exist_ok=True)
            filename = "assistant_journal.txt"
            file_path = os.path.join(docs_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 4. Ingest into Document Store
            # Remove old version if exists to ensure clean update
            old_id = self.document_store.get_document_by_filename(filename)
            if old_id:
                self.document_store.delete_document(old_id)
            
            # Process and Add
            file_hash = self.document_store.compute_file_hash(file_path)
            chunks, page_count, file_type = self.document_processor.process_document(file_path)
            
            self.document_store.add_document(
                file_hash=file_hash,
                filename=filename,
                file_type=file_type,
                file_size=os.path.getsize(file_path),
                page_count=page_count,
                chunks=chunks,
                upload_source="journal_sync"
            )
            
            self.refresh_documents()
            messagebox.showinfo("Journal Sync", f"Journal synced to documents ({len(notes)} entries).")
            
        except Exception as e:
            messagebox.showerror("Journal Sync Error", f"Failed to sync journal: {e}")

def main():
    root = tk.Tk()
    app = DesktopAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()