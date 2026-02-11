# E:/CodingWorkspace/Projects/AITelegramIntegration/event_bus.py
from typing import Callable, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class Event:
    """Standard event packet."""
    type: str
    data: Any = None
    source: str = "System"
    priority: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventBus:
    """
    Central nervous system for the architecture.
    Decouples components by allowing them to publish/subscribe to events.
    Supports prioritization and activity logging.
    """

    def __init__(self):
        # Subscribers: Dict[event_type, List[(priority, callback)]]
        self._subscribers: Dict[str, List[Tuple[int, Callable[[Event], None]]]] = {}
        self._history: List[Event] = []
        self._history_limit = 1000

    def subscribe(self, event_type: str, callback: Callable[[Event], None], priority: int = 0):
        """
        Register a callback for a specific event type.
        Priority: Higher values run first. Default 0.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append((priority, callback))
        # Sort by priority descending (High priority first)
        self._subscribers[event_type].sort(key=lambda x: x[0], reverse=True)

    def publish(self, event_type: str, data: Any = None, source: str = "System", priority: int = 0):
        """Broadcast an event to all subscribers."""
        event = Event(type=event_type, data=data, source=source, priority=priority)

        # Log to history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)

        # Notify specific listeners
        if event_type in self._subscribers:
            for _, callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"❌ Event Bus Error processing '{event_type}': {e}")

        # Notify wildcard listeners (optional, good for logging)
        if "*" in self._subscribers:
            for _, callback in self._subscribers["*"]:
                try:
                    callback(event)
                except Exception:
                    pass

    def get_history(self, limit: int = 100) -> List[Event]:
        """Get recent event history for visualization."""
        return self._history[-limit:]