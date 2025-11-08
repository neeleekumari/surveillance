"""
Alert Manager (Enhanced + Thread-Safe)
-------------------------------------
Handles cross-platform notifications, alert sounds, and UI callbacks.
Designed for integration with the real-time 3D worker monitoring suite.
"""

import logging
import threading
import time
import os
import platform
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

# Optional dependencies
try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False
    ToastNotifier = None

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    playsound = None

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Data model for an active or archived alert."""
    alert_id: str
    title: str
    message: str
    alert_type: str  # 'warning', 'alert', 'info'
    timestamp: float
    acknowledged: bool = False
    worker_id: Optional[int] = None
    duration: Optional[int] = None
    expires_after: float = 300.0  # auto-expire after 5 minute


class AlertManager:
    """Manages alerts and notifications in real time."""

    def __init__(self, config: Optional[dict] = None, ui_callback: Optional[Callable[[str, str, str], None]] = None):
        """
        Args:
            config: Configuration dictionary for alert settings.
            ui_callback: Optional function to send alert logs to UI.
        """
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.max_history = 500
        self.running = True
        self.ui_callback = ui_callback
        self._lock = threading.Lock()
        self._last_sound_time: Dict[str, float] = {}

        # Notification options
        notify_cfg = self.config.get("notifications", {})
        self.notifications_enabled = notify_cfg.get("enabled", True)
        self.sound_enabled = notify_cfg.get("sound", True)
        self.sound_cooldown = notify_cfg.get("sound_cooldown", 5.0)

        # Initialize notifier
        if TOAST_AVAILABLE and platform.system() == "Windows":
            try:
                self.toaster = ToastNotifier()
            except Exception:
                self.toaster = None
        else:
            self.toaster = None

        logger.info(f"AlertManager initialized (notifications={self.notifications_enabled}, sound={self.sound_enabled})")

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    # --------------------------------------------------------------
    def add_alert(self, title: str, message: str, alert_type: str = "info",
                  worker_id: Optional[int] = None, duration: Optional[int] = None) -> str:
        """Register a new alert."""
        alert_id = f"{int(time.time() * 1000)}"
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            alert_type=alert_type,
            timestamp=time.time(),
            worker_id=worker_id,
            duration=duration
        )

        with self._lock:
            self.alerts.append(alert)
            if len(self.alerts) > 100:
                # Prevent unbounded growth
                oldest = self.alerts.pop(0)
                self.alert_history.append(oldest)

        logger.info(f"[{alert_type.upper()}] {title}: {message}")

        # Send to UI if available
        if self.ui_callback:
            color_map = {"alert": "red", "warning": "orange", "info": "blue"}
            self.ui_callback(f"{title}: {message}", color_map.get(alert_type, "black"), alert_type)

        # Show notification
        if self.notifications_enabled:
            threading.Thread(target=self._show_notification, args=(alert,), daemon=True).start()

        # Play sound
        if self.sound_enabled:
            threading.Thread(target=self._play_alert_sound, args=(alert_type,), daemon=True).start()

        return alert_id

    # --------------------------------------------------------------
    def _show_notification(self, alert: Alert):
        """Display desktop notification (Windows only)."""
        if not self.toaster:
            return
        try:
            self.toaster.show_toast(
                alert.title,
                alert.message,
                duration=alert.duration or 5,
                threaded=True
            )
        except Exception as e:
            logger.debug(f"Toast notification failed: {e}")

    # --------------------------------------------------------------
    def _play_alert_sound(self, alert_type: str):
        """Play non-blocking alert sound (rate-limited)."""
        if not PLAYSOUND_AVAILABLE or not playsound:
            return

        now = time.time()
        last_time = self._last_sound_time.get(alert_type, 0)
        if now - last_time < self.sound_cooldown:
            return
        self._last_sound_time[alert_type] = now

        try:
            # Try multiple sound file locations and formats
            sound_candidates = [
                "assets/Beep.mpeg",  # User's existing beep file
            ]
            
            # Use the first available sound file
            path = None
            for candidate in sound_candidates:
                if os.path.exists(candidate):
                    path = candidate
                    break
            
            if path:
                logger.info(f"Playing alert sound: {path}")
                # Use blocking playback in daemon thread to avoid Windows WPARAM errors
                def play_blocking():
                    try:
                        playsound(path, block=True)
                    except Exception as e:
                        logger.debug(f"Playsound error: {e}")
                threading.Thread(target=play_blocking, daemon=True).start()
            else:
                # Minimal fallback beep
                logger.debug("No sound file found, using system beep")
                if platform.system() == "Windows":
                    import winsound
                    winsound.MessageBeep()
        except Exception as e:
            logger.warning(f"Sound playback failed: {e}")

    # --------------------------------------------------------------
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        with self._lock:
            for i, alert in enumerate(self.alerts):
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.alert_history.append(self.alerts.pop(i))
                    self._trim_history()
                    logger.info(f"Acknowledged alert {alert_id}")
                    return True
        return False

    def _trim_history(self):
        """Keep alert history within limit."""
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    # --------------------------------------------------------------
    def get_active_alerts(self) -> List[Alert]:
        with self._lock:
            return [a for a in self.alerts if not a.acknowledged]

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        with self._lock:
            return self.alert_history[-limit:]

    def clear_alerts(self) -> int:
        """Clear all active alerts."""
        with self._lock:
            count = len(self.alerts)
            self.alert_history.extend(self.alerts)
            self.alerts.clear()
            self._trim_history()
        logger.info(f"Cleared {count} active alerts.")
        return count

    # --------------------------------------------------------------
    def _cleanup_loop(self):
        """Background thread to auto-expire old alerts."""
        while self.running:
            try:
                now = time.time()
                with self._lock:
                    expired = [a for a in self.alerts if now - a.timestamp > a.expires_after]
                    for a in expired:
                        logger.debug(f"Auto-archiving alert {a.alert_id}")
                        self.alerts.remove(a)
                        self.alert_history.append(a)
                        self._trim_history()
                time.sleep(5)
            except Exception as e:
                logger.debug(f"Alert cleanup loop error: {e}")
                time.sleep(5)

    # --------------------------------------------------------------
    def stop(self):
        """Stop manager and cleanup thread."""
        self.running = False
        logger.info("AlertManager stopped.")
