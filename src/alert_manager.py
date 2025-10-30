"""
Alert Manager Module
------------------
Handles desktop notifications, sound alerts, and visual indicators for the floor monitoring system.
"""
import logging
import threading
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Try to import win10toast for Windows notifications
try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False
    ToastNotifier = None

# Try to import playsound for sound alerts
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    playsound = None

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Represents an alert to be displayed to the user."""
    alert_id: str
    title: str
    message: str
    alert_type: str  # 'warning', 'alert', 'info'
    timestamp: float
    acknowledged: bool = False
    worker_id: Optional[int] = None
    duration: Optional[int] = None

class AlertManager:
    """Manages alerts and notifications for the floor monitoring system."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the alert manager."""
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        self.running = True
        
        # Initialize toast notifier if available
        self.toaster = ToastNotifier() if TOAST_AVAILABLE and ToastNotifier is not None else None
        
        # Alert configuration
        self.notifications_enabled = self.config.get("notifications", {}).get("enabled", True)
        self.sound_enabled = self.config.get("notifications", {}).get("sound", True)
        
        logger.info(f"Initialized AlertManager (notifications: {self.notifications_enabled}, "
                   f"sound: {self.sound_enabled})")
    
    def add_alert(self, title: str, message: str, alert_type: str = "info", 
                  worker_id: Optional[int] = None, duration: Optional[int] = None) -> str:
        """Add a new alert to be displayed.
        
        Args:
            title: Alert title
            message: Alert message
            alert_type: Type of alert ('warning', 'alert', 'info')
            worker_id: Associated worker ID (optional)
            duration: Duration in seconds (optional)
            
        Returns:
            Alert ID
        """
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
        
        self.alerts.append(alert)
        logger.info(f"New {alert_type} alert: {title} - {message}")
        
        # Show notification if enabled
        if self.notifications_enabled:
            self._show_notification(alert)
        
        # Play sound if enabled
        if self.sound_enabled:
            self._play_alert_sound(alert_type)
        
        return alert_id
    
    def _show_notification(self, alert: Alert) -> None:
        """Show a desktop notification for the alert."""
        if not self.toaster:
            return
            
        try:
            # Duration in milliseconds (0 = system default)
            duration_ms = alert.duration * 1000 if alert.duration else 0
            
            # Show toast notification
            self.toaster.show_toast(
                alert.title,
                alert.message,
                duration=duration_ms,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Failed to show notification: {str(e)}")
    
    def _play_alert_sound(self, alert_type: str) -> None:
        """Play an alert sound based on the alert type."""
        if not PLAYSOUND_AVAILABLE or not playsound:
            return
            
        try:
            # Define sound paths based on alert type
            sound_path = None
            if alert_type == "alert":
                sound_path = "assets/sounds/alert.wav"
            elif alert_type == "warning":
                sound_path = "assets/sounds/warning.wav"
            elif alert_type == "info":
                sound_path = "assets/sounds/info.wav"
            
            # Play sound if path exists
            if sound_path and os.path.exists(sound_path):
                playsound(sound_path)
            elif sound_path:
                logger.warning(f"Sound file not found: {sound_path}")
            else:
                # Default sound if no specific sound is defined
                logger.info(f"No specific sound for {alert_type}, playing default")
        except Exception as e:
            logger.error(f"Failed to play sound: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        for i, alert in enumerate(self.alerts):
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                # Move to history
                self.alert_history.append(self.alerts.pop(i))
                # Keep history within limits
                if len(self.alert_history) > self.max_history:
                    self.alert_history.pop(0)
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def clear_alerts(self) -> int:
        """Clear all current alerts and return count."""
        count = len(self.alerts)
        # Move all alerts to history
        self.alert_history.extend(self.alerts)
        # Keep history within limits
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        self.alerts.clear()
        logger.info(f"Cleared {count} alerts")
        return count
    
    def stop(self) -> None:
        """Stop the alert manager."""
        self.running = False
        logger.info("AlertManager stopped")


def test_alert_manager():
    """Test function for the AlertManager class."""
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create alert manager
    config = {
        "notifications": {
            "enabled": True,
            "sound": True
        }
    }
    
    alert_manager = AlertManager(config)
    
    print("Testing AlertManager. Press Ctrl+C to stop.")
    
    try:
        # Add some test alerts
        alert_manager.add_alert(
            "Worker Alert",
            "Worker John Doe has been present for over 30 minutes",
            "alert",
            worker_id=1,
            duration=30
        )
        
        alert_manager.add_alert(
            "Worker Warning",
            "Worker Jane Smith has been present for over 15 minutes",
            "warning",
            worker_id=2,
            duration=15
        )
        
        alert_manager.add_alert(
            "System Info",
            "Application started successfully",
            "info"
        )
        
        # Show active alerts
        print("\nActive Alerts:")
        for alert in alert_manager.get_active_alerts():
            print(f"  {alert.title}: {alert.message} ({alert.alert_type})")
        
        # Wait a bit to see notifications
        time.sleep(5)
        
        # Acknowledge first alert
        active_alerts = alert_manager.get_active_alerts()
        if active_alerts:
            alert_id = active_alerts[0].alert_id
            alert_manager.acknowledge_alert(alert_id)
            print(f"\nAcknowledged alert {alert_id}")
        
        # Show remaining alerts
        print("\nRemaining Active Alerts:")
        for alert in alert_manager.get_active_alerts():
            print(f"  {alert.title}: {alert.message} ({alert.alert_type})")
        
        # Show alert history
        print("\nAlert History:")
        for alert in alert_manager.get_alert_history():
            status = "ACK" if alert.acknowledged else "NEW"
            print(f"  [{status}] {alert.title}: {alert.message}")
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        alert_manager.stop()


if __name__ == "__main__":
    test_alert_manager()