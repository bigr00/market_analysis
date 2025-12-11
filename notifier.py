"""
Audio notification module
Provides audio alerts for trading signals
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from config import TradingConfig, DEFAULT_TRADING_CONFIG


class NotificationType(Enum):
    """Types of notifications"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    ALERT = "alert"


@dataclass
class NotificationSound:
    """Represents a notification sound configuration"""
    name: str
    frequency: int  # Hz
    duration: float  # seconds
    pattern: str  # Pattern description


class AudioNotifier:
    """
    Handles audio notifications for trading signals

    Uses macOS 'afplay' for WAV files or system sounds,
    falls back to terminal bell on other systems
    """

    def __init__(self, config: TradingConfig = DEFAULT_TRADING_CONFIG):
        self.config = config
        self.enabled = config.enable_audio
        self.is_macos = sys.platform == 'darwin'

        # Sound configurations (frequency patterns for generated sounds)
        self.sounds = {
            NotificationType.BULLISH: NotificationSound(
                name="Bullish",
                frequency=880,  # A5 note
                duration=0.3,
                pattern="ascending"
            ),
            NotificationType.BEARISH: NotificationSound(
                name="Bearish",
                frequency=440,  # A4 note
                duration=0.3,
                pattern="descending"
            ),
            NotificationType.ALERT: NotificationSound(
                name="Alert",
                frequency=660,  # E5 note
                duration=0.2,
                pattern="double"
            ),
        }

    def notify(self, notification_type: NotificationType, message: Optional[str] = None) -> bool:
        """
        Play a notification sound

        Args:
            notification_type: Type of notification (bullish, bearish, alert)
            message: Optional message to speak (macOS only)

        Returns:
            True if notification was played successfully
        """
        if not self.enabled:
            return False

        try:
            if self.is_macos:
                return self._notify_macos(notification_type, message)
            else:
                return self._notify_fallback(notification_type)
        except Exception as e:
            print(f"[WARN] Audio notification failed: {e}")
            return False

    def _notify_macos(self, notification_type: NotificationType, message: Optional[str] = None) -> bool:
        """macOS-specific notification using system sounds and speech"""
        sound = self.sounds[notification_type]

        # Try to play system sound or generate tone
        if notification_type == NotificationType.BULLISH:
            # Ascending tone pattern (bullish)
            self._play_tone_sequence([660, 880, 1100], 0.15)
        elif notification_type == NotificationType.BEARISH:
            # Descending tone pattern (bearish)
            self._play_tone_sequence([880, 660, 440], 0.15)
        else:
            # Alert pattern
            self._play_tone_sequence([880, 880], 0.1)

        # Optionally speak the message
        if message:
            self._speak(message)

        return True

    def _play_tone_sequence(self, frequencies: list, duration: float) -> None:
        """Play a sequence of tones using macOS afplay with generated audio"""
        try:
            for freq in frequencies:
                # Use macOS 'say' command with a short beep alternative
                # Since generating actual audio requires more dependencies,
                # we'll use a simple system sound approach
                subprocess.run(
                    ['osascript', '-e', f'beep'],
                    capture_output=True,
                    timeout=1
                )
        except Exception:
            # Fall back to terminal bell
            print('\a', end='', flush=True)

    def _speak(self, message: str) -> None:
        """Use macOS text-to-speech"""
        try:
            # Sanitize message for shell
            safe_message = message.replace('"', '\\"').replace("'", "\\'")
            subprocess.run(
                ['say', '-v', 'Samantha', '-r', '200', safe_message],
                capture_output=True,
                timeout=10
            )
        except Exception:
            pass

    def _notify_fallback(self, notification_type: NotificationType) -> bool:
        """Fallback notification using terminal bell"""
        sound = self.sounds[notification_type]

        # Multiple bells for different notification types
        if notification_type == NotificationType.BULLISH:
            print('\a\a\a', end='', flush=True)  # Three beeps
        elif notification_type == NotificationType.BEARISH:
            print('\a\a', end='', flush=True)  # Two beeps
        else:
            print('\a', end='', flush=True)  # One beep

        return True

    def notify_bullish(self, strength: str = "", price: Optional[float] = None) -> bool:
        """Send bullish signal notification"""
        message = None
        if self.is_macos and price:
            message = f"Bullish {strength} signal at {price:.0f} dollars"
        return self.notify(NotificationType.BULLISH, message)

    def notify_bearish(self, strength: str = "", price: Optional[float] = None) -> bool:
        """Send bearish signal notification"""
        message = None
        if self.is_macos and price:
            message = f"Bearish {strength} signal at {price:.0f} dollars"
        return self.notify(NotificationType.BEARISH, message)

    def notify_alert(self, message: Optional[str] = None) -> bool:
        """Send general alert notification"""
        return self.notify(NotificationType.ALERT, message)

    def test_notifications(self) -> None:
        """Test all notification sounds"""
        print("Testing notification sounds...")

        print("1. Bullish signal...")
        self.notify_bullish("strong", 100000)
        import time
        time.sleep(1)

        print("2. Bearish signal...")
        self.notify_bearish("moderate", 95000)
        time.sleep(1)

        print("3. Alert...")
        self.notify_alert("Test alert")
        time.sleep(1)

        print("Notification test complete!")

    def enable(self) -> None:
        """Enable audio notifications"""
        self.enabled = True

    def disable(self) -> None:
        """Disable audio notifications"""
        self.enabled = False


class DesktopNotifier:
    """
    Desktop notification support (optional)
    Uses osascript on macOS for native notifications
    """

    def __init__(self):
        self.is_macos = sys.platform == 'darwin'

    def notify(self, title: str, message: str, sound: bool = True) -> bool:
        """
        Send a desktop notification

        Args:
            title: Notification title
            message: Notification body
            sound: Whether to play the default notification sound

        Returns:
            True if notification was sent successfully
        """
        if self.is_macos:
            return self._notify_macos(title, message, sound)
        return False

    def _notify_macos(self, title: str, message: str, sound: bool) -> bool:
        """Send macOS notification using osascript"""
        try:
            sound_str = 'with sound name "Ping"' if sound else ''
            script = f'''
            display notification "{message}" with title "{title}" {sound_str}
            '''
            subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                timeout=5
            )
            return True
        except Exception as e:
            print(f"[WARN] Desktop notification failed: {e}")
            return False

    def notify_signal(self, direction: str, strength: str, price: float,
                      score: float, timeframe: str) -> bool:
        """Send a trading signal notification"""
        title = f"BTC {direction.upper()} Signal - {strength}"
        message = f"Price: ${price:,.0f} | Score: {score:+.2f} | TF: {timeframe}"
        return self.notify(title, message)


if __name__ == "__main__":
    # Test notifications
    print("Testing Audio Notifier...")
    audio = AudioNotifier()
    audio.test_notifications()

    print("\nTesting Desktop Notifier...")
    desktop = DesktopNotifier()
    desktop.notify_signal("bullish", "strong", 100000, 0.75, "1h")
