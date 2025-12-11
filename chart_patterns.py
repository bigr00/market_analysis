"""
Chart pattern detection module
Detects multi-candle chart patterns like Head & Shoulders, Double Top/Bottom, etc.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

from config import SignalType


@dataclass
class ChartPattern:
    """Represents a detected chart pattern"""
    name: str
    signal_type: SignalType
    confidence: float
    start_index: int
    end_index: int
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    neckline: Optional[float] = None


class ChartPatternDetector:
    """Detects chart patterns from price data"""

    def __init__(self, df: pd.DataFrame, order: int = 5):
        """
        Initialize with OHLCV DataFrame

        Args:
            df: DataFrame with open, high, low, close, volume columns
            order: Number of candles on each side to compare for pivot detection
        """
        self.df = df.copy()
        self.order = order
        self._find_pivots()

    def _find_pivots(self) -> None:
        """Find local highs and lows (pivot points)"""
        highs = self.df['high'].values
        lows = self.df['low'].values

        # Find local maxima and minima
        self.pivot_high_idx = argrelextrema(highs, np.greater, order=self.order)[0]
        self.pivot_low_idx = argrelextrema(lows, np.less, order=self.order)[0]

        # Store pivot values
        self.pivot_highs = [(idx, highs[idx]) for idx in self.pivot_high_idx]
        self.pivot_lows = [(idx, lows[idx]) for idx in self.pivot_low_idx]

    def detect_all_patterns(self, lookback: int = 100) -> List[ChartPattern]:
        """
        Detect all chart patterns in recent data

        Args:
            lookback: Number of candles to analyze

        Returns:
            List of detected chart patterns
        """
        patterns = []

        # Get recent pivots within lookback period
        start_idx = max(0, len(self.df) - lookback)

        patterns.extend(self._detect_double_top(start_idx))
        patterns.extend(self._detect_double_bottom(start_idx))
        patterns.extend(self._detect_head_shoulders(start_idx))
        patterns.extend(self._detect_inverse_head_shoulders(start_idx))
        patterns.extend(self._detect_ascending_triangle(start_idx))
        patterns.extend(self._detect_descending_triangle(start_idx))
        patterns.extend(self._detect_support_resistance(start_idx))
        patterns.extend(self._detect_trend_channel(start_idx))

        return patterns

    def _detect_double_top(self, start_idx: int, tolerance: float = 0.02) -> List[ChartPattern]:
        """
        Detect Double Top pattern (bearish reversal)

        Two peaks at approximately the same level with a trough between
        """
        patterns = []
        recent_highs = [(idx, val) for idx, val in self.pivot_highs if idx >= start_idx]

        if len(recent_highs) < 2:
            return patterns

        for i in range(len(recent_highs) - 1):
            first_peak_idx, first_peak = recent_highs[i]
            second_peak_idx, second_peak = recent_highs[i + 1]

            # Check if peaks are at similar levels
            if abs(first_peak - second_peak) / first_peak > tolerance:
                continue

            # Find trough between peaks
            trough_between = [
                (idx, val) for idx, val in self.pivot_lows
                if first_peak_idx < idx < second_peak_idx
            ]

            if not trough_between:
                continue

            neckline = min(val for _, val in trough_between)
            pattern_height = ((first_peak + second_peak) / 2) - neckline

            # Check if price has broken below neckline
            current_price = self.df['close'].iloc[-1]
            confirmed = current_price < neckline

            if confirmed:
                target = neckline - pattern_height
                patterns.append(ChartPattern(
                    name="Double Top",
                    signal_type=SignalType.STRONG_BEARISH,
                    confidence=0.75,
                    start_index=first_peak_idx,
                    end_index=second_peak_idx,
                    description=f"Bearish reversal pattern. Neckline: {neckline:.2f}",
                    target_price=target,
                    stop_loss=max(first_peak, second_peak) * 1.01,
                    neckline=neckline
                ))
            else:
                # Pattern forming but not confirmed
                patterns.append(ChartPattern(
                    name="Double Top (Forming)",
                    signal_type=SignalType.WEAK_BEARISH,
                    confidence=0.5,
                    start_index=first_peak_idx,
                    end_index=second_peak_idx,
                    description=f"Potential bearish reversal. Watch neckline: {neckline:.2f}",
                    neckline=neckline
                ))

        return patterns

    def _detect_double_bottom(self, start_idx: int, tolerance: float = 0.02) -> List[ChartPattern]:
        """
        Detect Double Bottom pattern (bullish reversal)

        Two troughs at approximately the same level with a peak between
        """
        patterns = []
        recent_lows = [(idx, val) for idx, val in self.pivot_lows if idx >= start_idx]

        if len(recent_lows) < 2:
            return patterns

        for i in range(len(recent_lows) - 1):
            first_bottom_idx, first_bottom = recent_lows[i]
            second_bottom_idx, second_bottom = recent_lows[i + 1]

            # Check if bottoms are at similar levels
            if abs(first_bottom - second_bottom) / first_bottom > tolerance:
                continue

            # Find peak between bottoms
            peak_between = [
                (idx, val) for idx, val in self.pivot_highs
                if first_bottom_idx < idx < second_bottom_idx
            ]

            if not peak_between:
                continue

            neckline = max(val for _, val in peak_between)
            pattern_height = neckline - ((first_bottom + second_bottom) / 2)

            # Check if price has broken above neckline
            current_price = self.df['close'].iloc[-1]
            confirmed = current_price > neckline

            if confirmed:
                target = neckline + pattern_height
                patterns.append(ChartPattern(
                    name="Double Bottom",
                    signal_type=SignalType.STRONG_BULLISH,
                    confidence=0.75,
                    start_index=first_bottom_idx,
                    end_index=second_bottom_idx,
                    description=f"Bullish reversal pattern. Neckline: {neckline:.2f}",
                    target_price=target,
                    stop_loss=min(first_bottom, second_bottom) * 0.99,
                    neckline=neckline
                ))
            else:
                patterns.append(ChartPattern(
                    name="Double Bottom (Forming)",
                    signal_type=SignalType.WEAK_BULLISH,
                    confidence=0.5,
                    start_index=first_bottom_idx,
                    end_index=second_bottom_idx,
                    description=f"Potential bullish reversal. Watch neckline: {neckline:.2f}",
                    neckline=neckline
                ))

        return patterns

    def _detect_head_shoulders(self, start_idx: int, tolerance: float = 0.03) -> List[ChartPattern]:
        """
        Detect Head and Shoulders pattern (bearish reversal)

        Three peaks: left shoulder, head (highest), right shoulder
        """
        patterns = []
        recent_highs = [(idx, val) for idx, val in self.pivot_highs if idx >= start_idx]

        if len(recent_highs) < 3:
            return patterns

        for i in range(len(recent_highs) - 2):
            left_idx, left_shoulder = recent_highs[i]
            head_idx, head = recent_highs[i + 1]
            right_idx, right_shoulder = recent_highs[i + 2]

            # Head must be higher than shoulders
            if not (head > left_shoulder and head > right_shoulder):
                continue

            # Shoulders should be at similar levels
            if abs(left_shoulder - right_shoulder) / left_shoulder > tolerance:
                continue

            # Find troughs between peaks for neckline
            left_trough = [val for idx, val in self.pivot_lows if left_idx < idx < head_idx]
            right_trough = [val for idx, val in self.pivot_lows if head_idx < idx < right_idx]

            if not left_trough or not right_trough:
                continue

            neckline = (min(left_trough) + min(right_trough)) / 2
            pattern_height = head - neckline

            current_price = self.df['close'].iloc[-1]
            confirmed = current_price < neckline

            if confirmed:
                target = neckline - pattern_height
                patterns.append(ChartPattern(
                    name="Head and Shoulders",
                    signal_type=SignalType.STRONG_BEARISH,
                    confidence=0.85,
                    start_index=left_idx,
                    end_index=right_idx,
                    description=f"Strong bearish reversal. Target: {target:.2f}",
                    target_price=target,
                    stop_loss=head * 1.01,
                    neckline=neckline
                ))
            else:
                patterns.append(ChartPattern(
                    name="Head and Shoulders (Forming)",
                    signal_type=SignalType.BEARISH,
                    confidence=0.6,
                    start_index=left_idx,
                    end_index=right_idx,
                    description=f"Potential H&S pattern. Neckline: {neckline:.2f}",
                    neckline=neckline
                ))

        return patterns

    def _detect_inverse_head_shoulders(self, start_idx: int, tolerance: float = 0.03) -> List[ChartPattern]:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal)

        Three troughs: left shoulder, head (lowest), right shoulder
        """
        patterns = []
        recent_lows = [(idx, val) for idx, val in self.pivot_lows if idx >= start_idx]

        if len(recent_lows) < 3:
            return patterns

        for i in range(len(recent_lows) - 2):
            left_idx, left_shoulder = recent_lows[i]
            head_idx, head = recent_lows[i + 1]
            right_idx, right_shoulder = recent_lows[i + 2]

            # Head must be lower than shoulders
            if not (head < left_shoulder and head < right_shoulder):
                continue

            # Shoulders should be at similar levels
            if abs(left_shoulder - right_shoulder) / left_shoulder > tolerance:
                continue

            # Find peaks between troughs for neckline
            left_peak = [val for idx, val in self.pivot_highs if left_idx < idx < head_idx]
            right_peak = [val for idx, val in self.pivot_highs if head_idx < idx < right_idx]

            if not left_peak or not right_peak:
                continue

            neckline = (max(left_peak) + max(right_peak)) / 2
            pattern_height = neckline - head

            current_price = self.df['close'].iloc[-1]
            confirmed = current_price > neckline

            if confirmed:
                target = neckline + pattern_height
                patterns.append(ChartPattern(
                    name="Inverse Head and Shoulders",
                    signal_type=SignalType.STRONG_BULLISH,
                    confidence=0.85,
                    start_index=left_idx,
                    end_index=right_idx,
                    description=f"Strong bullish reversal. Target: {target:.2f}",
                    target_price=target,
                    stop_loss=head * 0.99,
                    neckline=neckline
                ))
            else:
                patterns.append(ChartPattern(
                    name="Inverse H&S (Forming)",
                    signal_type=SignalType.BULLISH,
                    confidence=0.6,
                    start_index=left_idx,
                    end_index=right_idx,
                    description=f"Potential IH&S pattern. Neckline: {neckline:.2f}",
                    neckline=neckline
                ))

        return patterns

    def _detect_ascending_triangle(self, start_idx: int, tolerance: float = 0.02) -> List[ChartPattern]:
        """
        Detect Ascending Triangle pattern (bullish continuation)

        Flat resistance line with rising support (higher lows)
        """
        patterns = []
        recent_highs = [(idx, val) for idx, val in self.pivot_highs if idx >= start_idx]
        recent_lows = [(idx, val) for idx, val in self.pivot_lows if idx >= start_idx]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return patterns

        # Check for flat resistance (similar highs)
        high_values = [val for _, val in recent_highs[-4:]]
        if len(high_values) >= 2:
            resistance = np.mean(high_values)
            high_variance = np.std(high_values) / resistance

            if high_variance < tolerance:
                # Check for rising lows
                low_values = [val for _, val in recent_lows[-4:]]
                if len(low_values) >= 2 and all(low_values[i] <= low_values[i+1] for i in range(len(low_values)-1)):
                    current_price = self.df['close'].iloc[-1]

                    # Breakout above resistance
                    if current_price > resistance * 1.01:
                        pattern_height = resistance - min(low_values)
                        target = resistance + pattern_height

                        patterns.append(ChartPattern(
                            name="Ascending Triangle Breakout",
                            signal_type=SignalType.STRONG_BULLISH,
                            confidence=0.7,
                            start_index=recent_lows[-4][0] if len(recent_lows) >= 4 else recent_lows[0][0],
                            end_index=len(self.df) - 1,
                            description=f"Bullish breakout above {resistance:.2f}",
                            target_price=target,
                            stop_loss=resistance * 0.98
                        ))
                    else:
                        patterns.append(ChartPattern(
                            name="Ascending Triangle",
                            signal_type=SignalType.WEAK_BULLISH,
                            confidence=0.55,
                            start_index=recent_lows[-4][0] if len(recent_lows) >= 4 else recent_lows[0][0],
                            end_index=len(self.df) - 1,
                            description=f"Forming. Resistance at {resistance:.2f}"
                        ))

        return patterns

    def _detect_descending_triangle(self, start_idx: int, tolerance: float = 0.02) -> List[ChartPattern]:
        """
        Detect Descending Triangle pattern (bearish continuation)

        Flat support line with falling resistance (lower highs)
        """
        patterns = []
        recent_highs = [(idx, val) for idx, val in self.pivot_highs if idx >= start_idx]
        recent_lows = [(idx, val) for idx, val in self.pivot_lows if idx >= start_idx]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return patterns

        # Check for flat support (similar lows)
        low_values = [val for _, val in recent_lows[-4:]]
        if len(low_values) >= 2:
            support = np.mean(low_values)
            low_variance = np.std(low_values) / support

            if low_variance < tolerance:
                # Check for falling highs
                high_values = [val for _, val in recent_highs[-4:]]
                if len(high_values) >= 2 and all(high_values[i] >= high_values[i+1] for i in range(len(high_values)-1)):
                    current_price = self.df['close'].iloc[-1]

                    # Breakdown below support
                    if current_price < support * 0.99:
                        pattern_height = max(high_values) - support
                        target = support - pattern_height

                        patterns.append(ChartPattern(
                            name="Descending Triangle Breakdown",
                            signal_type=SignalType.STRONG_BEARISH,
                            confidence=0.7,
                            start_index=recent_highs[-4][0] if len(recent_highs) >= 4 else recent_highs[0][0],
                            end_index=len(self.df) - 1,
                            description=f"Bearish breakdown below {support:.2f}",
                            target_price=target,
                            stop_loss=support * 1.02
                        ))
                    else:
                        patterns.append(ChartPattern(
                            name="Descending Triangle",
                            signal_type=SignalType.WEAK_BEARISH,
                            confidence=0.55,
                            start_index=recent_highs[-4][0] if len(recent_highs) >= 4 else recent_highs[0][0],
                            end_index=len(self.df) - 1,
                            description=f"Forming. Support at {support:.2f}"
                        ))

        return patterns

    def _detect_support_resistance(self, start_idx: int, tolerance: float = 0.01) -> List[ChartPattern]:
        """
        Detect key support and resistance levels
        """
        patterns = []
        current_price = self.df['close'].iloc[-1]

        # Find clusters of pivot highs and lows
        recent_highs = [val for idx, val in self.pivot_highs if idx >= start_idx]
        recent_lows = [val for idx, val in self.pivot_lows if idx >= start_idx]

        if not recent_highs or not recent_lows:
            return patterns

        # Find resistance levels near current price
        for high in recent_highs:
            if current_price < high and (high - current_price) / current_price < 0.05:
                # Count how many times this level was tested
                tests = sum(1 for h in recent_highs if abs(h - high) / high < tolerance)
                if tests >= 2:
                    patterns.append(ChartPattern(
                        name=f"Resistance Level ({tests}x tested)",
                        signal_type=SignalType.WEAK_BEARISH,
                        confidence=min(0.3 + tests * 0.1, 0.7),
                        start_index=start_idx,
                        end_index=len(self.df) - 1,
                        description=f"Key resistance at {high:.2f}",
                        target_price=high
                    ))
                    break  # Only report the nearest resistance

        # Find support levels near current price
        for low in sorted(recent_lows, reverse=True):
            if current_price > low and (current_price - low) / current_price < 0.05:
                tests = sum(1 for l in recent_lows if abs(l - low) / low < tolerance)
                if tests >= 2:
                    patterns.append(ChartPattern(
                        name=f"Support Level ({tests}x tested)",
                        signal_type=SignalType.WEAK_BULLISH,
                        confidence=min(0.3 + tests * 0.1, 0.7),
                        start_index=start_idx,
                        end_index=len(self.df) - 1,
                        description=f"Key support at {low:.2f}",
                        target_price=low
                    ))
                    break

        return patterns

    def _detect_trend_channel(self, start_idx: int, min_points: int = 3) -> List[ChartPattern]:
        """
        Detect trend channels (parallel support and resistance lines)
        """
        patterns = []
        recent_highs = [(idx, val) for idx, val in self.pivot_highs if idx >= start_idx]
        recent_lows = [(idx, val) for idx, val in self.pivot_lows if idx >= start_idx]

        if len(recent_highs) < min_points or len(recent_lows) < min_points:
            return patterns

        # Fit linear regression to highs and lows
        high_x = np.array([idx for idx, _ in recent_highs])
        high_y = np.array([val for _, val in recent_highs])
        low_x = np.array([idx for idx, _ in recent_lows])
        low_y = np.array([val for _, val in recent_lows])

        if len(high_x) < 2 or len(low_x) < 2:
            return patterns

        # Calculate slopes
        high_slope = np.polyfit(high_x, high_y, 1)[0]
        low_slope = np.polyfit(low_x, low_y, 1)[0]

        # Check if slopes are similar (parallel lines)
        avg_price = (np.mean(high_y) + np.mean(low_y)) / 2
        slope_diff = abs(high_slope - low_slope) / avg_price

        if slope_diff < 0.0001:  # Nearly parallel
            avg_slope = (high_slope + low_slope) / 2

            if avg_slope > 0:
                patterns.append(ChartPattern(
                    name="Ascending Channel",
                    signal_type=SignalType.BULLISH,
                    confidence=0.6,
                    start_index=start_idx,
                    end_index=len(self.df) - 1,
                    description="Price moving in upward channel"
                ))
            elif avg_slope < 0:
                patterns.append(ChartPattern(
                    name="Descending Channel",
                    signal_type=SignalType.BEARISH,
                    confidence=0.6,
                    start_index=start_idx,
                    end_index=len(self.df) - 1,
                    description="Price moving in downward channel"
                ))
            else:
                patterns.append(ChartPattern(
                    name="Horizontal Channel",
                    signal_type=SignalType.NEUTRAL,
                    confidence=0.5,
                    start_index=start_idx,
                    end_index=len(self.df) - 1,
                    description="Price ranging sideways"
                ))

        return patterns


def get_chart_pattern_summary(patterns: List[ChartPattern]) -> Tuple[float, str]:
    """
    Get an overall signal score and summary from detected patterns

    Returns:
        score: -1.0 (strong bearish) to +1.0 (strong bullish)
        summary: Text description
    """
    if not patterns:
        return 0.0, "No chart patterns detected"

    score = 0.0
    for pattern in patterns:
        weight = pattern.confidence

        if pattern.signal_type in [SignalType.STRONG_BULLISH, SignalType.BULLISH, SignalType.WEAK_BULLISH]:
            if pattern.signal_type == SignalType.STRONG_BULLISH:
                score += weight * 1.5
            elif pattern.signal_type == SignalType.BULLISH:
                score += weight
            else:
                score += weight * 0.5
        elif pattern.signal_type in [SignalType.STRONG_BEARISH, SignalType.BEARISH, SignalType.WEAK_BEARISH]:
            if pattern.signal_type == SignalType.STRONG_BEARISH:
                score -= weight * 1.5
            elif pattern.signal_type == SignalType.BEARISH:
                score -= weight
            else:
                score -= weight * 0.5

    # Normalize
    max_possible = sum(p.confidence * 1.5 for p in patterns)
    if max_possible > 0:
        score = score / max_possible

    # Clamp to -1 to 1
    score = max(-1.0, min(1.0, score))

    if score > 0.5:
        summary = "Strong bullish chart patterns"
    elif score > 0.2:
        summary = "Moderate bullish chart patterns"
    elif score > 0:
        summary = "Weak bullish chart patterns"
    elif score > -0.2:
        summary = "Weak bearish chart patterns"
    elif score > -0.5:
        summary = "Moderate bearish chart patterns"
    else:
        summary = "Strong bearish chart patterns"

    return score, summary
