"""
Signal aggregation module
Combines signals from candlestick patterns, chart patterns, and technical indicators
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd

from config import SignalType, TimeFrame, TradingConfig, DEFAULT_TRADING_CONFIG
from candlestick_patterns import CandlestickPattern, CandlestickPatternDetector, get_pattern_summary
from chart_patterns import ChartPattern, ChartPatternDetector, get_chart_pattern_summary
from technical_indicators import IndicatorSignal, TechnicalIndicators, get_indicator_summary


class SignalStrength(Enum):
    """Overall signal strength"""
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NEUTRAL = "NEUTRAL"


@dataclass
class AggregatedSignal:
    """Represents an aggregated trading signal"""
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    overall_score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0

    # Component scores
    candlestick_score: float
    chart_pattern_score: float
    indicator_score: float

    # Details
    candlestick_patterns: List[CandlestickPattern]
    chart_patterns: List[ChartPattern]
    indicator_signals: List[IndicatorSignal]

    # Price info
    current_price: float
    timeframe: TimeFrame

    # Summaries
    candlestick_summary: str
    chart_summary: str
    indicator_summary: str

    def get_description(self) -> str:
        """Get a human-readable description of the signal"""
        direction = "BULLISH" if self.overall_score > 0 else "BEARISH" if self.overall_score < 0 else "NEUTRAL"
        return f"{self.strength.value} {direction} signal (score: {self.overall_score:+.2f})"


@dataclass
class MultiTimeframeSignal:
    """Aggregated signal across multiple timeframes"""
    timestamp: datetime
    signals: Dict[TimeFrame, AggregatedSignal]
    combined_score: float
    combined_strength: SignalStrength
    primary_direction: SignalType
    alignment_score: float  # How well timeframes align (0-1)


class SignalAggregator:
    """Aggregates signals from multiple sources and timeframes"""

    def __init__(self, config: TradingConfig = DEFAULT_TRADING_CONFIG):
        self.config = config

        # Weights for different signal sources
        self.weights = {
            'candlestick': 0.25,
            'chart': 0.35,
            'indicator': 0.40
        }

        # Timeframe weights (higher timeframes more important)
        self.timeframe_weights = {
            TimeFrame.M1: 0.5,
            TimeFrame.M5: 0.6,
            TimeFrame.M15: 0.7,
            TimeFrame.M30: 0.8,
            TimeFrame.H1: 0.9,
            TimeFrame.H4: 1.0,
            TimeFrame.D1: 1.1
        }

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame) -> AggregatedSignal:
        """
        Analyze a single timeframe and generate aggregated signal

        Args:
            df: OHLCV DataFrame for the timeframe
            timeframe: The timeframe being analyzed

        Returns:
            AggregatedSignal with combined analysis
        """
        current_price = df['close'].iloc[-1]

        # Candlestick patterns
        candlestick_detector = CandlestickPatternDetector(df)
        candlestick_patterns = candlestick_detector.detect_all_patterns(lookback=5)
        candlestick_score, candlestick_summary = get_pattern_summary(candlestick_patterns)

        # Chart patterns
        chart_detector = ChartPatternDetector(df)
        chart_patterns = chart_detector.detect_all_patterns(lookback=100)
        chart_score, chart_summary = get_chart_pattern_summary(chart_patterns)

        # Technical indicators
        indicators = TechnicalIndicators(df)
        indicator_signals = indicators.analyze_signals()
        indicator_score, indicator_summary = get_indicator_summary(indicator_signals)

        # Calculate weighted overall score
        overall_score = (
            candlestick_score * self.weights['candlestick'] +
            chart_score * self.weights['chart'] +
            indicator_score * self.weights['indicator']
        )

        # Determine signal type
        if overall_score > 0.5:
            signal_type = SignalType.STRONG_BULLISH
        elif overall_score > 0.2:
            signal_type = SignalType.BULLISH
        elif overall_score > 0:
            signal_type = SignalType.WEAK_BULLISH
        elif overall_score > -0.2:
            signal_type = SignalType.WEAK_BEARISH
        elif overall_score > -0.5:
            signal_type = SignalType.BEARISH
        else:
            signal_type = SignalType.STRONG_BEARISH

        # Determine strength
        abs_score = abs(overall_score)
        if abs_score > 0.7:
            strength = SignalStrength.VERY_STRONG
        elif abs_score > 0.5:
            strength = SignalStrength.STRONG
        elif abs_score > 0.3:
            strength = SignalStrength.MODERATE
        elif abs_score > 0.1:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NEUTRAL

        # Calculate confidence based on signal agreement
        signals_agree = self._calculate_agreement(candlestick_score, chart_score, indicator_score)
        confidence = min(1.0, abs_score + signals_agree * 0.3)

        return AggregatedSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            strength=strength,
            overall_score=overall_score,
            confidence=confidence,
            candlestick_score=candlestick_score,
            chart_pattern_score=chart_score,
            indicator_score=indicator_score,
            candlestick_patterns=candlestick_patterns,
            chart_patterns=chart_patterns,
            indicator_signals=indicator_signals,
            current_price=current_price,
            timeframe=timeframe,
            candlestick_summary=candlestick_summary,
            chart_summary=chart_summary,
            indicator_summary=indicator_summary
        )

    def _calculate_agreement(self, *scores: float) -> float:
        """Calculate how well different signals agree (0-1)"""
        if not scores:
            return 0.0

        # Check if all scores have the same sign
        signs = [1 if s > 0 else -1 if s < 0 else 0 for s in scores]
        non_zero_signs = [s for s in signs if s != 0]

        if not non_zero_signs:
            return 0.5  # All neutral

        # All same direction = high agreement
        if all(s == non_zero_signs[0] for s in non_zero_signs):
            return 1.0

        # Mixed signals = low agreement
        positive = sum(1 for s in non_zero_signs if s > 0)
        negative = sum(1 for s in non_zero_signs if s < 0)
        return abs(positive - negative) / len(non_zero_signs)

    def analyze_multi_timeframe(self, data: Dict[TimeFrame, pd.DataFrame]) -> MultiTimeframeSignal:
        """
        Analyze multiple timeframes and combine signals

        Args:
            data: Dictionary mapping timeframes to their OHLCV DataFrames

        Returns:
            MultiTimeframeSignal with combined analysis
        """
        signals = {}
        weighted_scores = []
        total_weight = 0

        for timeframe, df in data.items():
            signal = self.analyze_timeframe(df, timeframe)
            signals[timeframe] = signal

            weight = self.timeframe_weights.get(timeframe, 1.0)
            weighted_scores.append(signal.overall_score * weight)
            total_weight += weight

        # Calculate combined score
        combined_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Calculate alignment (how well timeframes agree)
        scores = [s.overall_score for s in signals.values()]
        alignment = self._calculate_agreement(*scores)

        # Determine combined strength
        abs_score = abs(combined_score)
        if abs_score > 0.6 and alignment > 0.7:
            combined_strength = SignalStrength.VERY_STRONG
        elif abs_score > 0.4:
            combined_strength = SignalStrength.STRONG
        elif abs_score > 0.25:
            combined_strength = SignalStrength.MODERATE
        elif abs_score > 0.1:
            combined_strength = SignalStrength.WEAK
        else:
            combined_strength = SignalStrength.NEUTRAL

        # Primary direction
        if combined_score > 0.3:
            primary_direction = SignalType.BULLISH
        elif combined_score > 0:
            primary_direction = SignalType.WEAK_BULLISH
        elif combined_score > -0.3:
            primary_direction = SignalType.WEAK_BEARISH
        else:
            primary_direction = SignalType.BEARISH

        return MultiTimeframeSignal(
            timestamp=datetime.now(),
            signals=signals,
            combined_score=combined_score,
            combined_strength=combined_strength,
            primary_direction=primary_direction,
            alignment_score=alignment
        )

    def should_notify(self, signal: AggregatedSignal) -> bool:
        """Determine if a signal warrants notification"""
        if signal.strength == SignalStrength.NEUTRAL:
            return False

        if abs(signal.overall_score) >= self.config.min_signal_score:
            return True

        # Strong signals always notify
        if signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            return True

        return False

    def should_notify_mtf(self, signal: MultiTimeframeSignal) -> bool:
        """Determine if multi-timeframe signal warrants notification"""
        if signal.combined_strength == SignalStrength.NEUTRAL:
            return False

        # High alignment with moderate score
        if signal.alignment_score > 0.7 and abs(signal.combined_score) > 0.3:
            return True

        # Strong signal regardless of alignment
        if signal.combined_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            return True

        return abs(signal.combined_score) >= self.config.min_signal_score


def format_signal_report(signal: AggregatedSignal) -> str:
    """Format a signal into a readable report"""
    lines = [
        "=" * 60,
        f"SIGNAL DETECTED - {signal.timeframe.value}",
        "=" * 60,
        f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Price: ${signal.current_price:,.2f}",
        "",
        f"Overall: {signal.get_description()}",
        f"Confidence: {signal.confidence:.1%}",
        "",
        "Component Scores:",
        f"  Candlestick: {signal.candlestick_score:+.2f} - {signal.candlestick_summary}",
        f"  Chart Patterns: {signal.chart_pattern_score:+.2f} - {signal.chart_summary}",
        f"  Indicators: {signal.indicator_score:+.2f} - {signal.indicator_summary}",
    ]

    # Add detected patterns
    if signal.candlestick_patterns:
        lines.append("")
        lines.append("Candlestick Patterns:")
        for p in signal.candlestick_patterns[:5]:  # Top 5
            lines.append(f"  - {p.name}: {p.description}")

    if signal.chart_patterns:
        lines.append("")
        lines.append("Chart Patterns:")
        for p in signal.chart_patterns[:5]:
            lines.append(f"  - {p.name}: {p.description}")
            if p.target_price:
                lines.append(f"    Target: ${p.target_price:,.2f}")

    # Key indicator signals
    key_signals = [s for s in signal.indicator_signals if s.confidence >= 0.6]
    if key_signals:
        lines.append("")
        lines.append("Key Indicator Signals:")
        for s in key_signals[:5]:
            lines.append(f"  - {s.name}: {s.description}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_mtf_report(signal: MultiTimeframeSignal) -> str:
    """Format multi-timeframe signal into a readable report"""
    lines = [
        "=" * 70,
        "MULTI-TIMEFRAME ANALYSIS",
        "=" * 70,
        f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Combined Score: {signal.combined_score:+.2f}",
        f"Strength: {signal.combined_strength.value}",
        f"Direction: {signal.primary_direction.value}",
        f"Timeframe Alignment: {signal.alignment_score:.1%}",
        "",
        "Timeframe Breakdown:",
    ]

    for tf, sig in sorted(signal.signals.items(), key=lambda x: x[0].value):
        arrow = "▲" if sig.overall_score > 0 else "▼" if sig.overall_score < 0 else "●"
        lines.append(f"  {tf.value:>4}: {arrow} {sig.overall_score:+.2f} ({sig.strength.value})")

    lines.append("=" * 70)
    return "\n".join(lines)
