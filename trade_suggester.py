"""
Trade suggestion module
Calculates entry, stop loss, and take profit levels based on signals
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config import TradingConfig, SignalType, DEFAULT_TRADING_CONFIG
from signal_aggregator import AggregatedSignal, MultiTimeframeSignal, SignalStrength


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeSuggestion:
    """Represents a trade suggestion with entry and exit levels"""
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float  # First target (partial exit)
    take_profit_2: float  # Second target (full exit)
    take_profit_3: Optional[float] = None  # Extended target

    # Risk metrics
    risk_amount: float = 0.0  # Dollar risk per unit
    reward_amount: float = 0.0  # Dollar reward to TP1
    risk_reward_ratio: float = 0.0

    # Position sizing
    suggested_position_pct: float = 0.0  # % of portfolio
    confidence: float = 0.0

    # Context
    signal_score: float = 0.0
    signal_strength: str = ""
    reasoning: str = ""

    def format_levels(self) -> str:
        """Format the trade levels as a string"""
        direction_emoji = "🟢" if self.direction == TradeDirection.LONG else "🔴"
        lines = [
            f"{direction_emoji} {self.direction.value} Trade Suggestion",
            f"",
            f"Entry:      ${self.entry_price:,.2f}",
            f"Stop Loss:  ${self.stop_loss:,.2f} ({self._pct_from_entry(self.stop_loss):+.2f}%)",
            f"",
            f"Targets:",
            f"  TP1:      ${self.take_profit_1:,.2f} ({self._pct_from_entry(self.take_profit_1):+.2f}%)",
            f"  TP2:      ${self.take_profit_2:,.2f} ({self._pct_from_entry(self.take_profit_2):+.2f}%)",
        ]
        if self.take_profit_3:
            lines.append(f"  TP3:      ${self.take_profit_3:,.2f} ({self._pct_from_entry(self.take_profit_3):+.2f}%)")

        lines.extend([
            f"",
            f"Risk/Reward: 1:{self.risk_reward_ratio:.1f}",
            f"Confidence:  {self.confidence:.0%}",
            f"",
            f"Reasoning: {self.reasoning}"
        ])

        return "\n".join(lines)

    def _pct_from_entry(self, price: float) -> float:
        """Calculate percentage from entry price"""
        return ((price - self.entry_price) / self.entry_price) * 100


class TradeSuggester:
    """Generates trade suggestions based on signals and market conditions"""

    def __init__(self, config: TradingConfig = DEFAULT_TRADING_CONFIG):
        self.config = config

    def suggest_trade(self, signal: AggregatedSignal, df: pd.DataFrame) -> Optional[TradeSuggestion]:
        """
        Generate trade suggestion based on aggregated signal

        Args:
            signal: The aggregated signal from analysis
            df: OHLCV DataFrame for the analyzed timeframe

        Returns:
            TradeSuggestion if a trade is warranted, None otherwise
        """
        # Only suggest trades for significant signals
        if signal.strength == SignalStrength.NEUTRAL:
            return None

        if abs(signal.overall_score) < self.config.min_signal_score:
            return None

        current_price = signal.current_price

        # Determine direction
        direction = TradeDirection.LONG if signal.overall_score > 0 else TradeDirection.SHORT

        # Calculate ATR for dynamic stop loss
        atr = self._calculate_atr(df)
        atr_multiplier = self._get_atr_multiplier(signal.strength)

        # Find support/resistance levels
        support, resistance = self._find_sr_levels(df)

        # Calculate levels
        if direction == TradeDirection.LONG:
            entry_price = current_price
            stop_loss = self._calculate_long_stop(current_price, atr, atr_multiplier, support)
            tp1, tp2, tp3 = self._calculate_long_targets(current_price, stop_loss, resistance, signal)
        else:
            entry_price = current_price
            stop_loss = self._calculate_short_stop(current_price, atr, atr_multiplier, resistance)
            tp1, tp2, tp3 = self._calculate_short_targets(current_price, stop_loss, support, signal)

        # Calculate risk metrics
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(tp1 - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        # Position sizing suggestion
        position_pct = self._suggest_position_size(signal, rr_ratio)

        # Generate reasoning
        reasoning = self._generate_reasoning(signal, direction)

        return TradeSuggestion(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=rr_ratio,
            suggested_position_pct=position_pct,
            confidence=signal.confidence,
            signal_score=signal.overall_score,
            signal_strength=signal.strength.value,
            reasoning=reasoning
        )

    def suggest_from_mtf(self, mtf_signal: MultiTimeframeSignal,
                         primary_df: pd.DataFrame) -> Optional[TradeSuggestion]:
        """
        Generate trade suggestion from multi-timeframe signal

        Args:
            mtf_signal: Multi-timeframe signal analysis
            primary_df: DataFrame for the primary (lowest) timeframe

        Returns:
            TradeSuggestion if warranted
        """
        # Get the primary timeframe signal
        primary_tf = min(mtf_signal.signals.keys(), key=lambda x: x.value)
        primary_signal = mtf_signal.signals[primary_tf]

        # Boost confidence if timeframes align
        adjusted_signal = primary_signal
        if mtf_signal.alignment_score > 0.7:
            # Create a copy with boosted confidence
            adjusted_signal = AggregatedSignal(
                timestamp=primary_signal.timestamp,
                signal_type=primary_signal.signal_type,
                strength=primary_signal.strength,
                overall_score=mtf_signal.combined_score,  # Use combined score
                confidence=min(1.0, primary_signal.confidence + mtf_signal.alignment_score * 0.2),
                candlestick_score=primary_signal.candlestick_score,
                chart_pattern_score=primary_signal.chart_pattern_score,
                indicator_score=primary_signal.indicator_score,
                candlestick_patterns=primary_signal.candlestick_patterns,
                chart_patterns=primary_signal.chart_patterns,
                indicator_signals=primary_signal.indicator_signals,
                current_price=primary_signal.current_price,
                timeframe=primary_signal.timeframe,
                candlestick_summary=primary_signal.candlestick_summary,
                chart_summary=primary_signal.chart_summary,
                indicator_summary=primary_signal.indicator_summary
            )

        return self.suggest_trade(adjusted_signal, primary_df)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not pd.isna(atr) else (high.iloc[-1] - low.iloc[-1])

    def _get_atr_multiplier(self, strength: SignalStrength) -> float:
        """Get ATR multiplier based on signal strength"""
        multipliers = {
            SignalStrength.VERY_STRONG: 1.5,
            SignalStrength.STRONG: 2.0,
            SignalStrength.MODERATE: 2.5,
            SignalStrength.WEAK: 3.0,
            SignalStrength.NEUTRAL: 3.5
        }
        return multipliers.get(strength, 2.5)

    def _find_sr_levels(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        """Find nearest support and resistance levels"""
        recent = df.tail(lookback)
        current_price = df['close'].iloc[-1]

        # Simple pivot-based S/R
        highs = recent['high'].values
        lows = recent['low'].values

        # Find levels above and below current price
        resistance_candidates = [h for h in highs if h > current_price]
        support_candidates = [l for l in lows if l < current_price]

        resistance = min(resistance_candidates) if resistance_candidates else current_price * 1.02
        support = max(support_candidates) if support_candidates else current_price * 0.98

        return support, resistance

    def _calculate_long_stop(self, price: float, atr: float, multiplier: float,
                             support: float) -> float:
        """Calculate stop loss for long trade"""
        atr_stop = price - (atr * multiplier)
        # Use the higher of ATR-based stop or just below support
        support_stop = support * 0.995

        return max(atr_stop, support_stop)

    def _calculate_short_stop(self, price: float, atr: float, multiplier: float,
                              resistance: float) -> float:
        """Calculate stop loss for short trade"""
        atr_stop = price + (atr * multiplier)
        # Use the lower of ATR-based stop or just above resistance
        resistance_stop = resistance * 1.005

        return min(atr_stop, resistance_stop)

    def _calculate_long_targets(self, entry: float, stop: float, resistance: float,
                                signal: AggregatedSignal) -> Tuple[float, float, Optional[float]]:
        """Calculate take profit targets for long trade"""
        risk = entry - stop
        rr_ratio = self.config.risk_reward_ratio

        # TP1: 1.5x risk (partial exit)
        tp1 = entry + (risk * 1.5)

        # TP2: 2-3x risk or resistance
        tp2 = min(entry + (risk * rr_ratio), resistance) if resistance > tp1 else entry + (risk * rr_ratio)

        # TP3: Extended target for strong signals
        tp3 = None
        if signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            tp3 = entry + (risk * 4)

        # Check if chart patterns suggest specific targets
        for pattern in signal.chart_patterns:
            if pattern.target_price and pattern.target_price > entry:
                # Use pattern target if it's reasonable
                if pattern.target_price > tp1:
                    tp2 = max(tp2, pattern.target_price)

        return tp1, tp2, tp3

    def _calculate_short_targets(self, entry: float, stop: float, support: float,
                                 signal: AggregatedSignal) -> Tuple[float, float, Optional[float]]:
        """Calculate take profit targets for short trade"""
        risk = stop - entry
        rr_ratio = self.config.risk_reward_ratio

        # TP1: 1.5x risk
        tp1 = entry - (risk * 1.5)

        # TP2: 2-3x risk or support
        tp2 = max(entry - (risk * rr_ratio), support) if support < tp1 else entry - (risk * rr_ratio)

        # TP3: Extended target for strong signals
        tp3 = None
        if signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            tp3 = entry - (risk * 4)

        # Check chart pattern targets
        for pattern in signal.chart_patterns:
            if pattern.target_price and pattern.target_price < entry:
                if pattern.target_price < tp1:
                    tp2 = min(tp2, pattern.target_price)

        return tp1, tp2, tp3

    def _suggest_position_size(self, signal: AggregatedSignal, rr_ratio: float) -> float:
        """Suggest position size as percentage of portfolio"""
        base_size = 2.0  # Base 2% position

        # Adjust by signal strength
        strength_multipliers = {
            SignalStrength.VERY_STRONG: 1.5,
            SignalStrength.STRONG: 1.25,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.5,
            SignalStrength.NEUTRAL: 0.0
        }

        multiplier = strength_multipliers.get(signal.strength, 1.0)

        # Adjust by confidence
        multiplier *= signal.confidence

        # Adjust by risk/reward
        if rr_ratio >= 3:
            multiplier *= 1.2
        elif rr_ratio < 1.5:
            multiplier *= 0.7

        position_pct = base_size * multiplier

        # Cap at 5%
        return min(position_pct, 5.0)

    def _generate_reasoning(self, signal: AggregatedSignal, direction: TradeDirection) -> str:
        """Generate human-readable reasoning for the trade"""
        reasons = []

        # Direction
        dir_word = "bullish" if direction == TradeDirection.LONG else "bearish"
        reasons.append(f"Overall {dir_word} bias (score: {signal.overall_score:+.2f})")

        # Key patterns
        if signal.candlestick_patterns:
            top_pattern = signal.candlestick_patterns[0]
            reasons.append(f"{top_pattern.name} pattern detected")

        if signal.chart_patterns:
            top_chart = signal.chart_patterns[0]
            reasons.append(f"{top_chart.name}")

        # Key indicators
        key_indicators = [s for s in signal.indicator_signals if s.confidence >= 0.6]
        if key_indicators:
            top_indicator = key_indicators[0]
            reasons.append(f"{top_indicator.name}")

        return "; ".join(reasons[:3])


def format_trade_alert(suggestion: TradeSuggestion, price: float) -> str:
    """Format a trade suggestion as an alert message"""
    direction = "LONG 📈" if suggestion.direction == TradeDirection.LONG else "SHORT 📉"
    sl_pct = abs((suggestion.stop_loss - suggestion.entry_price) / suggestion.entry_price) * 100
    tp_pct = abs((suggestion.take_profit_1 - suggestion.entry_price) / suggestion.entry_price) * 100

    lines = [
        "╔══════════════════════════════════════════════════╗",
        f"║  🚨 TRADE SIGNAL: {direction:^28} ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Current Price:  ${price:>15,.2f}           ║",
        f"║  Entry:          ${suggestion.entry_price:>15,.2f}           ║",
        f"║  Stop Loss:      ${suggestion.stop_loss:>15,.2f} ({sl_pct:>4.1f}%)   ║",
        f"║  Take Profit 1:  ${suggestion.take_profit_1:>15,.2f} ({tp_pct:>4.1f}%)   ║",
        f"║  Take Profit 2:  ${suggestion.take_profit_2:>15,.2f}           ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Risk/Reward:    1:{suggestion.risk_reward_ratio:<5.1f}                       ║",
        f"║  Confidence:     {suggestion.confidence:>5.0%}                         ║",
        f"║  Signal Score:   {suggestion.signal_score:>+5.2f}                        ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  {suggestion.reasoning[:46]:<46} ║",
        "╚══════════════════════════════════════════════════╝",
    ]

    return "\n".join(lines)
