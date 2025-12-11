"""
Candlestick pattern detection module
Detects common bullish and bearish candlestick patterns
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config import SignalType


class PatternStrength(Enum):
    """Pattern strength classification"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class CandlestickPattern:
    """Represents a detected candlestick pattern"""
    name: str
    signal_type: SignalType
    strength: PatternStrength
    description: str
    confidence: float  # 0.0 to 1.0
    index: int  # Index in the dataframe where pattern was detected


class CandlestickPatternDetector:
    """Detects various candlestick patterns"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame

        Expected columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self._calculate_candle_properties()

    def _calculate_candle_properties(self) -> None:
        """Pre-calculate candle properties for pattern detection"""
        df = self.df

        # Basic properties
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        # Relative sizes (as percentage of range)
        df['body_pct'] = df['body_abs'] / df['total_range'].replace(0, np.nan)
        df['upper_wick_pct'] = df['upper_wick'] / df['total_range'].replace(0, np.nan)
        df['lower_wick_pct'] = df['lower_wick'] / df['total_range'].replace(0, np.nan)

        # Direction
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']

        # Average body size for relative comparisons
        df['avg_body'] = df['body_abs'].rolling(window=20).mean()
        df['avg_range'] = df['total_range'].rolling(window=20).mean()

    def detect_all_patterns(self, lookback: int = 5) -> List[CandlestickPattern]:
        """
        Detect all patterns in the most recent candles

        Args:
            lookback: Number of recent candles to analyze

        Returns:
            List of detected patterns
        """
        patterns = []

        # Single candle patterns
        patterns.extend(self._detect_doji(lookback))
        patterns.extend(self._detect_hammer(lookback))
        patterns.extend(self._detect_inverted_hammer(lookback))
        patterns.extend(self._detect_hanging_man(lookback))
        patterns.extend(self._detect_shooting_star(lookback))
        patterns.extend(self._detect_marubozu(lookback))
        patterns.extend(self._detect_spinning_top(lookback))

        # Two candle patterns
        patterns.extend(self._detect_engulfing(lookback))
        patterns.extend(self._detect_harami(lookback))
        patterns.extend(self._detect_piercing_dark_cloud(lookback))
        patterns.extend(self._detect_tweezer(lookback))

        # Three candle patterns
        patterns.extend(self._detect_morning_evening_star(lookback))
        patterns.extend(self._detect_three_soldiers_crows(lookback))
        patterns.extend(self._detect_three_inside(lookback))

        return patterns

    def _detect_doji(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Doji patterns (indecision)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']):
                continue

            # Doji: very small body relative to range
            if row['body_pct'] < 0.1 and row['total_range'] > 0:
                # Determine doji type
                if row['upper_wick_pct'] > 0.4 and row['lower_wick_pct'] > 0.4:
                    name = "Long-legged Doji"
                    strength = PatternStrength.MODERATE
                elif row['upper_wick_pct'] < 0.1:
                    name = "Dragonfly Doji"  # Bullish
                    strength = PatternStrength.STRONG
                elif row['lower_wick_pct'] < 0.1:
                    name = "Gravestone Doji"  # Bearish
                    strength = PatternStrength.STRONG
                else:
                    name = "Doji"
                    strength = PatternStrength.WEAK

                signal = SignalType.NEUTRAL
                if name == "Dragonfly Doji":
                    signal = SignalType.BULLISH
                elif name == "Gravestone Doji":
                    signal = SignalType.BEARISH

                patterns.append(CandlestickPattern(
                    name=name,
                    signal_type=signal,
                    strength=strength,
                    description="Indicates market indecision, potential reversal",
                    confidence=0.6 if strength == PatternStrength.STRONG else 0.4,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_hammer(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Hammer patterns (bullish reversal)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']) or pd.isna(row['avg_body']):
                continue

            # Hammer: small body at top, long lower wick, little/no upper wick
            if (row['body_pct'] > 0.1 and row['body_pct'] < 0.4 and
                row['lower_wick_pct'] > 0.5 and
                row['upper_wick_pct'] < 0.15):

                # Check if in downtrend (prior candles bearish)
                prior_idx = self.df.index.get_loc(idx) - 3
                if prior_idx >= 0:
                    prior_trend = self.df.iloc[prior_idx:self.df.index.get_loc(idx)]['close'].diff().mean()
                    if prior_trend < 0:  # In downtrend
                        patterns.append(CandlestickPattern(
                            name="Hammer",
                            signal_type=SignalType.BULLISH,
                            strength=PatternStrength.STRONG,
                            description="Bullish reversal signal after downtrend",
                            confidence=0.7,
                            index=self.df.index.get_loc(idx)
                        ))

        return patterns

    def _detect_inverted_hammer(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Inverted Hammer patterns (potential bullish reversal)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']):
                continue

            # Inverted Hammer: small body at bottom, long upper wick, little/no lower wick
            if (row['body_pct'] > 0.1 and row['body_pct'] < 0.4 and
                row['upper_wick_pct'] > 0.5 and
                row['lower_wick_pct'] < 0.15):

                prior_idx = self.df.index.get_loc(idx) - 3
                if prior_idx >= 0:
                    prior_trend = self.df.iloc[prior_idx:self.df.index.get_loc(idx)]['close'].diff().mean()
                    if prior_trend < 0:
                        patterns.append(CandlestickPattern(
                            name="Inverted Hammer",
                            signal_type=SignalType.WEAK_BULLISH,
                            strength=PatternStrength.MODERATE,
                            description="Potential bullish reversal, needs confirmation",
                            confidence=0.5,
                            index=self.df.index.get_loc(idx)
                        ))

        return patterns

    def _detect_hanging_man(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Hanging Man patterns (bearish reversal)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']):
                continue

            # Same shape as hammer but in uptrend
            if (row['body_pct'] > 0.1 and row['body_pct'] < 0.4 and
                row['lower_wick_pct'] > 0.5 and
                row['upper_wick_pct'] < 0.15):

                prior_idx = self.df.index.get_loc(idx) - 3
                if prior_idx >= 0:
                    prior_trend = self.df.iloc[prior_idx:self.df.index.get_loc(idx)]['close'].diff().mean()
                    if prior_trend > 0:  # In uptrend
                        patterns.append(CandlestickPattern(
                            name="Hanging Man",
                            signal_type=SignalType.BEARISH,
                            strength=PatternStrength.MODERATE,
                            description="Bearish reversal signal after uptrend",
                            confidence=0.6,
                            index=self.df.index.get_loc(idx)
                        ))

        return patterns

    def _detect_shooting_star(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Shooting Star patterns (bearish reversal)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']):
                continue

            # Shooting Star: small body at bottom, long upper wick after uptrend
            if (row['body_pct'] > 0.1 and row['body_pct'] < 0.4 and
                row['upper_wick_pct'] > 0.5 and
                row['lower_wick_pct'] < 0.15):

                prior_idx = self.df.index.get_loc(idx) - 3
                if prior_idx >= 0:
                    prior_trend = self.df.iloc[prior_idx:self.df.index.get_loc(idx)]['close'].diff().mean()
                    if prior_trend > 0:
                        patterns.append(CandlestickPattern(
                            name="Shooting Star",
                            signal_type=SignalType.BEARISH,
                            strength=PatternStrength.STRONG,
                            description="Strong bearish reversal after uptrend",
                            confidence=0.7,
                            index=self.df.index.get_loc(idx)
                        ))

        return patterns

    def _detect_marubozu(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Marubozu patterns (strong trend continuation)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']) or pd.isna(row['avg_body']):
                continue

            # Marubozu: large body with minimal wicks
            if (row['body_pct'] > 0.85 and
                row['body_abs'] > row['avg_body'] * 1.5):

                if row['is_bullish']:
                    patterns.append(CandlestickPattern(
                        name="Bullish Marubozu",
                        signal_type=SignalType.STRONG_BULLISH,
                        strength=PatternStrength.STRONG,
                        description="Strong buying pressure, trend continuation",
                        confidence=0.75,
                        index=self.df.index.get_loc(idx)
                    ))
                else:
                    patterns.append(CandlestickPattern(
                        name="Bearish Marubozu",
                        signal_type=SignalType.STRONG_BEARISH,
                        strength=PatternStrength.STRONG,
                        description="Strong selling pressure, trend continuation",
                        confidence=0.75,
                        index=self.df.index.get_loc(idx)
                    ))

        return patterns

    def _detect_spinning_top(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Spinning Top patterns (indecision)"""
        patterns = []
        df = self.df.tail(lookback)

        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row['body_pct']):
                continue

            # Spinning Top: small body with roughly equal wicks
            if (row['body_pct'] > 0.1 and row['body_pct'] < 0.35 and
                row['upper_wick_pct'] > 0.25 and row['lower_wick_pct'] > 0.25 and
                abs(row['upper_wick_pct'] - row['lower_wick_pct']) < 0.2):

                patterns.append(CandlestickPattern(
                    name="Spinning Top",
                    signal_type=SignalType.NEUTRAL,
                    strength=PatternStrength.WEAK,
                    description="Market indecision, watch for breakout",
                    confidence=0.4,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_engulfing(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Engulfing patterns"""
        patterns = []
        df = self.df.tail(lookback + 1)

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            if pd.isna(prev['body_abs']) or pd.isna(curr['body_abs']):
                continue

            idx = df.index[i]

            # Bullish Engulfing
            if (prev['is_bearish'] and curr['is_bullish'] and
                curr['open'] < prev['close'] and curr['close'] > prev['open'] and
                curr['body_abs'] > prev['body_abs']):

                patterns.append(CandlestickPattern(
                    name="Bullish Engulfing",
                    signal_type=SignalType.STRONG_BULLISH,
                    strength=PatternStrength.STRONG,
                    description="Strong bullish reversal pattern",
                    confidence=0.75,
                    index=self.df.index.get_loc(idx)
                ))

            # Bearish Engulfing
            elif (prev['is_bullish'] and curr['is_bearish'] and
                  curr['open'] > prev['close'] and curr['close'] < prev['open'] and
                  curr['body_abs'] > prev['body_abs']):

                patterns.append(CandlestickPattern(
                    name="Bearish Engulfing",
                    signal_type=SignalType.STRONG_BEARISH,
                    strength=PatternStrength.STRONG,
                    description="Strong bearish reversal pattern",
                    confidence=0.75,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_harami(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Harami patterns"""
        patterns = []
        df = self.df.tail(lookback + 1)

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            if pd.isna(prev['body_abs']) or pd.isna(curr['body_abs']):
                continue

            idx = df.index[i]

            # Current candle body is inside previous candle body
            curr_high = max(curr['open'], curr['close'])
            curr_low = min(curr['open'], curr['close'])
            prev_high = max(prev['open'], prev['close'])
            prev_low = min(prev['open'], prev['close'])

            if curr_high < prev_high and curr_low > prev_low:
                # Bullish Harami (after bearish candle)
                if prev['is_bearish'] and curr['is_bullish']:
                    patterns.append(CandlestickPattern(
                        name="Bullish Harami",
                        signal_type=SignalType.BULLISH,
                        strength=PatternStrength.MODERATE,
                        description="Potential bullish reversal",
                        confidence=0.55,
                        index=self.df.index.get_loc(idx)
                    ))

                # Bearish Harami (after bullish candle)
                elif prev['is_bullish'] and curr['is_bearish']:
                    patterns.append(CandlestickPattern(
                        name="Bearish Harami",
                        signal_type=SignalType.BEARISH,
                        strength=PatternStrength.MODERATE,
                        description="Potential bearish reversal",
                        confidence=0.55,
                        index=self.df.index.get_loc(idx)
                    ))

        return patterns

    def _detect_piercing_dark_cloud(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Piercing Line and Dark Cloud Cover patterns"""
        patterns = []
        df = self.df.tail(lookback + 1)

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            if pd.isna(prev['body_abs']) or pd.isna(curr['body_abs']):
                continue

            idx = df.index[i]
            prev_midpoint = (prev['open'] + prev['close']) / 2

            # Piercing Line (bullish)
            if (prev['is_bearish'] and curr['is_bullish'] and
                curr['open'] < prev['low'] and
                curr['close'] > prev_midpoint and curr['close'] < prev['open']):

                patterns.append(CandlestickPattern(
                    name="Piercing Line",
                    signal_type=SignalType.BULLISH,
                    strength=PatternStrength.MODERATE,
                    description="Bullish reversal pattern",
                    confidence=0.6,
                    index=self.df.index.get_loc(idx)
                ))

            # Dark Cloud Cover (bearish)
            elif (prev['is_bullish'] and curr['is_bearish'] and
                  curr['open'] > prev['high'] and
                  curr['close'] < prev_midpoint and curr['close'] > prev['open']):

                patterns.append(CandlestickPattern(
                    name="Dark Cloud Cover",
                    signal_type=SignalType.BEARISH,
                    strength=PatternStrength.MODERATE,
                    description="Bearish reversal pattern",
                    confidence=0.6,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_tweezer(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Tweezer Top and Bottom patterns"""
        patterns = []
        df = self.df.tail(lookback + 1)
        tolerance = 0.001  # 0.1% tolerance for matching highs/lows

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            idx = df.index[i]

            # Tweezer Top (bearish)
            if (prev['is_bullish'] and curr['is_bearish'] and
                abs(prev['high'] - curr['high']) / prev['high'] < tolerance):

                patterns.append(CandlestickPattern(
                    name="Tweezer Top",
                    signal_type=SignalType.BEARISH,
                    strength=PatternStrength.MODERATE,
                    description="Bearish reversal at resistance",
                    confidence=0.55,
                    index=self.df.index.get_loc(idx)
                ))

            # Tweezer Bottom (bullish)
            elif (prev['is_bearish'] and curr['is_bullish'] and
                  abs(prev['low'] - curr['low']) / prev['low'] < tolerance):

                patterns.append(CandlestickPattern(
                    name="Tweezer Bottom",
                    signal_type=SignalType.BULLISH,
                    strength=PatternStrength.MODERATE,
                    description="Bullish reversal at support",
                    confidence=0.55,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_morning_evening_star(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Morning Star and Evening Star patterns"""
        patterns = []
        df = self.df.tail(lookback + 2)

        for i in range(2, len(df)):
            first = df.iloc[i - 2]
            second = df.iloc[i - 1]
            third = df.iloc[i]

            if pd.isna(first['body_abs']) or pd.isna(second['body_abs']) or pd.isna(third['body_abs']):
                continue

            idx = df.index[i]

            # Morning Star (bullish)
            if (first['is_bearish'] and first['body_pct'] > 0.5 and
                second['body_pct'] < 0.3 and  # Small middle candle
                third['is_bullish'] and third['body_pct'] > 0.5 and
                third['close'] > (first['open'] + first['close']) / 2):

                patterns.append(CandlestickPattern(
                    name="Morning Star",
                    signal_type=SignalType.STRONG_BULLISH,
                    strength=PatternStrength.STRONG,
                    description="Strong bullish reversal pattern",
                    confidence=0.8,
                    index=self.df.index.get_loc(idx)
                ))

            # Evening Star (bearish)
            elif (first['is_bullish'] and first['body_pct'] > 0.5 and
                  second['body_pct'] < 0.3 and
                  third['is_bearish'] and third['body_pct'] > 0.5 and
                  third['close'] < (first['open'] + first['close']) / 2):

                patterns.append(CandlestickPattern(
                    name="Evening Star",
                    signal_type=SignalType.STRONG_BEARISH,
                    strength=PatternStrength.STRONG,
                    description="Strong bearish reversal pattern",
                    confidence=0.8,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_three_soldiers_crows(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Three White Soldiers and Three Black Crows"""
        patterns = []
        df = self.df.tail(lookback + 2)

        for i in range(2, len(df)):
            first = df.iloc[i - 2]
            second = df.iloc[i - 1]
            third = df.iloc[i]

            if pd.isna(first['body_pct']) or pd.isna(second['body_pct']) or pd.isna(third['body_pct']):
                continue

            idx = df.index[i]

            # Three White Soldiers (strong bullish)
            if (first['is_bullish'] and second['is_bullish'] and third['is_bullish'] and
                first['body_pct'] > 0.5 and second['body_pct'] > 0.5 and third['body_pct'] > 0.5 and
                second['close'] > first['close'] and third['close'] > second['close'] and
                second['open'] > first['open'] and third['open'] > second['open']):

                patterns.append(CandlestickPattern(
                    name="Three White Soldiers",
                    signal_type=SignalType.STRONG_BULLISH,
                    strength=PatternStrength.STRONG,
                    description="Very strong bullish continuation",
                    confidence=0.85,
                    index=self.df.index.get_loc(idx)
                ))

            # Three Black Crows (strong bearish)
            elif (first['is_bearish'] and second['is_bearish'] and third['is_bearish'] and
                  first['body_pct'] > 0.5 and second['body_pct'] > 0.5 and third['body_pct'] > 0.5 and
                  second['close'] < first['close'] and third['close'] < second['close'] and
                  second['open'] < first['open'] and third['open'] < second['open']):

                patterns.append(CandlestickPattern(
                    name="Three Black Crows",
                    signal_type=SignalType.STRONG_BEARISH,
                    strength=PatternStrength.STRONG,
                    description="Very strong bearish continuation",
                    confidence=0.85,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns

    def _detect_three_inside(self, lookback: int) -> List[CandlestickPattern]:
        """Detect Three Inside Up/Down patterns"""
        patterns = []
        df = self.df.tail(lookback + 2)

        for i in range(2, len(df)):
            first = df.iloc[i - 2]
            second = df.iloc[i - 1]
            third = df.iloc[i]

            idx = df.index[i]

            # Check if second candle is inside first (harami)
            second_high = max(second['open'], second['close'])
            second_low = min(second['open'], second['close'])
            first_high = max(first['open'], first['close'])
            first_low = min(first['open'], first['close'])

            is_harami = second_high < first_high and second_low > first_low

            if not is_harami:
                continue

            # Three Inside Up (bullish)
            if (first['is_bearish'] and second['is_bullish'] and third['is_bullish'] and
                third['close'] > first['open']):

                patterns.append(CandlestickPattern(
                    name="Three Inside Up",
                    signal_type=SignalType.STRONG_BULLISH,
                    strength=PatternStrength.STRONG,
                    description="Confirmed bullish reversal",
                    confidence=0.75,
                    index=self.df.index.get_loc(idx)
                ))

            # Three Inside Down (bearish)
            elif (first['is_bullish'] and second['is_bearish'] and third['is_bearish'] and
                  third['close'] < first['open']):

                patterns.append(CandlestickPattern(
                    name="Three Inside Down",
                    signal_type=SignalType.STRONG_BEARISH,
                    strength=PatternStrength.STRONG,
                    description="Confirmed bearish reversal",
                    confidence=0.75,
                    index=self.df.index.get_loc(idx)
                ))

        return patterns


def get_pattern_summary(patterns: List[CandlestickPattern]) -> Tuple[float, str]:
    """
    Get an overall signal score and summary from detected patterns

    Returns:
        score: -1.0 (strong bearish) to +1.0 (strong bullish)
        summary: Text description
    """
    if not patterns:
        return 0.0, "No patterns detected"

    score = 0.0
    for pattern in patterns:
        weight = pattern.confidence * pattern.strength.value

        if pattern.signal_type in [SignalType.STRONG_BULLISH, SignalType.BULLISH, SignalType.WEAK_BULLISH]:
            score += weight
        elif pattern.signal_type in [SignalType.STRONG_BEARISH, SignalType.BEARISH, SignalType.WEAK_BEARISH]:
            score -= weight

    # Normalize score to -1 to 1 range
    max_possible = sum(p.confidence * p.strength.value for p in patterns)
    if max_possible > 0:
        score = score / max_possible

    if score > 0.5:
        summary = "Strong bullish candlestick signals"
    elif score > 0.2:
        summary = "Moderate bullish candlestick signals"
    elif score > 0:
        summary = "Weak bullish candlestick signals"
    elif score > -0.2:
        summary = "Weak bearish candlestick signals"
    elif score > -0.5:
        summary = "Moderate bearish candlestick signals"
    else:
        summary = "Strong bearish candlestick signals"

    return score, summary
