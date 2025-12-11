"""
Technical indicators module
Implements MACD, RSI, Bollinger Bands, Stochastic, and other indicators
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config import SignalType, IndicatorConfig, DEFAULT_INDICATOR_CONFIG


@dataclass
class IndicatorSignal:
    """Represents a signal from a technical indicator"""
    name: str
    signal_type: SignalType
    value: float
    description: str
    confidence: float


class TechnicalIndicators:
    """Calculate and analyze technical indicators"""

    def __init__(self, df: pd.DataFrame, config: IndicatorConfig = DEFAULT_INDICATOR_CONFIG):
        """
        Initialize with OHLCV DataFrame

        Expected columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self.config = config
        self._calculate_all_indicators()

    def _calculate_all_indicators(self) -> None:
        """Calculate all technical indicators"""
        self._calculate_moving_averages()
        self._calculate_macd()
        self._calculate_rsi()
        self._calculate_bollinger_bands()
        self._calculate_stochastic()
        self._calculate_atr()
        self._calculate_volume_indicators()
        self._calculate_momentum()

    def _calculate_moving_averages(self) -> None:
        """Calculate various moving averages"""
        cfg = self.config
        df = self.df

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=cfg.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=cfg.ema_slow, adjust=False).mean()

        # SMAs
        df['sma_50'] = df['close'].rolling(window=cfg.sma_50).mean()
        df['sma_200'] = df['close'].rolling(window=cfg.sma_200).mean()

        # EMA crossover
        df['ema_cross'] = df['ema_fast'] - df['ema_slow']
        df['ema_cross_prev'] = df['ema_cross'].shift(1)

    def _calculate_macd(self) -> None:
        """Calculate MACD indicator"""
        cfg = self.config
        df = self.df

        # MACD line
        ema_fast = df['close'].ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=cfg.macd_slow, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd_line'].ewm(span=cfg.macd_signal, adjust=False).mean()

        # Histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        df['macd_histogram_prev'] = df['macd_histogram'].shift(1)

    def _calculate_rsi(self) -> None:
        """Calculate RSI indicator"""
        cfg = self.config
        df = self.df

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=cfg.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=cfg.rsi_period).mean()

        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_prev'] = df['rsi'].shift(1)

    def _calculate_bollinger_bands(self) -> None:
        """Calculate Bollinger Bands"""
        cfg = self.config
        df = self.df

        df['bb_middle'] = df['close'].rolling(window=cfg.bb_period).mean()
        rolling_std = df['close'].rolling(window=cfg.bb_period).std()

        df['bb_upper'] = df['bb_middle'] + (rolling_std * cfg.bb_std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * cfg.bb_std)

        # BB width for volatility
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # %B indicator
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    def _calculate_stochastic(self) -> None:
        """Calculate Stochastic Oscillator"""
        cfg = self.config
        df = self.df

        low_min = df['low'].rolling(window=cfg.stoch_k).min()
        high_max = df['high'].rolling(window=cfg.stoch_k).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=cfg.stoch_d).mean()

        df['stoch_k_prev'] = df['stoch_k'].shift(1)
        df['stoch_d_prev'] = df['stoch_d'].shift(1)

    def _calculate_atr(self) -> None:
        """Calculate Average True Range for volatility"""
        cfg = self.config
        df = self.df

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=cfg.atr_period).mean()

        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close'] * 100

    def _calculate_volume_indicators(self) -> None:
        """Calculate volume-based indicators"""
        cfg = self.config
        df = self.df

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=cfg.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # On Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(window=20).mean()

        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).fillna(0).cumsum()

    def _calculate_momentum(self) -> None:
        """Calculate momentum indicators"""
        df = self.df

        # Rate of Change (ROC)
        df['roc_10'] = df['close'].pct_change(periods=10) * 100
        df['roc_20'] = df['close'].pct_change(periods=20) * 100

        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)

    def get_current_values(self) -> Dict[str, float]:
        """Get the most recent indicator values"""
        latest = self.df.iloc[-1]
        return {
            'price': latest['close'],
            'ema_fast': latest['ema_fast'],
            'ema_slow': latest['ema_slow'],
            'sma_50': latest['sma_50'],
            'sma_200': latest['sma_200'],
            'macd_line': latest['macd_line'],
            'macd_signal': latest['macd_signal'],
            'macd_histogram': latest['macd_histogram'],
            'rsi': latest['rsi'],
            'bb_upper': latest['bb_upper'],
            'bb_middle': latest['bb_middle'],
            'bb_lower': latest['bb_lower'],
            'bb_pct': latest['bb_pct'],
            'stoch_k': latest['stoch_k'],
            'stoch_d': latest['stoch_d'],
            'atr': latest['atr'],
            'atr_pct': latest['atr_pct'],
            'volume_ratio': latest['volume_ratio'],
        }

    def analyze_signals(self) -> List[IndicatorSignal]:
        """Analyze all indicators and generate trading signals"""
        signals = []

        signals.extend(self._analyze_macd())
        signals.extend(self._analyze_rsi())
        signals.extend(self._analyze_bollinger())
        signals.extend(self._analyze_moving_averages())
        signals.extend(self._analyze_stochastic())
        signals.extend(self._analyze_volume())
        signals.extend(self._analyze_trend())

        return signals

    def _analyze_macd(self) -> List[IndicatorSignal]:
        """Analyze MACD for signals"""
        signals = []
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        macd = latest['macd_line']
        signal = latest['macd_signal']
        histogram = latest['macd_histogram']
        histogram_prev = prev['macd_histogram']

        # MACD Crossover
        if latest['macd_line'] > latest['macd_signal'] and prev['macd_line'] <= prev['macd_signal']:
            signals.append(IndicatorSignal(
                name="MACD Bullish Crossover",
                signal_type=SignalType.BULLISH,
                value=macd,
                description="MACD line crossed above signal line",
                confidence=0.7
            ))
        elif latest['macd_line'] < latest['macd_signal'] and prev['macd_line'] >= prev['macd_signal']:
            signals.append(IndicatorSignal(
                name="MACD Bearish Crossover",
                signal_type=SignalType.BEARISH,
                value=macd,
                description="MACD line crossed below signal line",
                confidence=0.7
            ))

        # MACD Zero Line Cross
        if macd > 0 and prev['macd_line'] <= 0:
            signals.append(IndicatorSignal(
                name="MACD Above Zero",
                signal_type=SignalType.BULLISH,
                value=macd,
                description="MACD crossed above zero line",
                confidence=0.6
            ))
        elif macd < 0 and prev['macd_line'] >= 0:
            signals.append(IndicatorSignal(
                name="MACD Below Zero",
                signal_type=SignalType.BEARISH,
                value=macd,
                description="MACD crossed below zero line",
                confidence=0.6
            ))

        # Histogram momentum
        if histogram > 0 and histogram > histogram_prev:
            signals.append(IndicatorSignal(
                name="MACD Histogram Rising",
                signal_type=SignalType.WEAK_BULLISH,
                value=histogram,
                description="Bullish momentum increasing",
                confidence=0.5
            ))
        elif histogram < 0 and histogram < histogram_prev:
            signals.append(IndicatorSignal(
                name="MACD Histogram Falling",
                signal_type=SignalType.WEAK_BEARISH,
                value=histogram,
                description="Bearish momentum increasing",
                confidence=0.5
            ))

        # MACD Divergence
        divergence = self._detect_macd_divergence()
        if divergence:
            signals.append(divergence)

        return signals

    def _detect_macd_divergence(self) -> Optional[IndicatorSignal]:
        """Detect MACD divergence with price"""
        df = self.df.tail(30)
        if len(df) < 10:
            return None

        # Find recent price highs/lows and corresponding MACD
        price_highs = df['high'].rolling(5, center=True).max() == df['high']
        price_lows = df['low'].rolling(5, center=True).min() == df['low']

        # Bullish divergence: lower price lows with higher MACD lows
        low_indices = df.index[price_lows]
        if len(low_indices) >= 2:
            recent_lows = low_indices[-2:]
            if (df.loc[recent_lows[-1], 'low'] < df.loc[recent_lows[-2], 'low'] and
                df.loc[recent_lows[-1], 'macd_line'] > df.loc[recent_lows[-2], 'macd_line']):
                return IndicatorSignal(
                    name="MACD Bullish Divergence",
                    signal_type=SignalType.BULLISH,
                    value=df.iloc[-1]['macd_line'],
                    description="Price making lower lows while MACD making higher lows",
                    confidence=0.75
                )

        # Bearish divergence: higher price highs with lower MACD highs
        high_indices = df.index[price_highs]
        if len(high_indices) >= 2:
            recent_highs = high_indices[-2:]
            if (df.loc[recent_highs[-1], 'high'] > df.loc[recent_highs[-2], 'high'] and
                df.loc[recent_highs[-1], 'macd_line'] < df.loc[recent_highs[-2], 'macd_line']):
                return IndicatorSignal(
                    name="MACD Bearish Divergence",
                    signal_type=SignalType.BEARISH,
                    value=df.iloc[-1]['macd_line'],
                    description="Price making higher highs while MACD making lower highs",
                    confidence=0.75
                )

        return None

    def _analyze_rsi(self) -> List[IndicatorSignal]:
        """Analyze RSI for signals"""
        signals = []
        cfg = self.config
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        rsi = latest['rsi']

        # Overbought/Oversold
        if rsi > cfg.rsi_overbought:
            signals.append(IndicatorSignal(
                name="RSI Overbought",
                signal_type=SignalType.BEARISH,
                value=rsi,
                description=f"RSI at {rsi:.1f}, overbought territory",
                confidence=0.6
            ))
        elif rsi < cfg.rsi_oversold:
            signals.append(IndicatorSignal(
                name="RSI Oversold",
                signal_type=SignalType.BULLISH,
                value=rsi,
                description=f"RSI at {rsi:.1f}, oversold territory",
                confidence=0.6
            ))

        # RSI crossing 50
        if rsi > 50 and prev['rsi'] <= 50:
            signals.append(IndicatorSignal(
                name="RSI Above 50",
                signal_type=SignalType.WEAK_BULLISH,
                value=rsi,
                description="RSI crossed above midline",
                confidence=0.45
            ))
        elif rsi < 50 and prev['rsi'] >= 50:
            signals.append(IndicatorSignal(
                name="RSI Below 50",
                signal_type=SignalType.WEAK_BEARISH,
                value=rsi,
                description="RSI crossed below midline",
                confidence=0.45
            ))

        # RSI Divergence
        divergence = self._detect_rsi_divergence()
        if divergence:
            signals.append(divergence)

        return signals

    def _detect_rsi_divergence(self) -> Optional[IndicatorSignal]:
        """Detect RSI divergence with price"""
        df = self.df.tail(30)
        if len(df) < 10:
            return None

        price_highs = df['high'].rolling(5, center=True).max() == df['high']
        price_lows = df['low'].rolling(5, center=True).min() == df['low']

        # Bullish divergence
        low_indices = df.index[price_lows]
        if len(low_indices) >= 2:
            recent_lows = low_indices[-2:]
            if (df.loc[recent_lows[-1], 'low'] < df.loc[recent_lows[-2], 'low'] and
                df.loc[recent_lows[-1], 'rsi'] > df.loc[recent_lows[-2], 'rsi']):
                return IndicatorSignal(
                    name="RSI Bullish Divergence",
                    signal_type=SignalType.STRONG_BULLISH,
                    value=df.iloc[-1]['rsi'],
                    description="Price making lower lows while RSI making higher lows",
                    confidence=0.8
                )

        # Bearish divergence
        high_indices = df.index[price_highs]
        if len(high_indices) >= 2:
            recent_highs = high_indices[-2:]
            if (df.loc[recent_highs[-1], 'high'] > df.loc[recent_highs[-2], 'high'] and
                df.loc[recent_highs[-1], 'rsi'] < df.loc[recent_highs[-2], 'rsi']):
                return IndicatorSignal(
                    name="RSI Bearish Divergence",
                    signal_type=SignalType.STRONG_BEARISH,
                    value=df.iloc[-1]['rsi'],
                    description="Price making higher highs while RSI making lower highs",
                    confidence=0.8
                )

        return None

    def _analyze_bollinger(self) -> List[IndicatorSignal]:
        """Analyze Bollinger Bands for signals"""
        signals = []
        latest = self.df.iloc[-1]

        close = latest['close']
        upper = latest['bb_upper']
        lower = latest['bb_lower']
        middle = latest['bb_middle']
        bb_pct = latest['bb_pct']

        # Price at bands
        if close >= upper:
            signals.append(IndicatorSignal(
                name="BB Upper Band Touch",
                signal_type=SignalType.WEAK_BEARISH,
                value=bb_pct,
                description="Price at upper Bollinger Band",
                confidence=0.5
            ))
        elif close <= lower:
            signals.append(IndicatorSignal(
                name="BB Lower Band Touch",
                signal_type=SignalType.WEAK_BULLISH,
                value=bb_pct,
                description="Price at lower Bollinger Band",
                confidence=0.5
            ))

        # BB Squeeze (low volatility)
        bb_width = latest['bb_width']
        avg_width = self.df['bb_width'].rolling(50).mean().iloc[-1]

        if bb_width < avg_width * 0.5:
            signals.append(IndicatorSignal(
                name="BB Squeeze",
                signal_type=SignalType.NEUTRAL,
                value=bb_width,
                description="Low volatility, potential breakout coming",
                confidence=0.6
            ))

        return signals

    def _analyze_moving_averages(self) -> List[IndicatorSignal]:
        """Analyze moving average relationships"""
        signals = []
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        close = latest['close']

        # EMA Crossover
        if latest['ema_cross'] > 0 and prev['ema_cross'] <= 0:
            signals.append(IndicatorSignal(
                name="EMA Bullish Crossover",
                signal_type=SignalType.BULLISH,
                value=close,
                description=f"EMA {self.config.ema_fast} crossed above EMA {self.config.ema_slow}",
                confidence=0.65
            ))
        elif latest['ema_cross'] < 0 and prev['ema_cross'] >= 0:
            signals.append(IndicatorSignal(
                name="EMA Bearish Crossover",
                signal_type=SignalType.BEARISH,
                value=close,
                description=f"EMA {self.config.ema_fast} crossed below EMA {self.config.ema_slow}",
                confidence=0.65
            ))

        # Golden Cross / Death Cross
        if not pd.isna(latest['sma_50']) and not pd.isna(latest['sma_200']):
            if not pd.isna(prev['sma_50']) and not pd.isna(prev['sma_200']):
                if latest['sma_50'] > latest['sma_200'] and prev['sma_50'] <= prev['sma_200']:
                    signals.append(IndicatorSignal(
                        name="Golden Cross",
                        signal_type=SignalType.STRONG_BULLISH,
                        value=close,
                        description="SMA 50 crossed above SMA 200",
                        confidence=0.8
                    ))
                elif latest['sma_50'] < latest['sma_200'] and prev['sma_50'] >= prev['sma_200']:
                    signals.append(IndicatorSignal(
                        name="Death Cross",
                        signal_type=SignalType.STRONG_BEARISH,
                        value=close,
                        description="SMA 50 crossed below SMA 200",
                        confidence=0.8
                    ))

        # Price vs MAs
        if not pd.isna(latest['sma_200']):
            if close > latest['sma_200']:
                signals.append(IndicatorSignal(
                    name="Above SMA 200",
                    signal_type=SignalType.WEAK_BULLISH,
                    value=close,
                    description="Price above 200 SMA - bullish trend",
                    confidence=0.4
                ))
            else:
                signals.append(IndicatorSignal(
                    name="Below SMA 200",
                    signal_type=SignalType.WEAK_BEARISH,
                    value=close,
                    description="Price below 200 SMA - bearish trend",
                    confidence=0.4
                ))

        return signals

    def _analyze_stochastic(self) -> List[IndicatorSignal]:
        """Analyze Stochastic Oscillator"""
        signals = []
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        k = latest['stoch_k']
        d = latest['stoch_d']

        # Overbought/Oversold
        if k > 80 and d > 80:
            signals.append(IndicatorSignal(
                name="Stochastic Overbought",
                signal_type=SignalType.WEAK_BEARISH,
                value=k,
                description=f"Stochastic at {k:.1f}, overbought",
                confidence=0.5
            ))
        elif k < 20 and d < 20:
            signals.append(IndicatorSignal(
                name="Stochastic Oversold",
                signal_type=SignalType.WEAK_BULLISH,
                value=k,
                description=f"Stochastic at {k:.1f}, oversold",
                confidence=0.5
            ))

        # Stochastic Crossover
        if k > d and prev['stoch_k'] <= prev['stoch_d'] and k < 30:
            signals.append(IndicatorSignal(
                name="Stochastic Bullish Cross",
                signal_type=SignalType.BULLISH,
                value=k,
                description="Stochastic %K crossed above %D in oversold zone",
                confidence=0.65
            ))
        elif k < d and prev['stoch_k'] >= prev['stoch_d'] and k > 70:
            signals.append(IndicatorSignal(
                name="Stochastic Bearish Cross",
                signal_type=SignalType.BEARISH,
                value=k,
                description="Stochastic %K crossed below %D in overbought zone",
                confidence=0.65
            ))

        return signals

    def _analyze_volume(self) -> List[IndicatorSignal]:
        """Analyze volume indicators"""
        signals = []
        latest = self.df.iloc[-1]

        volume_ratio = latest['volume_ratio']

        # High volume
        if volume_ratio > 2.0:
            if latest['close'] > latest['open']:
                signals.append(IndicatorSignal(
                    name="High Volume Bullish",
                    signal_type=SignalType.BULLISH,
                    value=volume_ratio,
                    description=f"Volume {volume_ratio:.1f}x average on bullish candle",
                    confidence=0.6
                ))
            else:
                signals.append(IndicatorSignal(
                    name="High Volume Bearish",
                    signal_type=SignalType.BEARISH,
                    value=volume_ratio,
                    description=f"Volume {volume_ratio:.1f}x average on bearish candle",
                    confidence=0.6
                ))

        # OBV trend
        obv = latest['obv']
        obv_ma = latest['obv_ma']
        if not pd.isna(obv_ma):
            if obv > obv_ma:
                signals.append(IndicatorSignal(
                    name="OBV Bullish",
                    signal_type=SignalType.WEAK_BULLISH,
                    value=obv,
                    description="OBV above its moving average",
                    confidence=0.4
                ))
            else:
                signals.append(IndicatorSignal(
                    name="OBV Bearish",
                    signal_type=SignalType.WEAK_BEARISH,
                    value=obv,
                    description="OBV below its moving average",
                    confidence=0.4
                ))

        return signals

    def _analyze_trend(self) -> List[IndicatorSignal]:
        """Analyze overall trend strength"""
        signals = []
        latest = self.df.iloc[-1]

        # Momentum
        roc = latest['roc_10']
        if not pd.isna(roc):
            if roc > 5:
                signals.append(IndicatorSignal(
                    name="Strong Momentum Up",
                    signal_type=SignalType.BULLISH,
                    value=roc,
                    description=f"10-period ROC at {roc:.1f}%",
                    confidence=0.55
                ))
            elif roc < -5:
                signals.append(IndicatorSignal(
                    name="Strong Momentum Down",
                    signal_type=SignalType.BEARISH,
                    value=roc,
                    description=f"10-period ROC at {roc:.1f}%",
                    confidence=0.55
                ))

        return signals


def get_indicator_summary(signals: List[IndicatorSignal]) -> Tuple[float, str]:
    """
    Get an overall signal score and summary from indicator signals

    Returns:
        score: -1.0 (strong bearish) to +1.0 (strong bullish)
        summary: Text description
    """
    if not signals:
        return 0.0, "No indicator signals"

    score = 0.0
    for signal in signals:
        weight = signal.confidence

        if signal.signal_type == SignalType.STRONG_BULLISH:
            score += weight * 1.5
        elif signal.signal_type == SignalType.BULLISH:
            score += weight
        elif signal.signal_type == SignalType.WEAK_BULLISH:
            score += weight * 0.5
        elif signal.signal_type == SignalType.STRONG_BEARISH:
            score -= weight * 1.5
        elif signal.signal_type == SignalType.BEARISH:
            score -= weight
        elif signal.signal_type == SignalType.WEAK_BEARISH:
            score -= weight * 0.5

    # Normalize
    max_possible = sum(s.confidence * 1.5 for s in signals)
    if max_possible > 0:
        score = score / max_possible

    score = max(-1.0, min(1.0, score))

    if score > 0.5:
        summary = "Strong bullish indicators"
    elif score > 0.2:
        summary = "Moderate bullish indicators"
    elif score > 0:
        summary = "Slight bullish bias"
    elif score > -0.2:
        summary = "Slight bearish bias"
    elif score > -0.5:
        summary = "Moderate bearish indicators"
    else:
        summary = "Strong bearish indicators"

    return score, summary
