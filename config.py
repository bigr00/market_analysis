"""
Configuration settings for Bitcoin Market Analysis Tool
"""
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class TimeFrame(Enum):
    """Supported timeframes for analysis"""
    M1 = "1m"      # 1 minute
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # 1 day


class SignalType(Enum):
    """Types of trading signals"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    WEAK_BEARISH = "WEAK_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # Symbol to track
    symbol: str = "BTCUSDT"

    # Timeframes to analyze (multi-timeframe analysis)
    timeframes: List[TimeFrame] = field(default_factory=lambda: [
        TimeFrame.M15,
        TimeFrame.H1,
        TimeFrame.H4
    ])

    # Number of candles to fetch for analysis
    candle_limit: int = 500

    # Polling interval in seconds (for REST API fallback)
    poll_interval: int = 60

    # Signal thresholds
    min_signal_score: float = 0.6  # Minimum score to trigger notification
    strong_signal_threshold: float = 0.8  # Score for strong signals

    # Risk management
    default_stop_loss_pct: float = 2.0  # 2% stop loss
    default_take_profit_pct: float = 4.0  # 4% take profit (2:1 R:R)
    risk_reward_ratio: float = 2.0

    # Audio settings
    enable_audio: bool = True
    bullish_sound: str = "sounds/bullish.wav"
    bearish_sound: str = "sounds/bearish.wav"


@dataclass
class IndicatorConfig:
    """Technical indicator parameters"""
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # RSI settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Bollinger Bands settings
    bb_period: int = 20
    bb_std: float = 2.0

    # Moving averages
    ema_fast: int = 9
    ema_slow: int = 21
    sma_50: int = 50
    sma_200: int = 200

    # ATR for volatility
    atr_period: int = 14

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # Volume
    volume_ma_period: int = 20


# Default configurations
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_INDICATOR_CONFIG = IndicatorConfig()
