# Bitcoin Market Analysis Tool

A comprehensive real-time cryptocurrency market analysis tool that detects trading signals using candlestick patterns, chart patterns, and technical indicators. Features both a command-line interface and a modern web dashboard with interactive TradingView-style charts.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface-recommended)
  - [CLI Mode](#cli-mode)
  - [Command Line Options](#command-line-options)
- [How It Works](#how-it-works)
  - [Signal Scoring System](#signal-scoring-system)
  - [Multi-Timeframe Analysis](#multi-timeframe-analysis)
  - [Trade Suggestions](#trade-suggestions)
- [Pattern Detection](#pattern-detection)
  - [Candlestick Patterns](#candlestick-patterns)
  - [Chart Patterns](#chart-patterns)
  - [Technical Indicators](#technical-indicators)
- [Web Interface Guide](#web-interface-guide)
- [Configuration](#configuration)
- [Project Architecture](#project-architecture)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Features

### Data & Analysis
- **Real-Time Data**: Fetches live OHLCV data from Binance API (no API key required for public data)
- **Multi-Timeframe Analysis**: Simultaneously analyzes multiple timeframes (5m, 15m, 1h, 4h, 1d)
- **Configurable Polling**: Adjustable update intervals from seconds to minutes
- **Any Trading Pair**: Supports any Binance trading pair (BTCUSDT, ETHUSDT, etc.)

### Pattern Recognition
- **15+ Candlestick Patterns**: Doji, Hammer, Engulfing, Morning/Evening Star, Three Soldiers, and more
- **6+ Chart Patterns**: Head & Shoulders, Double Top/Bottom, Triangles, Support/Resistance
- **Divergence Detection**: Identifies bullish/bearish divergences in MACD and RSI

### Technical Indicators
- **Trend Indicators**: EMA (9/21), SMA (50/200), MACD
- **Momentum Indicators**: RSI, Stochastic Oscillator, Rate of Change
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume Moving Average

### Signal Generation
- **Weighted Scoring**: Combines all signals into a -1.0 to +1.0 score
- **Confidence Levels**: Each signal includes a confidence percentage
- **Timeframe Alignment**: Measures agreement across multiple timeframes
- **Trade Suggestions**: Calculates entry, stop loss, and take profit levels

### User Interface
- **Web Dashboard**: Modern, responsive interface with TradingView-style charts
- **CLI Mode**: Terminal-based output with rich formatting
- **Audio Alerts**: Browser-based sound notifications for trading signals
- **Desktop Notifications**: Native macOS notifications (CLI mode)

---

## Screenshots

### Web Dashboard
```
┌─────────────────────────────────────────────────────────────────────────┐
│  BTC Market Analysis          BTC Price $97,500.00        🔊 [Live]    │
├─────────────────────────────────────────────────────────────────────────┤
│  [5m] [15m] [1H] [4H] [1D]                                              │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────┐  │
│  │                                     │  │  Signal Score           │  │
│  │     📈 Candlestick Chart            │  │  ████████░░  +0.65      │  │
│  │        with EMA & BB overlays       │  │  STRONG BULLISH         │  │
│  │        and pattern markers          │  │                         │  │
│  │                                     │  │  Components:            │  │
│  │                                     │  │  Candlestick: +0.45     │  │
│  │                                     │  │  Chart:       +0.70     │  │
│  └─────────────────────────────────────┘  │  Indicators:  +0.55     │  │
│  ┌─────────────────────────────────────┐  ├─────────────────────────┤  │
│  │  [MACD] [RSI] [Volume]              │  │  Trade Suggestion       │  │
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                    │  │  📈 LONG                │  │
│  └─────────────────────────────────────┘  │  Entry:  $97,500        │  │
│                                           │  SL:     $95,500        │  │
│                                           │  TP1:    $100,500       │  │
│                                           │  TP2:    $103,500       │  │
│                                           │  R:R     1:1.5          │  │
│                                           └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone or navigate to the project
cd market_analysis

# Create virtual environment
python3 -m venv .marketanalysisenv
source .marketanalysisenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate audio files for notifications
python generate_sounds.py

# Start the web interface
python main.py --web

# Open http://localhost:5000 in your browser
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for Binance API)

### Step-by-Step Installation

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/market_analysis
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .marketanalysisenv
   ```

3. **Activate the virtual environment**:
   ```bash
   # macOS/Linux
   source .marketanalysisenv/bin/activate

   # Windows
   .marketanalysisenv\Scripts\activate
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate notification sounds**:
   ```bash
   python generate_sounds.py
   ```

6. **Verify installation**:
   ```bash
   python main.py --help
   ```

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical computations |
| ta | Technical analysis library |
| python-binance | Binance API client |
| flask | Web server framework |
| flask-socketio | Real-time WebSocket support |
| eventlet | Async networking library |
| scipy | Scientific computing (pattern detection) |
| rich | Beautiful terminal output |

---

## Usage

### Web Interface (Recommended)

The web interface provides a visual dashboard with interactive charts.

```bash
# Start with default settings (port 5000, 60-second updates)
python main.py --web

# Custom port
python main.py --web --port 8080

# Faster updates (30 seconds)
python main.py --web --interval 30

# Analyze Ethereum instead of Bitcoin
python main.py --web --symbol ETHUSDT

# Lower sensitivity (only strong signals)
python main.py --web --min-score 0.6
```

Then open **http://localhost:5000** in your browser.

### CLI Mode

For terminal-based monitoring without a browser.

```bash
# Basic usage
python main.py

# With custom settings
python main.py --interval 30 --timeframes 5m,15m,1h

# Disable audio notifications
python main.py --no-audio

# Test audio notifications
python main.py --test-audio
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--web` | `-w` | False | Run web interface instead of CLI |
| `--port` | `-p` | 5000 | Port for web server |
| `--interval` | `-i` | 60 | Polling interval in seconds |
| `--timeframes` | `-t` | 15m,1h,4h | Comma-separated timeframes to analyze |
| `--symbol` | `-s` | BTCUSDT | Trading pair to analyze |
| `--min-score` | | 0.4 | Minimum score to trigger alerts |
| `--no-audio` | | False | Disable audio notifications |
| `--test-audio` | | False | Test audio and exit |

### Examples

```bash
# Scalping setup: fast updates, short timeframes
python main.py --web -i 15 -t 1m,5m,15m

# Swing trading setup: slower updates, longer timeframes
python main.py --web -i 300 -t 1h,4h,1d

# High sensitivity (more alerts)
python main.py --web --min-score 0.3

# Low sensitivity (only strong signals)
python main.py --web --min-score 0.7

# Monitor multiple pairs (run in separate terminals)
python main.py --web --port 5000 --symbol BTCUSDT
python main.py --web --port 5001 --symbol ETHUSDT
```

---

## How It Works

### Signal Scoring System

The tool combines signals from three categories into a weighted final score:

```
Final Score = (Candlestick × 0.25) + (Chart Pattern × 0.35) + (Indicators × 0.40)
```

| Score Range | Classification | Action |
|-------------|---------------|--------|
| +0.7 to +1.0 | Very Strong Bullish | Strong buy signal |
| +0.4 to +0.7 | Strong Bullish | Buy signal |
| +0.2 to +0.4 | Moderate Bullish | Weak buy signal |
| +0.0 to +0.2 | Weak Bullish | Slight bullish bias |
| -0.2 to +0.0 | Weak Bearish | Slight bearish bias |
| -0.4 to -0.2 | Moderate Bearish | Weak sell signal |
| -0.7 to -0.4 | Strong Bearish | Sell signal |
| -1.0 to -0.7 | Very Strong Bearish | Strong sell signal |

### Multi-Timeframe Analysis

The tool analyzes multiple timeframes simultaneously and weights them:

| Timeframe | Weight | Purpose |
|-----------|--------|---------|
| 1m, 5m | 0.5-0.6 | Entry timing |
| 15m, 30m | 0.7-0.8 | Short-term trend |
| 1h | 0.9 | Medium-term trend |
| 4h | 1.0 | Primary trend |
| 1d | 1.1 | Major trend |

**Timeframe Alignment Score**: Measures how well signals agree across timeframes (0-100%). Higher alignment = higher confidence.

### Trade Suggestions

When a signal is detected, the tool calculates:

1. **Entry Price**: Current market price
2. **Stop Loss**: Based on ATR (Average True Range) and nearby support/resistance
3. **Take Profit 1**: 1.5× risk (partial exit point)
4. **Take Profit 2**: 2-3× risk or next resistance level
5. **Take Profit 3**: 4× risk (for strong signals only)

**Risk/Reward Calculation**:
```
Risk = Entry - Stop Loss
Reward = Take Profit - Entry
R:R Ratio = Reward / Risk
```

---

## Pattern Detection

### Candlestick Patterns

#### Single Candle Patterns
| Pattern | Signal | Description |
|---------|--------|-------------|
| Doji | Neutral | Indecision, potential reversal |
| Dragonfly Doji | Bullish | Long lower wick, reversal at support |
| Gravestone Doji | Bearish | Long upper wick, reversal at resistance |
| Hammer | Bullish | Small body, long lower wick after downtrend |
| Inverted Hammer | Bullish | Small body, long upper wick after downtrend |
| Hanging Man | Bearish | Hammer shape after uptrend |
| Shooting Star | Bearish | Inverted hammer after uptrend |
| Marubozu | Continuation | Full-body candle, strong momentum |
| Spinning Top | Neutral | Small body, equal wicks, indecision |

#### Two Candle Patterns
| Pattern | Signal | Description |
|---------|--------|-------------|
| Bullish Engulfing | Strong Bullish | Green candle engulfs previous red |
| Bearish Engulfing | Strong Bearish | Red candle engulfs previous green |
| Bullish Harami | Bullish | Small green inside large red |
| Bearish Harami | Bearish | Small red inside large green |
| Piercing Line | Bullish | Gap down, closes above midpoint |
| Dark Cloud Cover | Bearish | Gap up, closes below midpoint |
| Tweezer Bottom | Bullish | Equal lows, reversal signal |
| Tweezer Top | Bearish | Equal highs, reversal signal |

#### Three Candle Patterns
| Pattern | Signal | Description |
|---------|--------|-------------|
| Morning Star | Strong Bullish | Red, small, green - reversal |
| Evening Star | Strong Bearish | Green, small, red - reversal |
| Three White Soldiers | Strong Bullish | Three consecutive green candles |
| Three Black Crows | Strong Bearish | Three consecutive red candles |
| Three Inside Up | Bullish | Harami + confirmation |
| Three Inside Down | Bearish | Harami + confirmation |

### Chart Patterns

| Pattern | Signal | Target Calculation |
|---------|--------|-------------------|
| Double Top | Bearish | Neckline - Pattern Height |
| Double Bottom | Bullish | Neckline + Pattern Height |
| Head & Shoulders | Strong Bearish | Neckline - (Head - Neckline) |
| Inverse H&S | Strong Bullish | Neckline + (Neckline - Head) |
| Ascending Triangle | Bullish | Breakout + Triangle Height |
| Descending Triangle | Bearish | Breakdown - Triangle Height |
| Support Level | Bullish | Bounce zone |
| Resistance Level | Bearish | Rejection zone |

### Technical Indicators

#### MACD (Moving Average Convergence Divergence)
- **Settings**: 12, 26, 9
- **Signals**:
  - Bullish crossover (MACD crosses above signal)
  - Bearish crossover (MACD crosses below signal)
  - Zero line cross
  - Histogram momentum
  - Divergence with price

#### RSI (Relative Strength Index)
- **Settings**: 14 periods
- **Signals**:
  - Overbought (>70): Bearish
  - Oversold (<30): Bullish
  - Centerline cross (50)
  - Divergence with price

#### Bollinger Bands
- **Settings**: 20 periods, 2 standard deviations
- **Signals**:
  - Upper band touch: Potential reversal down
  - Lower band touch: Potential reversal up
  - Squeeze: Low volatility, breakout expected

#### Moving Averages
- **EMA 9/21**: Short-term crossovers
- **SMA 50/200**: Golden Cross (bullish) / Death Cross (bearish)

#### Stochastic Oscillator
- **Settings**: 14, 3, 3
- **Signals**:
  - Overbought (>80) + crossover: Bearish
  - Oversold (<20) + crossover: Bullish

---

## Web Interface Guide

### Header Bar
- **Logo & Title**: App identification
- **Price Display**: Current BTC price with change percentage
- **Connection Status**: WebSocket connection indicator (Live/Connecting/Disconnected)
- **Sound Toggle**: Enable/disable audio notifications

### Main Chart Area
- **Timeframe Buttons**: Switch between 5m, 15m, 1H, 4H, 1D
- **Candlestick Chart**: OHLC data with:
  - EMA 9 (orange line)
  - EMA 21 (blue line)
  - Bollinger Bands (purple dashed lines)
  - Pattern markers (arrows and circles)

### Indicator Panel
- **Tab Selection**: MACD, RSI, Volume
- **MACD**: Blue line, orange signal, colored histogram
- **RSI**: Purple line with 30/70 levels
- **Volume**: Bar chart colored by candle direction

### Signal Dashboard (Right Panel)

#### Signal Score Card
- **Gauge**: Visual representation of -1.0 to +1.0 score
- **Score Value**: Numerical score with color coding
- **Strength Label**: WEAK, MODERATE, STRONG, VERY_STRONG
- **Component Bars**: Individual scores for candlestick, chart, indicators

#### Trade Suggestion Card
- **Direction**: LONG (green) or SHORT (red)
- **Entry Price**: Suggested entry point
- **Stop Loss**: Risk management level
- **Take Profit 1/2**: Target levels
- **R:R Ratio**: Risk/Reward ratio
- **Confidence**: Percentage confidence in signal

#### Patterns Card
- **Pattern List**: Detected patterns with:
  - Colored dot (green=bullish, red=bearish)
  - Pattern name
  - Pattern type (candle, chart, indicator)

#### Timeframe Alignment Card
- **Alignment Bar**: Visual percentage of agreement
- **Alignment Value**: Percentage number
- **TF Signals**: Individual timeframe directions

#### Signal History Card
- **Recent Signals**: Last 10 signals with:
  - Timestamp
  - Direction (LONG/SHORT)
  - Price at signal

### Alert Modal
- Appears automatically on strong signals
- Shows direction, price, and score
- Auto-dismisses after 10 seconds
- Click outside or "Dismiss" to close

---

## Configuration

### Trading Configuration (config.py)

```python
@dataclass
class TradingConfig:
    symbol: str = "BTCUSDT"           # Trading pair
    timeframes: List[TimeFrame]       # Timeframes to analyze
    candle_limit: int = 500           # Historical candles to fetch
    poll_interval: int = 60           # Seconds between updates
    min_signal_score: float = 0.6     # Minimum score for alerts
    strong_signal_threshold: float = 0.8
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0
    risk_reward_ratio: float = 2.0
    enable_audio: bool = True
```

### Indicator Configuration (config.py)

```python
@dataclass
class IndicatorConfig:
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Moving Averages
    ema_fast: int = 9
    ema_slow: int = 21
    sma_50: int = 50
    sma_200: int = 200

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
```

---

## Project Architecture

```
market_analysis/
├── main.py                    # Application entry point
│                              # - Parses command line arguments
│                              # - Initializes CLI or Web mode
│                              # - Manages main analysis loop (CLI)
│
├── config.py                  # Configuration classes
│                              # - TradingConfig: trading parameters
│                              # - IndicatorConfig: indicator settings
│                              # - TimeFrame enum: supported intervals
│                              # - SignalType enum: signal classifications
│
├── data_fetcher.py            # Binance API integration
│                              # - BinanceDataFetcher: REST API client
│                              # - DataBuffer: rolling data storage
│                              # - WebSocket support for streaming
│
├── candlestick_patterns.py    # Candlestick pattern detection
│                              # - CandlestickPatternDetector class
│                              # - 15+ pattern recognition methods
│                              # - Pattern scoring and classification
│
├── chart_patterns.py          # Chart pattern detection
│                              # - ChartPatternDetector class
│                              # - Pivot point detection
│                              # - Support/resistance identification
│                              # - Pattern target calculation
│
├── technical_indicators.py    # Technical indicator calculations
│                              # - TechnicalIndicators class
│                              # - MACD, RSI, BB, Stochastic, etc.
│                              # - Divergence detection
│                              # - Signal generation
│
├── signal_aggregator.py       # Signal combination and scoring
│                              # - SignalAggregator class
│                              # - Weighted score calculation
│                              # - Multi-timeframe analysis
│                              # - Signal report formatting
│
├── trade_suggester.py         # Trade level calculations
│                              # - TradeSuggester class
│                              # - Entry/exit price calculation
│                              # - Risk/reward analysis
│                              # - Position sizing suggestions
│
├── notifier.py                # Notification system
│                              # - AudioNotifier: sound alerts
│                              # - DesktopNotifier: macOS notifications
│
├── web_server.py              # Flask web application
│                              # - WebAnalyzer: background analysis
│                              # - REST API endpoints
│                              # - WebSocket event handlers
│
├── generate_sounds.py         # Audio file generator
│                              # - Creates WAV notification sounds
│
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
├── README.md                 # This documentation
│
├── templates/
│   └── index.html            # Web interface HTML
│                             # - Dashboard layout
│                             # - Chart containers
│                             # - Signal panels
│
└── static/
    ├── css/
    │   └── style.css         # Web interface styles
    │                         # - Dark theme
    │                         # - Responsive layout
    │                         # - Component styling
    │
    ├── js/
    │   └── app.js            # Frontend JavaScript
    │                         # - Chart initialization
    │                         # - WebSocket handling
    │                         # - UI updates
    │                         # - Audio playback
    │
    └── sounds/
        ├── bullish.wav       # Ascending tone (buy signal)
        └── bearish.wav       # Descending tone (sell signal)
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Binance    │────▶│ DataFetcher  │────▶│   DataBuffer      │
│    API      │     │              │     │ (Rolling Storage) │
└─────────────┘     └──────────────┘     └─────────┬─────────┘
                                                   │
                    ┌──────────────────────────────┴──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
        ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
        │   Candlestick     │        │     Chart         │        │    Technical      │
        │    Patterns       │        │    Patterns       │        │   Indicators      │
        └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
                  │                            │                            │
                  └──────────────┬─────────────┴────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │   SignalAggregator    │
                    │  (Weighted Scoring)   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    TradeSuggester     │
                    │  (Entry/Exit Levels)  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
    ┌───────────────────┐               ┌───────────────────┐
    │   Web Interface   │               │   CLI Interface   │
    │   (WebSocket)     │               │   (Rich Console)  │
    └───────────────────┘               └───────────────────┘
```

---

## API Reference

### REST Endpoints

#### GET /api/chart/{timeframe}
Returns OHLCV data and indicators for charting.

**Parameters:**
- `timeframe`: One of `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`

**Response:**
```json
{
  "candles": [
    {"time": 1699900800, "open": 97000, "high": 97500, "low": 96800, "close": 97300, "volume": 1234.5}
  ],
  "patterns": [
    {"time": 1699900800, "position": "belowBar", "color": "#22c55e", "shape": "arrowUp", "text": "Hammer"}
  ],
  "indicators": {
    "ema_fast": [{"time": 1699900800, "value": 97100}],
    "macd_line": [{"time": 1699900800, "value": 150.5}],
    "rsi": [{"time": 1699900800, "value": 65.3}]
  }
}
```

#### GET /api/analysis
Returns current signal analysis.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "price": 97500.00,
  "signal": {
    "combined_score": 0.65,
    "strength": "STRONG",
    "direction": "BULLISH",
    "alignment": 0.85,
    "timeframes": {
      "15m": {"score": 0.55, "patterns": ["Hammer", "RSI Oversold"]},
      "1h": {"score": 0.70, "patterns": ["Bullish Engulfing"]},
      "4h": {"score": 0.60, "patterns": ["MACD Bullish Crossover"]}
    }
  },
  "suggestion": {
    "direction": "LONG",
    "entry": 97500,
    "stop_loss": 95500,
    "take_profit_1": 100500,
    "take_profit_2": 103500,
    "risk_reward": 1.5,
    "confidence": 0.75
  }
}
```

#### GET /api/price
Returns current price.

**Response:**
```json
{
  "price": 97500.00
}
```

### WebSocket Events

#### Client → Server
- `request_update`: Request latest analysis data

#### Server → Client
- `analysis_update`: Pushed on each analysis cycle
  ```json
  {
    "timestamp": "...",
    "price": 97500,
    "signal": {...},
    "suggestion": {...},
    "should_notify": true,
    "signal_history": [...]
  }
  ```

---

## Troubleshooting

### Installation Issues

#### "No module named 'eventlet'"
```bash
pip install eventlet
```

#### pip install fails with Python 3.13
Some packages may not support Python 3.13 yet. Try Python 3.11:
```bash
python3.11 -m venv .marketanalysisenv
```

### Connection Issues

#### "Could not connect to Binance API"
1. Check internet connection
2. Verify Binance is accessible in your region
3. Some countries require VPN to access Binance API

#### WebSocket disconnects frequently
- Check for firewall blocking WebSocket connections
- Try a different port: `python main.py --web --port 8080`
- Check browser console for errors

### Audio Issues

#### No sound in browser
1. Click anywhere on the page first (browser autoplay policy)
2. Check the sound toggle is enabled (speaker icon)
3. Verify sound files exist: `ls static/sounds/`
4. Regenerate sounds: `python generate_sounds.py`
5. Check browser console for audio errors

#### CLI audio not working
- macOS uses `afplay` command - verify it exists
- Falls back to terminal bell if unavailable

### Analysis Issues

#### No patterns detected
- Normal during ranging/quiet markets
- Try lower timeframes for more signals
- Reduce `--min-score` threshold

#### Signals don't match expected behavior
- Technical analysis is probabilistic, not deterministic
- Signals are suggestions, not guarantees
- Always use proper risk management

### Performance Issues

#### High CPU usage
- Increase polling interval: `--interval 120`
- Reduce number of timeframes: `-t 1h,4h`

#### Slow chart loading
- Reduce candle limit in config.py
- Use longer timeframes

---

## Contributing

Contributions are welcome! Here are some ways to contribute:

1. **Bug Reports**: Open an issue with reproduction steps
2. **Feature Requests**: Describe the feature and use case
3. **Code Contributions**: Submit a pull request
4. **Documentation**: Improve or translate docs

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd market_analysis

# Create development environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run in development mode
python main.py --web --interval 30
```

---

## Disclaimer

**IMPORTANT: This tool is for educational and informational purposes only.**

- This software does NOT constitute financial advice
- Past performance does NOT guarantee future results
- Cryptocurrency trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- Always do your own research (DYOR)
- The developers are not responsible for any financial losses

**Use this tool at your own risk.**

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
