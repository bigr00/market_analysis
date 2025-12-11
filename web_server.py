"""
Web server for Bitcoin Market Analysis Tool
Provides real-time charts and pattern visualization
"""
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from config import TimeFrame, TradingConfig, SignalType
from data_fetcher import BinanceDataFetcher, DataBuffer
from signal_aggregator import SignalAggregator, SignalStrength
from trade_suggester import TradeSuggester, TradeDirection
from candlestick_patterns import CandlestickPatternDetector
from chart_patterns import ChartPatternDetector
from technical_indicators import TechnicalIndicators

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'btc-analysis-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')


class WebAnalyzer:
    """Web-based market analyzer with real-time updates"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.fetcher = BinanceDataFetcher(config=config)
        self.buffer = DataBuffer()
        self.aggregator = SignalAggregator(config)
        self.suggester = TradeSuggester(config)

        self.running = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.last_analysis = {}
        self.signal_history = []
        # Track the currently selected timeframe (defaults to first configured)
        self.current_timeframe = config.timeframes[0] if config.timeframes else TimeFrame.H1

    def start(self):
        """Start the analysis loop in a background thread"""
        if self.running:
            return

        self.running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("[INFO] Analysis thread started")

    def stop(self):
        """Stop the analysis loop"""
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        self.fetcher.close()

    def set_timeframe(self, timeframe_str: str):
        """Set the current timeframe to fetch data for"""
        tf_map = {
            '1m': TimeFrame.M1, '5m': TimeFrame.M5, '15m': TimeFrame.M15,
            '30m': TimeFrame.M30, '1h': TimeFrame.H1, '4h': TimeFrame.H4, '1d': TimeFrame.D1
        }
        if timeframe_str in tf_map:
            self.current_timeframe = tf_map[timeframe_str]
            print(f"[INFO] Timeframe set to {timeframe_str}")

    def _analysis_loop(self):
        """Background analysis loop"""
        # Initial data fetch
        try:
            self._fetch_and_analyze()
        except Exception as e:
            print(f"[ERROR] Initial fetch failed: {e}")

        while self.running:
            try:
                time.sleep(self.config.poll_interval)
                if self.running:
                    self._fetch_and_analyze()
            except Exception as e:
                print(f"[ERROR] Analysis error: {e}")
                time.sleep(5)

    def _fetch_and_analyze(self):
        """Fetch data and perform analysis"""
        # Only fetch data for the currently selected timeframe to reduce API usage
        df = self.fetcher.get_klines(self.current_timeframe)
        self.buffer.update(self.current_timeframe, df)
        print(f"[INFO] Fetched {len(df)} candles for {self.current_timeframe.value}")

        # Perform analysis on the current timeframe only
        current_df = self.buffer.get(self.current_timeframe)
        if current_df is None:
            return

        buffer_data = {self.current_timeframe: current_df}

        # Single-timeframe analysis (passed as dict for compatibility)
        mtf_signal = self.aggregator.analyze_multi_timeframe(buffer_data)

        # Use current timeframe for trade suggestion
        primary_df = current_df

        suggestion = None
        if primary_df is not None:
            suggestion = self.suggester.suggest_from_mtf(mtf_signal, primary_df)

        # Store analysis results
        self.last_analysis = {
            'timestamp': datetime.now().isoformat(),
            'mtf_signal': mtf_signal,
            'suggestion': suggestion,
            'price': self.fetcher.get_current_price()
        }

        # Check if signal warrants notification
        should_notify = self.aggregator.should_notify_mtf(mtf_signal)

        # Emit update via WebSocket
        self._emit_update(should_notify)

    def _emit_update(self, should_notify: bool):
        """Emit analysis update via WebSocket"""
        data = self.get_analysis_data()
        data['should_notify'] = should_notify

        if should_notify and self.last_analysis.get('suggestion'):
            # Add to signal history
            self.signal_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': self.last_analysis['mtf_signal'].combined_score,
                'direction': 'LONG' if self.last_analysis['suggestion'].direction == TradeDirection.LONG else 'SHORT',
                'price': self.last_analysis['price']
            })
            # Keep last 20
            self.signal_history = self.signal_history[-20:]

        socketio.emit('analysis_update', data)

    def get_chart_data(self, timeframe: str) -> Dict:
        """Get OHLCV data formatted for charts"""
        tf_map = {
            '1m': TimeFrame.M1, '5m': TimeFrame.M5, '15m': TimeFrame.M15,
            '30m': TimeFrame.M30, '1h': TimeFrame.H1, '4h': TimeFrame.H4, '1d': TimeFrame.D1
        }

        tf = tf_map.get(timeframe, TimeFrame.H1)
        df = self.buffer.get(tf)

        if df is None or len(df) == 0:
            return {'candles': [], 'patterns': [], 'indicators': {}}

        # Format candles for lightweight-charts
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                'time': int(idx.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        # Detect patterns for visualization
        patterns = self._get_pattern_markers(df)

        # Get indicator data
        indicators = self._get_indicator_data(df)

        return {
            'candles': candles,
            'patterns': patterns,
            'indicators': indicators
        }

    def _get_pattern_markers(self, df) -> List[Dict]:
        """Get pattern markers for chart annotation"""
        markers = []

        # Candlestick patterns
        try:
            detector = CandlestickPatternDetector(df)
            patterns = detector.detect_all_patterns(lookback=20)

            for pattern in patterns:
                if pattern.index < len(df):
                    timestamp = int(df.index[pattern.index].timestamp())
                    is_bullish = pattern.signal_type in [
                        SignalType.STRONG_BULLISH, SignalType.BULLISH, SignalType.WEAK_BULLISH
                    ]
                    markers.append({
                        'time': timestamp,
                        'position': 'belowBar' if is_bullish else 'aboveBar',
                        'color': '#22c55e' if is_bullish else '#ef4444',
                        'shape': 'arrowUp' if is_bullish else 'arrowDown',
                        'text': pattern.name,
                        'type': 'candlestick'
                    })
        except Exception as e:
            print(f"[WARN] Pattern detection error: {e}")

        # Chart patterns
        try:
            chart_detector = ChartPatternDetector(df)
            chart_patterns = chart_detector.detect_all_patterns(lookback=100)

            for pattern in chart_patterns:
                if pattern.end_index < len(df):
                    timestamp = int(df.index[pattern.end_index].timestamp())
                    is_bullish = pattern.signal_type in [
                        SignalType.STRONG_BULLISH, SignalType.BULLISH, SignalType.WEAK_BULLISH
                    ]
                    markers.append({
                        'time': timestamp,
                        'position': 'belowBar' if is_bullish else 'aboveBar',
                        'color': '#3b82f6' if is_bullish else '#f97316',
                        'shape': 'circle',
                        'text': pattern.name,
                        'type': 'chart_pattern',
                        'target': pattern.target_price,
                        'neckline': pattern.neckline
                    })
        except Exception as e:
            print(f"[WARN] Chart pattern detection error: {e}")

        return markers

    def _get_indicator_data(self, df) -> Dict:
        """Get indicator values for chart overlays"""
        try:
            indicators = TechnicalIndicators(df)
            ind_df = indicators.df

            # Format for charts
            result = {
                'ema_fast': [],
                'ema_slow': [],
                'sma_50': [],
                'sma_200': [],
                'bb_upper': [],
                'bb_middle': [],
                'bb_lower': [],
                'macd_line': [],
                'macd_signal': [],
                'macd_histogram': [],
                'rsi': [],
                'volume': []
            }

            for idx, row in ind_df.iterrows():
                timestamp = int(idx.timestamp())

                # Moving averages
                if not pd.isna(row.get('ema_fast')):
                    result['ema_fast'].append({'time': timestamp, 'value': float(row['ema_fast'])})
                if not pd.isna(row.get('ema_slow')):
                    result['ema_slow'].append({'time': timestamp, 'value': float(row['ema_slow'])})
                if not pd.isna(row.get('sma_50')):
                    result['sma_50'].append({'time': timestamp, 'value': float(row['sma_50'])})
                if not pd.isna(row.get('sma_200')):
                    result['sma_200'].append({'time': timestamp, 'value': float(row['sma_200'])})

                # Bollinger Bands
                if not pd.isna(row.get('bb_upper')):
                    result['bb_upper'].append({'time': timestamp, 'value': float(row['bb_upper'])})
                if not pd.isna(row.get('bb_middle')):
                    result['bb_middle'].append({'time': timestamp, 'value': float(row['bb_middle'])})
                if not pd.isna(row.get('bb_lower')):
                    result['bb_lower'].append({'time': timestamp, 'value': float(row['bb_lower'])})

                # MACD
                if not pd.isna(row.get('macd_line')):
                    result['macd_line'].append({'time': timestamp, 'value': float(row['macd_line'])})
                if not pd.isna(row.get('macd_signal')):
                    result['macd_signal'].append({'time': timestamp, 'value': float(row['macd_signal'])})
                if not pd.isna(row.get('macd_histogram')):
                    result['macd_histogram'].append({
                        'time': timestamp,
                        'value': float(row['macd_histogram']),
                        'color': '#22c55e' if row['macd_histogram'] > 0 else '#ef4444'
                    })

                # RSI
                if not pd.isna(row.get('rsi')):
                    result['rsi'].append({'time': timestamp, 'value': float(row['rsi'])})

                # Volume
                result['volume'].append({
                    'time': timestamp,
                    'value': float(row['volume']),
                    'color': '#22c55e88' if row['close'] > row['open'] else '#ef444488'
                })

            return result

        except Exception as e:
            print(f"[WARN] Indicator calculation error: {e}")
            return {}

    def get_analysis_data(self) -> Dict:
        """Get current analysis data for API response"""
        if not self.last_analysis:
            return {
                'timestamp': datetime.now().isoformat(),
                'price': 0,
                'signal': None,
                'suggestion': None
            }

        mtf = self.last_analysis.get('mtf_signal')
        suggestion = self.last_analysis.get('suggestion')

        signal_data = None
        if mtf:
            signal_data = {
                'combined_score': mtf.combined_score,
                'strength': mtf.combined_strength.value,
                'direction': mtf.primary_direction.value,
                'alignment': mtf.alignment_score,
                'timeframes': {}
            }
            for tf, sig in mtf.signals.items():
                signal_data['timeframes'][tf.value] = {
                    'score': sig.overall_score,
                    'strength': sig.strength.value,
                    'candlestick_score': sig.candlestick_score,
                    'chart_score': sig.chart_pattern_score,
                    'indicator_score': sig.indicator_score,
                    'patterns': [p.name for p in sig.candlestick_patterns[:5]],
                    'chart_patterns': [p.name for p in sig.chart_patterns[:3]],
                    'key_signals': [s.name for s in sig.indicator_signals if s.confidence >= 0.6][:5]
                }

        suggestion_data = None
        if suggestion:
            suggestion_data = {
                'direction': suggestion.direction.value,
                'entry': suggestion.entry_price,
                'stop_loss': suggestion.stop_loss,
                'take_profit_1': suggestion.take_profit_1,
                'take_profit_2': suggestion.take_profit_2,
                'take_profit_3': suggestion.take_profit_3,
                'risk_reward': suggestion.risk_reward_ratio,
                'confidence': suggestion.confidence,
                'reasoning': suggestion.reasoning
            }

        return {
            'timestamp': self.last_analysis.get('timestamp'),
            'price': self.last_analysis.get('price', 0),
            'signal': signal_data,
            'suggestion': suggestion_data,
            'signal_history': self.signal_history
        }


# Global analyzer instance
analyzer: Optional[WebAnalyzer] = None

import pandas as pd


# Routes
@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')


@app.route('/api/chart/<timeframe>')
def get_chart(timeframe):
    """Get chart data for a specific timeframe"""
    if analyzer:
        data = analyzer.get_chart_data(timeframe)
        return jsonify(data)
    return jsonify({'error': 'Analyzer not initialized'}), 500


@app.route('/api/analysis')
def get_analysis():
    """Get current analysis data"""
    if analyzer:
        return jsonify(analyzer.get_analysis_data())
    return jsonify({'error': 'Analyzer not initialized'}), 500


@app.route('/api/price')
def get_price():
    """Get current price"""
    if analyzer:
        try:
            price = analyzer.fetcher.get_current_price()
            return jsonify({'price': price})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Analyzer not initialized'}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("[INFO] Client connected")
    if analyzer:
        emit('analysis_update', analyzer.get_analysis_data())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("[INFO] Client disconnected")


@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request"""
    if analyzer:
        emit('analysis_update', analyzer.get_analysis_data())


@socketio.on('set_timeframe')
def handle_set_timeframe(data):
    """Handle timeframe change from client"""
    if analyzer and data and 'timeframe' in data:
        analyzer.set_timeframe(data['timeframe'])


def create_app(config: TradingConfig) -> Flask:
    """Create and configure the Flask app"""
    global analyzer
    analyzer = WebAnalyzer(config)
    analyzer.start()
    return app


def run_server(host: str = '0.0.0.0', port: int = 5000, config: Optional[TradingConfig] = None):
    """Run the web server"""
    global analyzer

    if config is None:
        config = TradingConfig()

    analyzer = WebAnalyzer(config)
    analyzer.start()

    print(f"\n{'='*50}")
    print("Bitcoin Market Analysis - Web Interface")
    print(f"{'='*50}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"{'='*50}\n")

    try:
        socketio.run(app, host=host, port=port, debug=False)
    finally:
        if analyzer:
            analyzer.stop()


if __name__ == '__main__':
    run_server()
