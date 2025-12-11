#!/usr/bin/env python3
"""
Bitcoin Market Analysis Tool
Real-time trading signal detection with audio notifications

Usage:
    python main.py [--interval SECONDS] [--no-audio] [--timeframes TF1,TF2,...]
"""
import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from config import TimeFrame, TradingConfig, SignalType
from data_fetcher import BinanceDataFetcher, DataBuffer
from signal_aggregator import (
    SignalAggregator, AggregatedSignal, MultiTimeframeSignal,
    format_signal_report, format_mtf_report, SignalStrength
)
from trade_suggester import TradeSuggester, TradeSuggestion, format_trade_alert, TradeDirection
from notifier import AudioNotifier, DesktopNotifier, NotificationType

console = Console()


class BitcoinAnalyzer:
    """Main application class for Bitcoin market analysis"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.fetcher = BinanceDataFetcher(config=config)
        self.buffer = DataBuffer()
        self.aggregator = SignalAggregator(config)
        self.suggester = TradeSuggester(config)
        self.audio = AudioNotifier(config)
        self.desktop = DesktopNotifier()

        self.running = False
        self.last_signal: Optional[AggregatedSignal] = None
        self.last_suggestion: Optional[TradeSuggestion] = None
        self.signal_history = []
        self.analysis_count = 0

    def start(self) -> None:
        """Start the analyzer"""
        console.print("\n[bold cyan]Bitcoin Market Analysis Tool[/bold cyan]")
        console.print("=" * 50)
        console.print(f"Symbol: {self.config.symbol}")
        console.print(f"Timeframes: {', '.join(tf.value for tf in self.config.timeframes)}")
        console.print(f"Poll Interval: {self.config.poll_interval}s")
        console.print(f"Audio: {'Enabled' if self.config.enable_audio else 'Disabled'}")
        console.print("=" * 50)
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

        self.running = True

        # Initial data fetch
        try:
            self._fetch_initial_data()
        except Exception as e:
            console.print(f"[red]Error fetching initial data: {e}[/red]")
            console.print("[yellow]Make sure you have internet connection and Binance API is accessible.[/yellow]")
            return

        # Main analysis loop
        self._run_analysis_loop()

    def _fetch_initial_data(self) -> None:
        """Fetch initial historical data for all timeframes"""
        console.print("[yellow]Fetching historical data...[/yellow]")

        data = self.fetcher.get_multi_timeframe_data()
        for tf, df in data.items():
            self.buffer.update(tf, df)
            console.print(f"  ✓ {tf.value}: {len(df)} candles loaded")

        console.print("[green]Initial data loaded successfully![/green]\n")

    def _run_analysis_loop(self) -> None:
        """Main polling/analysis loop"""
        try:
            while self.running:
                self._perform_analysis()
                self._display_status()

                # Wait for next interval
                for _ in range(self.config.poll_interval):
                    if not self.running:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            self.stop()

    def _perform_analysis(self) -> None:
        """Perform market analysis on all timeframes"""
        self.analysis_count += 1

        try:
            # Refresh data
            data = self.fetcher.get_multi_timeframe_data()
            for tf, df in data.items():
                self.buffer.update(tf, df)

            # Multi-timeframe analysis
            buffer_data = {tf: self.buffer.get(tf) for tf in self.config.timeframes
                          if self.buffer.get(tf) is not None}

            if not buffer_data:
                console.print("[red]No data available for analysis[/red]")
                return

            mtf_signal = self.aggregator.analyze_multi_timeframe(buffer_data)

            # Check if we should notify
            if self.aggregator.should_notify_mtf(mtf_signal):
                self._handle_signal(mtf_signal, buffer_data)

            self.last_signal = list(mtf_signal.signals.values())[0] if mtf_signal.signals else None

        except Exception as e:
            console.print(f"[red]Analysis error: {e}[/red]")

    def _handle_signal(self, mtf_signal: MultiTimeframeSignal, data: dict) -> None:
        """Handle a detected trading signal"""
        # Get primary timeframe data for trade suggestion
        primary_tf = min(data.keys(), key=lambda x: x.value)
        primary_df = data[primary_tf]

        # Generate trade suggestion
        suggestion = self.suggester.suggest_from_mtf(mtf_signal, primary_df)

        if suggestion:
            self.last_suggestion = suggestion
            self._display_signal_alert(mtf_signal, suggestion)
            self._send_notifications(mtf_signal, suggestion)

            # Store in history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signal': mtf_signal,
                'suggestion': suggestion
            })

            # Keep only last 50 signals
            if len(self.signal_history) > 50:
                self.signal_history = self.signal_history[-50:]

    def _display_signal_alert(self, mtf_signal: MultiTimeframeSignal,
                              suggestion: TradeSuggestion) -> None:
        """Display signal alert in console"""
        console.print("\n")

        # Get current price
        price = suggestion.entry_price

        # Display trade alert
        alert = format_trade_alert(suggestion, price)
        console.print(alert)

        # Display MTF breakdown
        console.print("\n[bold]Multi-Timeframe Analysis:[/bold]")
        console.print(format_mtf_report(mtf_signal))

        console.print("\n")

    def _send_notifications(self, mtf_signal: MultiTimeframeSignal,
                           suggestion: TradeSuggestion) -> None:
        """Send audio and desktop notifications"""
        price = suggestion.entry_price
        strength = suggestion.signal_strength

        # Audio notification
        if suggestion.direction == TradeDirection.LONG:
            self.audio.notify_bullish(strength, price)
        else:
            self.audio.notify_bearish(strength, price)

        # Desktop notification
        direction = "BULLISH" if suggestion.direction == TradeDirection.LONG else "BEARISH"
        self.desktop.notify_signal(
            direction=direction,
            strength=strength,
            price=price,
            score=suggestion.signal_score,
            timeframe=self.config.timeframes[0].value
        )

    def _display_status(self) -> None:
        """Display current status"""
        try:
            price = self.fetcher.get_current_price()
        except:
            price = 0

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build status line
        status_parts = [
            f"[dim]{timestamp}[/dim]",
            f"BTC: [bold]${price:,.2f}[/bold]",
            f"Analysis #{self.analysis_count}",
        ]

        if self.last_signal:
            score = self.last_signal.overall_score
            if score > 0:
                status_parts.append(f"[green]▲ {score:+.2f}[/green]")
            elif score < 0:
                status_parts.append(f"[red]▼ {score:+.2f}[/red]")
            else:
                status_parts.append(f"[yellow]● {score:+.2f}[/yellow]")

        status_parts.append(f"Signals: {len(self.signal_history)}")

        console.print(" | ".join(status_parts))

    def stop(self) -> None:
        """Stop the analyzer"""
        self.running = False
        self.fetcher.close()
        console.print("[green]Analyzer stopped.[/green]")


def create_dashboard(analyzer: BitcoinAnalyzer) -> Table:
    """Create a rich dashboard table"""
    table = Table(title="Bitcoin Market Analysis")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if analyzer.last_signal:
        table.add_row("Price", f"${analyzer.last_signal.current_price:,.2f}")
        table.add_row("Signal Score", f"{analyzer.last_signal.overall_score:+.2f}")
        table.add_row("Strength", analyzer.last_signal.strength.value)
        table.add_row("Candlestick", f"{analyzer.last_signal.candlestick_score:+.2f}")
        table.add_row("Chart Pattern", f"{analyzer.last_signal.chart_pattern_score:+.2f}")
        table.add_row("Indicators", f"{analyzer.last_signal.indicator_score:+.2f}")

    return table


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Bitcoin Market Analysis Tool - Real-time trading signal detection"
    )

    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='Run web interface instead of CLI'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port for web server (default: 5000)'
    )

    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Polling interval in seconds (default: 60)'
    )

    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio notifications'
    )

    parser.add_argument(
        '--timeframes', '-t',
        type=str,
        default='15m,1h,4h',
        help='Comma-separated timeframes to analyze (default: 15m,1h,4h)'
    )

    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )

    parser.add_argument(
        '--min-score',
        type=float,
        default=0.4,
        help='Minimum signal score to trigger alerts (default: 0.4)'
    )

    parser.add_argument(
        '--test-audio',
        action='store_true',
        help='Test audio notifications and exit'
    )

    return parser.parse_args()


def parse_timeframes(tf_string: str) -> list:
    """Parse timeframe string into TimeFrame enum list"""
    tf_map = {
        '1m': TimeFrame.M1,
        '5m': TimeFrame.M5,
        '15m': TimeFrame.M15,
        '30m': TimeFrame.M30,
        '1h': TimeFrame.H1,
        '4h': TimeFrame.H4,
        '1d': TimeFrame.D1,
    }

    timeframes = []
    for tf in tf_string.split(','):
        tf = tf.strip().lower()
        if tf in tf_map:
            timeframes.append(tf_map[tf])
        else:
            console.print(f"[yellow]Warning: Unknown timeframe '{tf}', skipping[/yellow]")

    return timeframes if timeframes else [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]


def main():
    """Main entry point"""
    args = parse_args()

    # Test audio mode
    if args.test_audio:
        console.print("[cyan]Testing audio notifications...[/cyan]")
        notifier = AudioNotifier()
        notifier.test_notifications()
        return

    # Parse timeframes
    timeframes = parse_timeframes(args.timeframes)

    # Create config
    config = TradingConfig(
        symbol=args.symbol,
        timeframes=timeframes,
        poll_interval=args.interval,
        enable_audio=not args.no_audio,
        min_signal_score=args.min_score
    )

    # Web mode
    if args.web:
        console.print("[cyan]Starting web interface...[/cyan]")
        from web_server import run_server
        run_server(host='0.0.0.0', port=args.port, config=config)
        return

    # CLI mode
    # Create and start analyzer
    analyzer = BitcoinAnalyzer(config)

    # Setup signal handler
    def signal_handler(sig, frame):
        console.print("\n[yellow]Received interrupt signal...[/yellow]")
        analyzer.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start analysis
    analyzer.start()


if __name__ == "__main__":
    main()
