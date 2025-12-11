"""
Live data fetching module using Binance API
Supports both REST API polling and WebSocket streaming
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass

import pandas as pd
import numpy as np
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager

from config import TimeFrame, TradingConfig, DEFAULT_TRADING_CONFIG


@dataclass
class Candle:
    """Represents a single candlestick"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        return self.high - self.low


class BinanceDataFetcher:
    """Fetches live Bitcoin data from Binance"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 config: TradingConfig = DEFAULT_TRADING_CONFIG):
        """
        Initialize the data fetcher.
        API key/secret are optional for public market data.
        """
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.config = config
        self.client: Optional[Client] = None
        self.async_client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self._running = False
        self._callbacks: List[Callable] = []

    def connect(self) -> None:
        """Establish connection to Binance REST API"""
        # Empty strings work for public endpoints
        self.client = Client(self.api_key, self.api_secret)
        print(f"[INFO] Connected to Binance API")

    async def connect_async(self) -> None:
        """Establish async connection for WebSocket"""
        self.async_client = await AsyncClient.create(self.api_key, self.api_secret)
        self.socket_manager = BinanceSocketManager(self.async_client)
        print(f"[INFO] Async client connected")

    def get_current_price(self) -> float:
        """Get current Bitcoin price"""
        if not self.client:
            self.connect()
        ticker = self.client.get_symbol_ticker(symbol=self.config.symbol)
        return float(ticker['price'])

    def get_klines(self, timeframe: TimeFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical klines (candlestick data)

        Returns DataFrame with columns:
        timestamp, open, high, low, close, volume
        """
        if not self.client:
            self.connect()

        limit = limit or self.config.candle_limit

        klines = self.client.get_klines(
            symbol=self.config.symbol,
            interval=timeframe.value,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)

        return df

    def get_multi_timeframe_data(self) -> Dict[TimeFrame, pd.DataFrame]:
        """Fetch data for all configured timeframes"""
        data = {}
        for tf in self.config.timeframes:
            data[tf] = self.get_klines(tf)
            print(f"[INFO] Fetched {len(data[tf])} candles for {tf.value}")
        return data

    def register_callback(self, callback: Callable[[str, dict], None]) -> None:
        """Register a callback function for real-time updates"""
        self._callbacks.append(callback)

    async def start_websocket_stream(self) -> None:
        """Start WebSocket stream for real-time kline updates"""
        if not self.async_client:
            await self.connect_async()

        self._running = True

        # Create kline socket for primary timeframe
        primary_tf = self.config.timeframes[0]
        socket = self.socket_manager.kline_socket(
            symbol=self.config.symbol,
            interval=primary_tf.value
        )

        print(f"[INFO] Starting WebSocket stream for {self.config.symbol} {primary_tf.value}")

        async with socket as stream:
            while self._running:
                try:
                    msg = await asyncio.wait_for(stream.recv(), timeout=30)
                    if msg:
                        self._process_websocket_message(msg)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"[ERROR] WebSocket error: {e}")
                    await asyncio.sleep(5)

    def _process_websocket_message(self, msg: dict) -> None:
        """Process incoming WebSocket message"""
        if msg.get('e') == 'kline':
            kline = msg['k']
            candle_data = {
                'symbol': kline['s'],
                'interval': kline['i'],
                'is_closed': kline['x'],  # True if candle is closed
                'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }

            # Notify all registered callbacks
            for callback in self._callbacks:
                try:
                    callback('kline', candle_data)
                except Exception as e:
                    print(f"[ERROR] Callback error: {e}")

    async def stop_websocket_stream(self) -> None:
        """Stop the WebSocket stream"""
        self._running = False
        if self.async_client:
            await self.async_client.close_connection()
            print("[INFO] WebSocket stream stopped")

    def close(self) -> None:
        """Close all connections"""
        if self.client:
            self.client = None
        print("[INFO] Connections closed")


class DataBuffer:
    """Maintains a rolling buffer of candle data for analysis"""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.data: Dict[TimeFrame, pd.DataFrame] = {}

    def update(self, timeframe: TimeFrame, df: pd.DataFrame) -> None:
        """Update buffer with new data"""
        if timeframe in self.data:
            # Append new data and remove duplicates
            combined = pd.concat([self.data[timeframe], df])
            combined = combined[~combined.index.duplicated(keep='last')]
            # Keep only max_size rows
            self.data[timeframe] = combined.tail(self.max_size)
        else:
            self.data[timeframe] = df.tail(self.max_size)

    def add_candle(self, timeframe: TimeFrame, candle: dict) -> None:
        """Add a single candle to the buffer"""
        new_row = pd.DataFrame([{
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }], index=[candle['timestamp']])

        if timeframe not in self.data:
            self.data[timeframe] = new_row
        else:
            # Update last candle or append
            if candle['timestamp'] in self.data[timeframe].index:
                self.data[timeframe].loc[candle['timestamp']] = new_row.iloc[0]
            else:
                self.data[timeframe] = pd.concat([self.data[timeframe], new_row])
                if len(self.data[timeframe]) > self.max_size:
                    self.data[timeframe] = self.data[timeframe].tail(self.max_size)

    def get(self, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        """Get data for a specific timeframe"""
        return self.data.get(timeframe)

    def get_latest_candle(self, timeframe: TimeFrame) -> Optional[pd.Series]:
        """Get the most recent candle for a timeframe"""
        df = self.data.get(timeframe)
        if df is not None and len(df) > 0:
            return df.iloc[-1]
        return None


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = BinanceDataFetcher()

    print("\n--- Testing Binance Data Fetcher ---\n")

    # Get current price
    price = fetcher.get_current_price()
    print(f"Current BTC price: ${price:,.2f}")

    # Get 1-hour candles
    df = fetcher.get_klines(TimeFrame.H1, limit=10)
    print(f"\nLast 10 hourly candles:")
    print(df.tail())

    fetcher.close()
