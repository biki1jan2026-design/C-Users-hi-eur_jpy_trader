"""
Real-time EUR/JPY Data Feed Module
Handles 1-minute candle streaming and historical data management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """Single OHLCV candlestick data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class CandleBuffer:
    """Circular buffer for storing recent candles"""
    max_size: int = 500
    candles: List[Candle] = field(default_factory=list)

    def add(self, candle: Candle):
        self.candles.append(candle)
        if len(self.candles) > self.max_size:
            self.candles = self.candles[-self.max_size:]

    def get_dataframe(self) -> pd.DataFrame:
        if not self.candles:
            return pd.DataFrame()
        return pd.DataFrame([c.to_dict() for c in self.candles])

    def get_latest(self, n: int = 1) -> List[Candle]:
        return self.candles[-n:] if len(self.candles) >= n else self.candles

    def __len__(self) -> int:
        return len(self.candles)


class DataFeed:
    """
    Real-time data feed for EUR/JPY 1-minute candles.
    Simulates tick data when no live connection is available.
    """

    def __init__(self, symbol: str = "EUR/JPY", timeframe: str = "1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer = CandleBuffer(max_size=500)
        self.current_candle: Optional[Candle] = None
        self.callbacks: List[Callable] = []
        self._running = False
        self._last_tick_time: Optional[datetime] = None

        # EUR/JPY typical price range (around 160-165)
        self._base_price = 162.500
        self._price_std = 0.015

    def register_callback(self, callback: Callable):
        """Register callback for new candle events"""
        self.callbacks.append(callback)

    async def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch or generate historical data for backtesting.
        In production, this would call broker API.
        """
        logger.info(f"Fetching {days} days of historical data for {self.symbol}")

        # Generate realistic EUR/JPY 1-minute data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Calculate number of 1-minute candles (excluding weekends)
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        # Exclude weekends (~2/7 of time)
        trading_minutes = int(total_minutes * 5/7)

        timestamps = pd.date_range(
            start=start_time,
            periods=trading_minutes,
            freq='1min'
        )

        # Generate realistic price series with trends and volatility
        np.random.seed(42)  # Reproducibility

        # Random walk with mean reversion
        returns = np.random.normal(0, self._price_std, trading_minutes)

        # Add some trend components
        trend = np.sin(np.linspace(0, 4 * np.pi, trading_minutes)) * 0.05
        returns += trend

        # Cumulative returns to get price
        prices = self._base_price + np.cumsum(returns)

        # Generate OHLC from prices
        data = []
        for i, ts in enumerate(timestamps):
            base_price = prices[i]
            volatility = abs(returns[i]) if i < len(returns) else self._price_std

            open_price = base_price
            close_price = base_price + returns[i] if i < len(returns) else base_price
            high_price = max(open_price, close_price) + volatility * np.random.uniform(0.5, 1.5)
            low_price = min(open_price, close_price) - volatility * np.random.uniform(0.5, 1.5)
            volume = np.random.uniform(50, 500)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} historical candles")
        return df

    def load_historical_to_buffer(self, df: pd.DataFrame):
        """Load historical dataframe into candle buffer"""
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            self.buffer.add(candle)
        logger.info(f"Loaded {len(df)} candles into buffer")

    async def start_live_feed(self):
        """
        Start real-time data feed.
        Simulates tick data when no broker connection is available.
        """
        self._running = True
        logger.info(f"Starting live feed for {self.symbol}")

        # Initialize current candle from last historical
        last_candle = self.buffer.get_latest(1)[0] if len(self.buffer) > 0 else None
        if last_candle:
            self.current_candle = Candle(
                timestamp=datetime.now().replace(second=0, microsecond=0),
                open=last_candle.close,
                high=last_candle.close,
                low=last_candle.close,
                close=last_candle.close,
                volume=0
            )
        else:
            self.current_candle = Candle(
                timestamp=datetime.now().replace(second=0, microsecond=0),
                open=self._base_price,
                high=self._base_price,
                low=self._base_price,
                close=self._base_price,
                volume=0
            )

        while self._running:
            await self._simulate_tick()
            await asyncio.sleep(1)  # Simulate ~1 tick per second

    async def _simulate_tick(self):
        """Simulate incoming tick data"""
        now = datetime.now()

        # Check if we need to close current candle and start new one
        if now.minute != self.current_candle.timestamp.minute or \
           now.hour != self.current_candle.timestamp.hour:
            # Close current candle
            self.buffer.add(self.current_candle)
            self._notify_candle_closed(self.current_candle)

            # Start new candle
            self.current_candle = Candle(
                timestamp=now.replace(second=0, microsecond=0),
                open=self.current_candle.close,
                high=self.current_candle.close,
                low=self.current_candle.close,
                close=self.current_candle.close,
                volume=0
            )

        # Simulate price movement
        tick_change = np.random.normal(0, 0.003)
        new_price = self.current_candle.close + tick_change

        self.current_candle.close = new_price
        self.current_candle.high = max(self.current_candle.high, new_price)
        self.current_candle.low = min(self.current_candle.low, new_price)
        self.current_candle.volume += np.random.uniform(0.1, 5)
        self.current_candle.timestamp = now

    def _notify_candle_closed(self, candle: Candle):
        """Notify all registered callbacks of new closed candle"""
        for callback in self.callbacks:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_current_candle(self) -> Optional[Candle]:
        """Get the currently forming candle"""
        return self.current_candle

    def get_latest_candles(self, n: int = 100) -> pd.DataFrame:
        """Get last n candles as DataFrame"""
        return self.buffer.get_dataframe().tail(n)

    def stop(self):
        """Stop the live feed"""
        self._running = False
        logger.info("Live feed stopped")
