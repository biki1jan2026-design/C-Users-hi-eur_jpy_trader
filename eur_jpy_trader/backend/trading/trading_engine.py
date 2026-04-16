"""
Main Trading Engine
Coordinates data feed, indicators, decisions, and execution
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Callable, List
import json

from ..data.data_feed import DataFeed, Candle
from ..indicators.indicator_engine import IndicatorEngine
from ..models.decision_engine import TradeDecisionModel, DecisionEngine
from .portfolio_manager import PortfolioManager, Trade

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main orchestrator for the trading system.
    Coordinates all components and manages the trading loop.
    """

    def __init__(
        self,
        symbol: str = "EUR/JPY",
        initial_balance: float = 50000.0,
        risk_per_trade: float = 0.02,
        model_path: Optional[str] = None
    ):
        # Components
        self.data_feed = DataFeed(symbol=symbol)
        self.decision_model = TradeDecisionModel()
        self.decision_engine = DecisionEngine(self.decision_model)
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade
        )

        # State
        self._running = False
        self._last_decision_time: Optional[datetime] = None
        self._decision_cooldown = 60  # Seconds between decisions

        # Callbacks
        self._state_callbacks: List[Callable] = []

        # Load model if path provided
        if model_path:
            try:
                self.decision_model.load(model_path)
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

        # Register data feed callback
        self.data_feed.register_callback(self._on_candle_closed)

    def register_state_callback(self, callback: Callable):
        """Register callback for state updates (GUI updates)"""
        self._state_callbacks.append(callback)

    def _notify_state_change(self, state: Dict):
        """Notify all callbacks of state change"""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _on_candle_closed(self, candle: Candle):
        """Handle new closed candle - main decision point"""
        if not self._running:
            return

        # Get recent data
        df = self.data_feed.get_latest_candles(100)
        if len(df) < 60:
            logger.debug("Insufficient data for decision")
            return

        # Calculate indicators
        indicator_engine = IndicatorEngine(df)
        indicators = indicator_engine.get_all_indicators()
        signal_summary = indicator_engine.get_signals_summary()

        logger.debug(
            f"Indicators - Bullish: {signal_summary['bullish_count']}, "
            f"Bearish: {signal_summary['bearish_count']}"
        )

        # Make decision
        decision = self.decision_engine.make_decision(df, indicators)

        # Execute decision
        self._execute_decision(decision, df, indicators)

        # Notify state change
        self._notify_state_change(self.get_full_state())

    def _execute_decision(self, decision: Dict, df: pd.DataFrame, indicators: Dict):
        """Execute trading decision"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.05

        current_position = self.portfolio.get_open_position()

        if decision['action'] == 'WAIT':
            if current_position is None:
                logger.debug("Decision: WAIT (no position)")
                return

        elif decision['action'] == 'BUY':
            if current_position is None:
                # Open buy position
                trade = self.portfolio.open_position(
                    side='BUY',
                    entry_price=current_price,
                    atr=atr,
                    trigger_indicators={k: v.value for k, v in indicators.items()},
                    decision_probability=decision['win_probability'],
                    expected_return=decision['expected_return_pips']
                )
                if trade:
                    logger.info(
                        f"BUY signal executed at {current_price:.3f}, "
                        f"win_prob={decision['win_probability']:.2%}"
                    )
            else:
                logger.debug("BUY signal ignored - position already open")

        elif decision['action'] == 'SELL':
            if current_position is None:
                # Open sell position
                trade = self.portfolio.open_position(
                    side='SELL',
                    entry_price=current_price,
                    atr=atr,
                    trigger_indicators={k: v.value for k, v in indicators.items()},
                    decision_probability=decision['win_probability'],
                    expected_return=decision['expected_return_pips']
                )
                if trade:
                    logger.info(
                        f"SELL signal executed at {current_price:.3f}, "
                        f"win_prob={decision['win_probability']:.2%}"
                    )
            else:
                logger.debug("SELL signal ignored - position already open")

    def update_positions(self):
        """Check and update open positions"""
        if not self.data_feed.current_candle:
            return

        current_price = self.data_feed.current_candle.close
        current_time = self.data_feed.current_candle.timestamp

        trades_to_close = self.portfolio.update_position(current_price, current_time)

        for trade in trades_to_close:
            self.portfolio.close_position(trade, reason='auto')
            self.decision_engine.record_trade({
                'pnl_pips': trade.pnl_pips,
                'pnl_usd': trade.pnl_usd,
                'side': trade.side
            })

            logger.info(
                f"Trade closed: {trade.id}, PnL={trade.pnl_pips:.2f} pips "
                f"(${trade.pnl_usd:.2f})"
            )

        if trades_to_close:
            self._notify_state_change(self.get_full_state())

    def get_full_state(self) -> Dict:
        """Get complete system state for GUI"""
        portfolio_summary = self.portfolio.get_portfolio_summary()
        performance = self.portfolio.get_performance_stats()
        decision_stats = self.decision_engine.get_performance_stats()

        # Current candle
        current_candle = self.data_feed.get_current_candle()
        current_candle_data = current_candle.to_dict() if current_candle else None

        # Recent candles for chart
        recent_candles = self.data_feed.get_latest_candles(50)
        candles_data = recent_candles.to_dict('records') if len(recent_candles) > 0 else []

        # Last decision
        last_decision = None
        if hasattr(self, '_last_decision'):
            last_decision = self._last_decision

        return {
            'timestamp': datetime.now().isoformat(),
            'balance': portfolio_summary['balance'],
            'total_pnl': portfolio_summary['total_pnl'],
            'total_pnl_pct': portfolio_summary['total_pnl_pct'],
            'daily_pnl': portfolio_summary['daily_pnl'],
            'open_position': portfolio_summary['current_trade'],
            'performance': performance,
            'decision_stats': decision_stats,
            'current_candle': current_candle_data,
            'recent_candles': candles_data,
            'last_decision': last_decision
        }

    async def start(self):
        """Start the trading engine"""
        self._running = True
        logger.info("Trading engine started")

        # Load historical data
        historical_df = await self.data_feed.fetch_historical_data(days=30)
        self.data_feed.load_historical_to_buffer(historical_df)

        logger.info(f"Loaded {len(historical_df)} historical candles")

        # Start live feed
        asyncio.create_task(self.data_feed.start_live_feed())

        # Main loop
        while self._running:
            self.portfolio.reset_daily_counters()
            self.update_positions()

            await asyncio.sleep(1)

    def stop(self):
        """Stop the trading engine"""
        self._running = False
        self.data_feed.stop()
        logger.info("Trading engine stopped")

    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.portfolio.get_trade_history()

    def get_screenshot_data(self) -> Dict:
        """Get data needed for screenshot (chart state, indicators)"""
        df = self.data_feed.get_latest_candles(100)
        if len(df) < 2:
            return {}

        indicator_engine = IndicatorEngine(df)

        return {
            'candles': df.to_dict('records'),
            'indicators': {k: v.value for k, v in indicator_engine.get_all_indicators().items()},
            'signals': indicator_engine.get_signals_summary(),
            'timestamp': datetime.now().isoformat()
        }


# Import pandas for type hints
import pandas as pd
