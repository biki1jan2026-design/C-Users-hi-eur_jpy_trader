"""
Backtesting and Validation System
Simulates 30 days of 1-minute data and reports aggregate P/L and win rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

from data.data_feed import DataFeed, Candle
from indicators.indicator_engine import IndicatorEngine
from models.decision_engine import TradeDecisionModel, DecisionEngine
from trading.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Runs backtesting simulations on historical or synthetic data.
    """

    def __init__(
        self,
        initial_balance: float = 50000.0,
        risk_per_trade: float = 0.02,
        holding_period: int = 5,
        atr_multiplier: float = 2.0
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.holding_period = holding_period
        self.atr_multiplier = atr_multiplier

        self.model = TradeDecisionModel()
        self.decision_engine = DecisionEngine(self.model)
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            atr_multiplier=atr_multiplier
        )

        self.trades = []
        self.decisions = []

    def run_backtest(
        self,
        historical_df: pd.DataFrame,
        use_model: bool = True
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            historical_df: DataFrame with OHLCV data
            use_model: Whether to use ML model or simple indicator logic
        """

        logger.info(f"Starting backtest on {len(historical_df)} candles")

        # Initialize indicators
        indicator_engine = IndicatorEngine(historical_df)

        # Track state
        open_position = None
        entry_candle_idx = 0

        for i in range(60, len(historical_df) - self.holding_period):
            # Get data up to this point
            df = historical_df.iloc[:i].copy()

            # Recalculate indicators for current data
            try:
                ind_engine = IndicatorEngine(df)
                indicators = ind_engine.get_all_indicators()
            except Exception as e:
                logger.debug(f"Indicator calculation failed at {i}: {e}")
                continue

            # Get current candle
            current_candle = df.iloc[-1]
            current_price = current_candle['close']
            current_time = current_candle['timestamp'] if 'timestamp' in df.columns else datetime.now()

            # Make decision
            if use_model:
                decision = self.decision_engine.make_decision(df, indicators)
            else:
                decision = self._simple_decision(indicators, df)

            self.decisions.append({
                'index': i,
                'timestamp': current_time,
                'action': decision['action'],
                'probability': decision['probabilities'],
                'win_probability': decision['win_probability']
            })

            # Check existing position for exit
            if open_position:
                hold_time = i - entry_candle_idx

                # Check if we should exit
                should_exit = False
                exit_price = current_price
                exit_reason = 'time_exit'

                # Check ATR stop
                if open_position['side'] == 'BUY':
                    stop_price = open_position['entry'] - (open_position['atr'] * self.atr_multiplier)
                    if current_price <= stop_price:
                        should_exit = True
                        exit_reason = 'stop_loss'
                else:
                    stop_price = open_position['entry'] + (open_position['atr'] * self.atr_multiplier)
                    if current_price >= stop_price:
                        should_exit = True
                        exit_reason = 'stop_loss'

                # Check time exit
                if hold_time >= self.holding_period:
                    should_exit = True
                    exit_reason = 'time_exit'

                if should_exit:
                    # Close position
                    if open_position['side'] == 'BUY':
                        pnl_pips = (exit_price - open_position['entry']) * 100
                    else:
                        pnl_pips = (open_position['entry'] - exit_price) * 100

                    pnl_usd = pnl_pips * open_position['size'] * 6.50

                    self.portfolio.balance += pnl_usd

                    trade_record = {
                        'entry_time': open_position['time'],
                        'exit_time': current_time,
                        'side': open_position['side'],
                        'entry_price': open_position['entry'],
                        'exit_price': exit_price,
                        'pnl_pips': pnl_pips,
                        'pnl_usd': pnl_usd,
                        'size': open_position['size'],
                        'exit_reason': exit_reason,
                        'win_probability': open_position['win_prob']
                    }

                    self.trades.append(trade_record)
                    self.decision_engine.record_trade(trade_record)

                    open_position = None

            # Open new position if signal and no existing position
            if decision['action'] in ['BUY', 'SELL'] and open_position is None:
                # Get ATR
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.05

                # Calculate position size
                risk_amount = self.portfolio.balance * self.risk_per_trade
                stop_distance = atr * self.atr_multiplier
                stop_pips = stop_distance * 100
                size = risk_amount / (stop_pips * 6.50) if stop_pips > 0 else 0.1

                open_position = {
                    'side': decision['action'],
                    'entry': current_price,
                    'time': current_time,
                    'atr': atr,
                    'size': size,
                    'win_prob': decision['win_probability']
                }

                entry_candle_idx = i

        # Close any remaining open position at last price
        if open_position:
            last_candle = historical_df.iloc[-1]
            last_price = last_candle['close']

            if open_position['side'] == 'BUY':
                pnl_pips = (last_price - open_position['entry']) * 100
            else:
                pnl_pips = (open_position['entry'] - last_price) * 100

            pnl_usd = pnl_pips * open_position['size'] * 6.50
            self.portfolio.balance += pnl_usd

            self.trades.append({
                'entry_time': open_position['time'],
                'exit_time': last_candle.get('timestamp', datetime.now()),
                'side': open_position['side'],
                'entry_price': open_position['entry'],
                'exit_price': last_price,
                'pnl_pips': pnl_pips,
                'pnl_usd': pnl_usd,
                'size': open_position['size'],
                'exit_reason': 'end_of_data'
            })

        return self.get_results()

    def _simple_decision(
        self,
        indicators: Dict,
        df: pd.DataFrame
    ) -> Dict:
        """Simple indicator-based decision (no ML model)"""

        # Count bullish vs bearish signals
        bullish = 0
        bearish = 0

        for name, result in indicators.items():
            if result.signal == 1:
                bullish += 1
            elif result.signal == -1:
                bearish += 1

        total = bullish + bearish
        if total == 0:
            action = 'WAIT'
            buy_prob = 0.33
            sell_prob = 0.33
            wait_prob = 0.34
        elif bullish > bearish * 1.5:
            action = 'BUY'
            buy_prob = min(0.5 + (bullish - bearish) / total * 0.3, 0.8)
            sell_prob = 0.2
            wait_prob = 1 - buy_prob - sell_prob
        elif bearish > bullish * 1.5:
            action = 'SELL'
            sell_prob = min(0.5 + (bearish - bullish) / total * 0.3, 0.8)
            buy_prob = 0.2
            wait_prob = 1 - buy_prob - sell_prob
        else:
            action = 'WAIT'
            buy_prob = 0.3
            sell_prob = 0.3
            wait_prob = 0.4

        return {
            'action': action,
            'probabilities': [buy_prob, sell_prob, wait_prob],
            'win_probability': max(buy_prob, sell_prob),
            'expected_return_pips': np.random.uniform(-2, 2)
        }

    def get_results(self) -> Dict:
        """Calculate and return backtest results"""

        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl_pips': 0,
                'total_pnl_usd': 0,
                'final_balance': self.portfolio.balance,
                'return_pct': 0,
                'avg_win_pips': 0,
                'avg_loss_pips': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': []
            }

        # Basic stats
        wins = [t for t in self.trades if t['pnl_pips'] > 0]
        losses = [t for t in self.trades if t['pnl_pips'] <= 0]

        win_pips = [t['pnl_pips'] for t in wins]
        loss_pips = [t['pnl_pips'] for t in losses]

        total_pnl_pips = sum(t['pnl_pips'] for t in self.trades)
        total_pnl_usd = sum(t['pnl_usd'] for t in self.trades)

        # Calculate drawdown
        balance_curve = [self.initial_balance]
        for trade in self.trades:
            balance_curve.append(balance_curve[-1] + trade['pnl_usd'])

        max_balance = max(balance_curve)
        max_drawdown = 0
        for balance in balance_curve:
            drawdown = (max_balance - balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio (assuming 252 trading days, 1440 minutes per day)
        if len(self.trades) > 1:
            returns = [t['pnl_usd'] / self.initial_balance for t in self.trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * np.sqrt(252 * 1440 / self.holding_period) if std_return > 0 else 0
        else:
            sharpe = 0

        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_usd': total_pnl_usd,
            'final_balance': self.portfolio.balance,
            'return_pct': (self.portfolio.balance - self.initial_balance) / self.initial_balance * 100,
            'avg_win_pips': sum(win_pips) / len(win_pips) if win_pips else 0,
            'avg_loss_pips': sum(loss_pips) / len(loss_pips) if loss_pips else 0,
            'best_trade': max(t['pnl_pips'] for t in self.trades),
            'worst_trade': min(t['pnl_pips'] for t in self.trades),
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'profit_factor': abs(sum(win_pips) / sum(loss_pips)) if loss_pips and sum(loss_pips) != 0 else float('inf'),
            'trades': self.trades,
            'decisions': self.decisions
        }


def generate_synthetic_data(
    days: int = 30,
    timeframe_minutes: int = 1
) -> pd.DataFrame:
    """
    Generate realistic synthetic EUR/JPY data for backtesting.
    """

    logger.info(f"Generating {days} days of synthetic {timeframe_minutes}-min data")

    # Calculate total candles (excluding weekends)
    minutes_per_day = 24 * 60 // timeframe_minutes
    trading_days = days * 5 / 7  # Exclude weekends
    total_candles = int(trading_days * minutes_per_day)

    logger.info(f"Generating {total_candles} candles")

    # Generate price series
    np.random.seed(42)

    base_price = 162.500

    # Random walk with mean reversion and trends
    returns = np.random.normal(0, 0.0001, total_candles)

    # Add intraday patterns (higher volatility during London/Tokyo overlap)
    for i in range(total_candles):
        hour = (i // (60 // timeframe_minutes)) % 24
        # Higher volatility during active hours (7-17 UTC)
        if 7 <= hour <= 17:
            returns[i] *= 1.5

    # Add some trending periods
    trend_periods = [
        (0, 500, 0.00005),   # Uptrend
        (500, 1000, -0.00003),  # Downtrend
        (1000, 1500, 0.00002),  # Slight uptrend
        (1500, 2000, -0.00004), # Downtrend
        (2000, total_candles, 0.00006)  # Uptrend
    ]

    for start, end, trend in trend_periods:
        if start < total_candles:
            actual_end = min(end, total_candles)
            returns[start:actual_end] += trend

    # Generate prices
    prices = base_price + np.cumsum(returns)

    # Generate OHLCV
    data = []
    base_time = datetime.now() - timedelta(days=days)

    for i in range(total_candles):
        timestamp = base_time + timedelta(minutes=i * timeframe_minutes)

        open_price = prices[i]
        close_price = prices[i] if i >= len(prices) else prices[min(i + 1, len(prices) - 1)]

        # Add some noise for high/low
        volatility = abs(returns[i]) * 2 if i < len(returns) else 0.01
        high_price = max(open_price, close_price) + volatility * np.random.uniform(0.5, 1.5)
        low_price = min(open_price, close_price) - volatility * np.random.uniform(0.5, 1.5)

        volume = np.random.uniform(50, 500) * (1.5 if 7 <= (i // 60) % 24 <= 17 else 0.8)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} synthetic candles")

    return df


def run_backtest_demo():
    """Run a demo backtest and print results"""

    print("=" * 60)
    print("EUR/JPY 1-Minute MAI-Trader Backtest")
    print("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_data(days=30)

    # Run backtest
    backtester = Backtester(
        initial_balance=50000,
        risk_per_trade=0.02,
        holding_period=5,
        atr_multiplier=2.0
    )

    results = backtester.run_backtest(df, use_model=False)

    # Print results
    print(f"\n{'=' * 40}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 40}\n")

    print(f"Total Trades:        {results['total_trades']}")
    print(f"Win Rate:            {results['win_rate']:.1%}")
    print(f"Total P/L (pips):    {results['total_pnl_pips']:+.2f}")
    print(f"Total P/L (USD):     ${results['total_pnl_usd']:+,.2f}")
    print(f"Return:              {results['return_pct']:+.2f}%")
    print(f"Final Balance:       ${results['final_balance']:,.2f}")
    print()
    print(f"Average Win:         {results['avg_win_pips']:+.2f} pips")
    print(f"Average Loss:        {results['avg_loss_pips']:+.2f} pips")
    print(f"Best Trade:          {results['best_trade']:+.2f} pips")
    print(f"Worst Trade:         {results['worst_trade']:+.2f} pips")
    print(f"Profit Factor:       {results['profit_factor']:.2f}")
    print(f"Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")

    print(f"\n{'=' * 40}")

    return results


if __name__ == "__main__":
    run_backtest_demo()
