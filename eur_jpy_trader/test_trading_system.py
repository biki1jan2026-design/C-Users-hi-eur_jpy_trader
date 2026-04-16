#!/usr/bin/env python3
"""
Test Script for EUR/JPY Trading System
Validates all components and runs a demo backtest
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


class TestDataFeed(unittest.TestCase):
    """Test data feed module"""

    def test_candle_creation(self):
        from data.data_feed import Candle

        candle = Candle(
            timestamp=datetime.now(),
            open=162.500,
            high=162.550,
            low=162.450,
            close=162.520,
            volume=100
        )

        self.assertEqual(candle.open, 162.500)
        self.assertEqual(candle.high, 162.550)
        self.assertGreater(candle.high, candle.low)

    def test_candle_buffer(self):
        from data.data_feed import Candle, CandleBuffer

        buffer = CandleBuffer(max_size=10)

        for i in range(15):
            buffer.add(Candle(
                timestamp=datetime.now(),
                open=162.500 + i * 0.001,
                high=162.550 + i * 0.001,
                low=162.450 + i * 0.001,
                close=162.520 + i * 0.001,
                volume=100
            ))

        self.assertEqual(len(buffer), 10)  # Should cap at max_size

    def test_data_feed_historical(self):
        from data.data_feed import DataFeed

        feed = DataFeed()
        df = feed.fetch_historical_data(days=7)

        self.assertGreater(len(df), 0)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)


class TestIndicatorEngine(unittest.TestCase):
    """Test indicator calculations"""

    def setUp(self):
        """Create sample data"""
        np.random.seed(42)
        n = 200

        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        prices = 162.5 + np.cumsum(np.random.normal(0, 0.01, n))

        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 0.02, n)),
            'low': prices - np.abs(np.random.normal(0, 0.02, n)),
            'close': prices,
            'volume': np.random.uniform(50, 500, n)
        })

    def test_indicator_calculation(self):
        from indicators.indicator_engine import IndicatorEngine

        engine = IndicatorEngine(self.df)
        results = engine.get_all_indicators()

        # Check that indicators are calculated
        self.assertIn('ema_cross', results)
        self.assertIn('rsi', results)
        self.assertIn('macd', results)
        self.assertIn('bollinger', results)

    def test_signals_summary(self):
        from indicators.indicator_engine import IndicatorEngine

        engine = IndicatorEngine(self.df)
        summary = engine.get_signals_summary()

        self.assertIn('bullish_count', summary)
        self.assertIn('bearish_count', summary)
        self.assertIn('net_signal', summary)

    def test_feature_vector(self):
        from indicators.indicator_engine import IndicatorEngine

        engine = IndicatorEngine(self.df)
        features = engine.get_feature_vector()

        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)


class TestPortfolioManager(unittest.TestCase):
    """Test portfolio and risk management"""

    def test_position_sizing(self):
        from trading.portfolio_manager import PortfolioManager

        pm = PortfolioManager(initial_balance=50000, risk_per_trade=0.02)

        size = pm.calculate_position_size(
            entry_price=162.500,
            stop_loss_price=162.400,
            atr=0.05
        )

        self.assertGreater(size, 0)
        self.assertLessEqual(size, pm.balance * 0.1 / 650)  # Max 10% margin

    def test_open_position(self):
        from trading.portfolio_manager import PortfolioManager

        pm = PortfolioManager(initial_balance=50000)

        trade = pm.open_position(
            side='BUY',
            entry_price=162.500,
            atr=0.05
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.side, 'BUY')
        self.assertEqual(pm.portfolio_summary()['open_positions'], 1)

    def test_daily_loss_limit(self):
        from trading.portfolio_manager import PortfolioManager

        pm = PortfolioManager(
            initial_balance=50000,
            max_daily_loss=0.05
        )

        # Simulate hitting daily loss limit
        pm.daily_pnl = -50000 * 0.06  # Below limit

        trade = pm.open_position(
            side='BUY',
            entry_price=162.500,
            atr=0.05
        )

        self.assertIsNone(trade)  # Should not open


class TestBacktester(unittest.TestCase):
    """Test backtesting functionality"""

    def test_synthetic_data_generation(self):
        from backtester import generate_synthetic_data

        df = generate_synthetic_data(days=7)

        self.assertGreater(len(df), 0)
        self.assertIn('close', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)

    def test_backtest_results(self):
        from backtester import Backtester, generate_synthetic_data

        df = generate_synthetic_data(days=7)

        backtester = Backtester(
            initial_balance=50000,
            risk_per_trade=0.02
        )

        results = backtester.run_backtest(df, use_model=False)

        self.assertIn('total_trades', results)
        self.assertIn('win_rate', results)
        self.assertIn('total_pnl_usd', results)


class TestDecisionEngine(unittest.TestCase):
    """Test ML decision engine"""

    def test_model_initialization(self):
        from models.decision_engine import TradeDecisionModel

        model = TradeDecisionModel()

        self.assertIsNotNone(model.model)
        self.assertGreater(model.model.count_params(), 0)

    def test_prediction_output(self):
        from models.decision_engine import TradeDecisionModel

        model = TradeDecisionModel()

        # Create dummy feature sequence
        features = np.random.randn(1, 60, 60).astype(np.float32)

        probs, win_prob, expected_return = model.predict(features)

        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertGreaterEqual(win_prob, 0)
        self.assertLessEqual(win_prob, 1)


def run_all_tests():
    """Run all tests and print summary"""

    print("=" * 60)
    print("EUR/JPY Trading System - Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataFeed))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicatorEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktester))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
