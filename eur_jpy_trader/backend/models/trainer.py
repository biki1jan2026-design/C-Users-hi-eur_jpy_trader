"""
Model Training Module
Handles data preparation, training, and validation for the decision model
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from datetime import datetime, timedelta
import logging

from .decision_engine import TradeDecisionModel
from ..indicators.indicator_engine import IndicatorEngine
from ..data.data_feed import DataFeed

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Prepares training data and trains the decision model.
    """

    def __init__(
        self,
        model: TradeDecisionModel,
        lookback_period: int = 60,
        hold_period: int = 5
    ):
        self.model = model
        self.lookback_period = lookback_period
        self.hold_period = hold_period

    def prepare_training_data(
        self,
        historical_df: pd.DataFrame,
        test_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical prices.

        Creates sequences of length `lookback_period` with labels
        based on whether holding for `hold_period` minutes would be profitable.
        """

        logger.info(f"Preparing training data from {len(historical_df)} candles")

        # Calculate indicators for entire dataset
        indicator_engine = IndicatorEngine(historical_df)
        indicator_df = indicator_engine.get_dataframe()

        X_sequences = []
        y_labels = []

        for i in range(self.lookback_period, len(historical_df) - self.hold_period):
            # Get sequence of past data
            sequence_df = historical_df.iloc[i - self.lookback_period:i]

            # Get indicators at decision point
            indicators = indicator_engine._results  # Use latest indicators

            # Prepare features
            features = self.model.prepare_features(sequence_df, indicators)
            X_sequences.append(features)

            # Determine label based on future price movement
            current_price = historical_df['close'].iloc[i]
            future_prices = historical_df['close'].iloc[i:i + self.hold_period]

            # Find best exit point during hold period
            max_price = future_prices.max()
            min_price = future_prices.min()

            # Calculate potential outcomes
            long_return = (max_price - current_price) / current_price * 10000  # pips
            short_return = (current_price - min_price) / current_price * 10000

            # Label: best action (Buy=0, Sell=1, Wait=2)
            threshold = 2  # Minimum pips to consider worthwhile

            if long_return > threshold and long_return > short_return:
                label = [1, 0, 0]  # Buy
            elif short_return > threshold and short_return > long_return:
                label = [0, 1, 0]  # Sell
            else:
                label = [0, 0, 1]  # Wait

            y_labels.append(label)

        X = np.array(X_sequences)
        y = np.array(y_labels)

        logger.info(f"Generated {len(X)} training samples")
        logger.info(f"Label distribution: Buy={sum(y[:, 0])}, Sell={sum(y[:, 1])}, Wait={sum(y[:, 2])}")

        # Train/test split
        split_idx = int(len(X) * (1 - test_split))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        return X_train, y_train, X_test, y_test

    def train(
        self,
        historical_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32
    ) -> dict:
        """
        Full training pipeline.
        """

        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_training_data(historical_df)

        # Train model
        history = self.model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size
        )

        # Evaluate
        test_loss, test_accuracy = self.model.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        return {
            'history': history.history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

    def generate_synthetic_data(
        self,
        n_samples: int = 10000,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for initial model warm-up.
        Uses realistic EUR/JPY statistical properties.
        """

        logger.info(f"Generating {n_samples} synthetic samples")

        X = np.zeros((n_samples, sequence_length, self.model.input_dim))
        y = np.zeros((n_samples, 3))

        for i in range(n_samples):
            # Generate realistic price sequence
            base_price = 162.5
            returns = np.random.normal(0, 0.0001, sequence_length + 10)

            # Add some patterns
            if np.random.random() > 0.5:
                # Uptrend
                returns += np.linspace(0, 0.0002, sequence_length + 10)
            else:
                # Downtrend
                returns -= np.linspace(0, 0.0002, sequence_length + 10)

            prices = base_price + np.cumsum(returns)

            # Create OHLCV dataframe
            df = pd.DataFrame({
                'close': prices[10:],
                'open': np.roll(prices[10:], 1),
                'high': prices[10:] * (1 + np.abs(np.random.normal(0, 0.0005, sequence_length))),
                'low': prices[10:] * (1 - np.abs(np.random.normal(0, 0.0005, sequence_length))),
                'volume': np.random.uniform(100, 1000, sequence_length)
            })
            df['open'] = df['open'].fillna(df['close'].iloc[0])

            # Generate indicators
            try:
                indicator_engine = IndicatorEngine(df)
                indicators = indicator_engine.get_all_indicators()
            except:
                indicators = {}

            # Prepare features
            features = self.model.prepare_features(df, indicators)
            X[i] = features

            # Determine label from price movement
            price_change = prices[-1] - prices[0]
            if price_change > 0.01:
                y[i] = [1, 0, 0]  # Buy
            elif price_change < -0.01:
                y[i] = [0, 1, 0]  # Sell
            else:
                y[i] = [0, 0, 1]  # Wait

        return X, y
