"""
Neural Network Decision Engine
LSTM + Attention model for trade classification with probability output
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class PositionalEncoding(layers.Layer):
    """Positional encoding for attention mechanism"""

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def positional_encoding(self, position, d_model):
        position = tf.range(position, dtype=tf.float32)[tf.newaxis, :]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
        pos_encoding = tf.concat([
            tf.sin(position * div_term),
            tf.cos(position * div_term)
        ], axis=-1)
        return pos_encoding[tf.newaxis, :, :]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class AttentionBlock(layers.Layer):
    """Self-attention block for sequence processing"""

    def __init__(self, embedding_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(embedding_dim * 4, activation='relu'),
            layers.Dense(embedding_dim)
        ])

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, inputs)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)


class TradeDecisionModel:
    """
    LSTM + Attention model for trade classification.
    Outputs: Buy/Sell/Wait probabilities with calibration.
    """

    def __init__(
        self,
        input_dim: int = 60,
        sequence_length: int = 60,
        embedding_dim: int = 128,
        num_heads: int = 4,
        lstm_units: int = 64
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units

        self.model: Optional[keras.Model] = None
        self._build_model()

        # Calibration parameters (Platt scaling)
        self.calibration_params = {'scale': 1.0, 'shift': 0.0}
        self.training_history = []

    def _build_model(self):
        """Build the neural network architecture"""

        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim))

        # LSTM layers with dropout
        x = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )(inputs)
        x = layers.LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            dropout=0.2
        )(x)

        # Dense layer
        x = layers.Dense(self.embedding_dim, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Reshape for attention
        x = layers.RepeatVector(10)(x)
        x = AttentionBlock(self.embedding_dim, self.num_heads)(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)

        # Three outputs: Buy, Sell, Wait probabilities
        outputs = layers.Dense(3, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Model built with {self.model.count_params()} parameters")

    def prepare_features(
        self,
        df: pd.DataFrame,
        indicator_results: Dict
    ) -> np.ndarray:
        """
        Prepare feature matrix from price data and indicators.
        Returns sequence of shape (sequence_length, input_dim)
        """

        # Extract OHLCV features
        ohlcv_features = []

        if len(df) >= self.sequence_length:
            recent_df = df.tail(self.sequence_length)
        else:
            # Pad if not enough data
            padding = self.sequence_length - len(df)
            if len(df) > 0:
                first_row = df.iloc[[0]]
                recent_df = pd.concat([first_row] * padding + [df], ignore_index=True)
            else:
                return np.zeros((self.sequence_length, self.input_dim))

        # Normalize price data
        close_prices = recent_df['close'].values
        returns = recent_df['close'].pct_change().fillna(0).values
        high_low_range = (recent_df['high'] - recent_df['low']).values / close_prices
        volume_norm = recent_df['volume'] / recent_df['volume'].max()

        # Technical features
        features = np.column_stack([
            returns,
            high_low_range,
            volume_norm.values if hasattr(volume_norm, 'values') else volume_norm,
        ])

        # Add indicator values (broadcast across sequence)
        indicator_values = np.array([r.value for r in indicator_results.values()])
        indicator_signals = np.array([r.signal for r in indicator_results.values()])
        indicator_strengths = np.array([r.strength for r in indicator_results.values()])

        # Tile indicator data across sequence
        indicator_features = np.tile(
            np.concatenate([indicator_values, indicator_signals, indicator_strengths]),
            (self.sequence_length, 1)
        )

        # Combine all features
        all_features = np.hstack([features, indicator_features])

        # Pad or truncate to input_dim
        if all_features.shape[1] < self.input_dim:
            padding_cols = self.input_dim - all_features.shape[1]
            all_features = np.pad(
                all_features,
                ((0, 0), (0, padding_cols)),
                mode='constant'
            )
        elif all_features.shape[1] > self.input_dim:
            all_features = all_features[:, :self.input_dim]

        # Normalize
        all_features = (all_features - np.mean(all_features, axis=0)) / (np.std(all_features, axis=0) + 1e-8)

        return all_features.astype(np.float32)

    def predict(
        self,
        feature_sequence: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Make prediction for trade decision.

        Returns:
            - probabilities: [buy_prob, sell_prob, wait_prob]
            - win_probability: estimated probability of winning trade
            - expected_return: expected return in pips
        """

        if self.model is None:
            # Return default wait if model not loaded
            return np.array([0.1, 0.1, 0.8]), 0.45, 0.0

        # Ensure correct shape
        if len(feature_sequence.shape) == 2:
            feature_sequence = feature_sequence[np.newaxis, ...]

        # Get raw predictions
        raw_probs = self.model.predict(feature_sequence, verbose=0)[0]

        # Apply calibration
        calibrated_probs = self._calibrate_probs(raw_probs)

        # Calculate win probability (max of buy/sell, adjusted)
        action_probs = calibrated_probs[:2]  # Buy and Sell only
        max_action_prob = np.max(action_probs)
        win_probability = max_action_prob * 0.9 + 0.05  # Scale to 0.05-0.95

        # Calculate expected return based on confidence and historical performance
        confidence = max_action_prob - calibrated_probs[2]  # Action prob vs wait
        expected_return = confidence * 5.0  # Scale: ±5 pips typical

        return calibrated_probs, win_probability, expected_return

    def _calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration to probabilities"""
        # In production, this would use validation set statistics
        scale = self.calibration_params['scale']
        shift = self.calibration_params['shift']

        calibrated = probs * scale + shift
        calibrated = np.clip(calibrated, 0, 1)
        calibrated = calibrated / calibrated.sum()  # Renormalize

        return calibrated

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train the model on historical data.

        Args:
            X_train: Training sequences (n_samples, sequence_length, input_dim)
            y_train: Training labels (n_samples, 3) one-hot encoded
            X_val: Validation sequences
            y_val: Validation labels
        """

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.training_history = history.history

        # Update calibration based on validation performance
        self._update_calibration(X_val, y_val)

        return history

    def _update_calibration(self, X_val: np.ndarray, y_val: np.ndarray):
        """Update Platt scaling parameters based on validation set"""
        raw_preds = self.model.predict(X_val, verbose=0)

        # Simple calibration: match mean prediction to mean actual
        mean_pred = np.max(raw_preds[:, :2], axis=1)  # Max of buy/sell
        mean_actual = np.max(y_val[:, :2], axis=1)

        # Linear regression for calibration
        if np.std(mean_pred) > 0:
            self.calibration_params['scale'] = np.std(mean_actual) / np.std(mean_pred)
            self.calibration_params['shift'] = np.mean(mean_actual) - np.mean(mean_pred) * self.calibration_params['scale']

    def save(self, path: str):
        """Save model weights and calibration"""
        self.model.save_weights(path)
        calib_path = path.replace('.h5', '_calib.npy')
        np.save(calib_path, self.calibration_params)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights and calibration"""
        if os.path.exists(path):
            self.model.load_weights(path)
            calib_path = path.replace('.h5', '_calib.npy')
            if os.path.exists(calib_path):
                self.calibration_params = np.load(calib_path, allow_pickle=True).item()
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model file not found: {path}")


class DecisionEngine:
    """
    High-level decision engine that combines model predictions
    with risk management rules.
    """

    def __init__(self, model: TradeDecisionModel = None):
        self.model = model or TradeDecisionModel()
        self.win_rate_target = 0.65
        self.min_rr_target = 1.2

        # Performance tracking
        self.recent_predictions = []
        self.last_30_trades = []

    def make_decision(
        self,
        df: pd.DataFrame,
        indicator_results: Dict
    ) -> Dict:
        """
        Generate trading decision with full metadata.

        Returns dict with:
            - action: 'BUY', 'SELL', or 'WAIT'
            - probabilities: [buy, sell, wait]
            - win_probability: 0-1
            - expected_return_pips: float
            - expected_return_pct: float
            - confidence: 0-1
            - reasoning: list of indicator signals
        """

        # Prepare features
        features = self.model.prepare_features(df, indicator_results)

        # Get model prediction
        probs, win_prob, expected_pips = self.model.predict(features)

        # Determine action
        action_idx = np.argmax(probs)
        actions = ['BUY', 'SELL', 'WAIT']
        action = actions[action_idx]

        # Get signal summary for reasoning
        signal_summary = {
            k: v.signal for k, v in indicator_results.items()
            if abs(v.strength) > 0.5
        }

        # Adjust for risk management
        if action != 'WAIT':
            # Check if conditions meet minimum standards
            if win_prob < 0.55:  # Below threshold
                action = 'WAIT'
                probs[2] = 1 - probs[action_idx]
                probs[action_idx] = 0

        decision = {
            'action': action,
            'probabilities': probs.tolist(),
            'win_probability': float(win_prob),
            'expected_return_pips': float(expected_pips),
            'expected_return_pct': float(expected_pips * 0.06),  # ~0.06% per pip for EUR/JPY
            'confidence': float(probs[action_idx]),
            'reasoning': signal_summary,
            'timestamp': datetime.now().isoformat()
        }

        return decision

    def record_trade(self, trade_result: Dict):
        """Record trade outcome for performance tracking"""
        self.last_30_trades.append(trade_result)
        if len(self.last_30_trades) > 30:
            self.last_30_trades = self.last_30_trades[-30:]

        # Calculate rolling win rate
        wins = sum(1 for t in self.last_30_trades if t.get('pnl_pips', 0) > 0)
        win_rate = wins / len(self.last_30_trades) if self.last_30_trades else 0

        # Adjust model if win rate below target
        if win_rate < self.win_rate_target and len(self.last_30_trades) >= 10:
            logger.warning(
                f"Win rate {win_rate:.2%} below target {self.win_rate_target:.2%}"
            )

    def get_performance_stats(self) -> Dict:
        """Get recent performance statistics"""
        if not self.last_30_trades:
            return {'win_rate': 0, 'avg_pnl': 0, 'total_trades': 0}

        wins = sum(1 for t in self.last_30_trades if t.get('pnl_pips', 0) > 0)
        pnls = [t.get('pnl_pips', 0) for t in self.last_30_trades]

        return {
            'win_rate': wins / len(self.last_30_trades),
            'avg_pnl_pips': np.mean(pnls),
            'total_trades': len(self.last_30_trades),
            'best_trade': max(pnls),
            'worst_trade': min(pnls)
        }
