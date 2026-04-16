"""
Comprehensive Technical Indicator Engine
Calculates all required indicators for the trading system
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Container for indicator values"""
    name: str
    value: float
    signal: int  # -1: bearish, 0: neutral, 1: bullish
    strength: float  # 0-1 confidence


class IndicatorEngine:
    """
    Calculates comprehensive technical indicators across all families:
    - Trend: EMA, SuperTrend, ADX, ATR
    - Momentum: RSI, Stochastic, MACD, CCI
    - Volatility: Bollinger Bands, Keltner, WMA Range
    - Oscillators: VWMA Slope, Elder Ray, TEMA Divergence
    - Quantum Patterns: Volatility breakout, fractal patterns
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe.
        Expects columns: timestamp, open, high, low, close, volume
        """
        self.df = df.copy()
        self.close = df['close']
        self.high = df['high']
        self.low = df['low']
        self.open = df['open']
        self.volume = df['volume']

        self._results: Dict[str, IndicatorResult] = {}
        self._calculate_all()

    def _calculate_all(self):
        """Calculate all indicator families"""
        self._calculate_trend_indicators()
        self._calculate_momentum_indicators()
        self._calculate_volatility_indicators()
        self._calculate_oscillators()
        self._calculate_quantum_patterns()

    # ==================== TREND INDICATORS ====================

    def _calculate_trend_indicators(self):
        """Calculate trend-following indicators"""

        # EMA Cross (8/21)
        ema8 = ta.ema(self.close, length=8)
        ema21 = ta.ema(self.close, length=21)
        self.df['ema_8'] = ema8
        self.df['ema_21'] = ema21

        ema_cross = ema8.iloc[-1] - ema21.iloc[-1]
        self._results['ema_cross'] = IndicatorResult(
            name='EMA Cross',
            value=ema_cross,
            signal=1 if ema_cross > 0 else -1,
            strength=min(abs(ema_cross) / 0.01, 1.0)
        )

        # SuperTrend
        supertrend = ta.supertrend(self.high, self.low, self.close, length=10, multiplier=3)
        self.df['supertrend'] = supertrend['SUPERT_10_3.0']
        self.df['supertrend_dir'] = supertrend['SUPERTd_10_3.0']

        st_signal = int(supertrend['SUPERTd_10_3.0'].iloc[-1])
        self._results['supertrend'] = IndicatorResult(
            name='SuperTrend',
            value=supertrend['SUPERT_10_3.0'].iloc[-1],
            signal=1 if st_signal == 1 else -1,
            strength=0.8 if st_signal == 1 else 0.7
        )

        # ADX (Average Directional Index)
        adx = ta.adx(self.high, self.low, self.close, length=14)
        self.df['adx'] = adx['ADX_14']
        self.df['plus_di'] = adx['DMP_14']
        self.df['minus_di'] = adx['DMN_14']

        adx_val = adx['ADX_14'].iloc[-1]
        plus_di = adx['DMP_14'].iloc[-1]
        minus_di = adx['DMN_14'].iloc[-1]

        adx_signal = 0
        if adx_val > 25:  # Strong trend
            adx_signal = 1 if plus_di > minus_di else -1

        self._results['adx'] = IndicatorResult(
            name='ADX',
            value=adx_val,
            signal=adx_signal,
            strength=min(adx_val / 50, 1.0)
        )

        # ATR (Average True Range)
        atr = ta.atr(self.high, self.low, self.close, length=14)
        self.df['atr'] = atr
        self._results['atr'] = IndicatorResult(
            name='ATR',
            value=atr.iloc[-1],
            signal=0,  # ATR is magnitude, not directional
            strength=min(atr.iloc[-1] / 0.1, 1.0)
        )

    # ==================== MOMENTUM INDICATORS ====================

    def _calculate_momentum_indicators(self):
        """Calculate momentum-based indicators"""

        # RSI (Relative Strength Index)
        rsi = ta.rsi(self.close, length=14)
        self.df['rsi'] = rsi
        rsi_val = rsi.iloc[-1]
        rsi_signal = -1 if rsi_val > 70 else (1 if rsi_val < 30 else 0)
        self._results['rsi'] = IndicatorResult(
            name='RSI',
            value=rsi_val,
            signal=rsi_signal,
            strength=min(abs(rsi_val - 50) / 30, 1.0)
        )

        # Stochastic
        stoch = ta.stoch(self.high, self.low, self.close, k=14, d=3)
        self.df['stoch_k'] = stoch['STOCHk_14_3_3']
        self.df['stoch_d'] = stoch['STOCHd_14_3_3']

        stoch_k = stoch['STOCHk_14_3_3'].iloc[-1]
        stoch_d = stoch['STOCHd_14_3_3'].iloc[-1]
        stoch_signal = 1 if stoch_k < 20 and stoch_k > stoch_d else (-1 if stoch_k > 80 else 0)
        self._results['stochastic'] = IndicatorResult(
            name='Stochastic',
            value=stoch_k,
            signal=stoch_signal,
            strength=min(abs(stoch_k - 50) / 40, 1.0)
        )

        # MACD
        macd = ta.macd(self.close, fast=12, slow=26, signal=9)
        self.df['macd'] = macd['MACD_12_26_9']
        self.df['macd_signal'] = macd['MACDs_12_26_9']
        self.df['macd_hist'] = macd['MACDh_12_26_9']

        macd_val = macd['MACD_12_26_9'].iloc[-1]
        macd_sig = macd['MACDs_12_26_9'].iloc[-1]
        macd_signal = 1 if macd_val > macd_sig else -1
        self._results['macd'] = IndicatorResult(
            name='MACD',
            value=macd_val,
            signal=macd_signal,
            strength=min(abs(macd_val - macd_sig) / 0.01, 1.0)
        )

        # CCI (Commodity Channel Index)
        cci = ta.cci(self.high, self.low, self.close, length=20)
        self.df['cci'] = cci
        cci_val = cci.iloc[-1]
        cci_signal = -1 if cci_val > 100 else (1 if cci_val < -100 else 0)
        self._results['cci'] = IndicatorResult(
            name='CCI',
            value=cci_val,
            signal=cci_signal,
            strength=min(abs(cci_val) / 150, 1.0)
        )

        # 3-Point Wave
        self._calculate_wave_momentum()

    def _calculate_wave_momentum(self):
        """Calculate 3-point wave momentum"""
        if len(self.close) < 6:
            self.df['wave_momentum'] = 0
            self._results['wave'] = IndicatorResult('Wave', 0, 0, 0)
            return

        # Simple 3-point wave: compare current swing to previous
        swing_highs = self.high.rolling(window=3).apply(
            lambda x: 1 if x.iloc[1] == max(x) else 0
        )
        swing_lows = self.low.rolling(window=3).apply(
            lambda x: 1 if x.iloc[1] == min(x) else 0
        )

        wave_momentum = (self.close - self.close.shift(3)) / self.close.shift(3) * 100
        self.df['wave_momentum'] = wave_momentum

        wave_val = wave_momentum.iloc[-1]
        wave_signal = 1 if wave_val > 0.1 else (-1 if wave_val < -0.1 else 0)
        self._results['wave'] = IndicatorResult(
            name='3-Point Wave',
            value=wave_val,
            signal=wave_signal,
            strength=min(abs(wave_val) / 0.3, 1.0)
        )

    # ==================== VOLATILITY INDICATORS ====================

    def _calculate_volatility_indicators(self):
        """Calculate volatility-based indicators"""

        # Bollinger Bands
        bbands = ta.bbands(self.close, length=20, std=2)
        self.df['bb_upper'] = bbands['BBU_20_2.0']
        self.df['bb_middle'] = bbands['BBM_20_2.0']
        self.df['bb_lower'] = bbands['BBL_20_2.0']
        self.df['bb_pct'] = (self.close - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])

        bb_pct = self.df['bb_pct'].iloc[-1]
        bb_signal = -1 if bb_pct > 0.8 else (1 if bb_pct < 0.2 else 0)
        self._results['bollinger'] = IndicatorResult(
            name='Bollinger Bands',
            value=bb_pct,
            signal=bb_signal,
            strength=min(abs(bb_pct - 0.5) * 2, 1.0)
        )

        # Keltner Channel
        keltner = ta.kc(self.high, self.low, self.close, length=20, multiplier=1.5)
        self.df['kc_upper'] = keltner['KCU_20_1.5']
        self.df['kc_lower'] = keltner['KCL_20_1.5']

        kc_pct = (self.close.iloc[-1] - self.df['kc_lower'].iloc[-1]) / \
                 (self.df['kc_upper'].iloc[-1] - self.df['kc_lower'].iloc[-1])
        kc_signal = -1 if kc_pct > 0.8 else (1 if kc_pct < 0.2 else 0)
        self._results['keltner'] = IndicatorResult(
            name='Keltner',
            value=kc_pct,
            signal=kc_signal,
            strength=min(abs(kc_pct - 0.5) * 2, 1.0)
        )

        # WMA Range
        wma = ta.wma(self.close, length=14)
        self.df['wma'] = wma
        wma_range = (self.high - self.low) / self.close * 100
        self.df['wma_range'] = wma_range

        range_val = wma_range.iloc[-1]
        self._results['wma_range'] = IndicatorResult(
            name='WMA Range',
            value=range_val,
            signal=0,
            strength=min(range_val / 0.5, 1.0)
        )

    # ==================== OSCILLATORS ====================

    def _calculate_oscillators(self):
        """Calculate oscillator indicators"""

        # VWMA Slope
        vwma = ta.vwma(self.close, self.volume, length=14)
        self.df['vwma'] = vwma
        vwma_slope = (vwma.iloc[-1] - vwma.iloc[-5]) / 5
        self.df['vwma_slope'] = vwma_slope

        slope_signal = 1 if vwma_slope > 0.001 else (-1 if vwma_slope < -0.001 else 0)
        self._results['vwma_slope'] = IndicatorResult(
            name='VWMA Slope',
            value=vwma_slope,
            signal=slope_signal,
            strength=min(abs(vwma_slope) / 0.005, 1.0)
        )

        # Elder Ray (Bull/Bear Power)
        ema13 = ta.ema(self.close, length=13)
        bull_power = self.high - ema13
        bear_power = self.low - ema13
        self.df['bull_power'] = bull_power
        self.df['bear_power'] = bear_power

        bull = bull_power.iloc[-1]
        bear = bear_power.iloc[-1]
        elder_signal = 1 if bull > abs(bear) else (-1 if abs(bull) < bear else 0)
        self._results['elder_ray'] = IndicatorResult(
            name='Elder Ray',
            value=bull - bear,
            signal=elder_signal,
            strength=min(abs(bull - bear) / 0.02, 1.0)
        )

        # TEMA Divergence
        tema = ta.tema(self.close, length=9)
        self.df['tema'] = tema

        # Detect divergence (price vs TEMA)
        price_change = self.close.iloc[-1] - self.close.iloc[-5]
        tema_change = tema.iloc[-1] - tema.iloc[-5]

        divergence = 0
        if price_change > 0 and tema_change < 0:
            divergence = -1  # Bearish divergence
        elif price_change < 0 and tema_change > 0:
            divergence = 1  # Bullish divergence

        self._results['tema_div'] = IndicatorResult(
            name='TEMA Divergence',
            value=divergence,
            signal=divergence,
            strength=abs(divergence)
        )

        # 1-Minute Turtle (simplified breakout system)
        self._calculate_turtle()

    def _calculate_turtle(self):
        """Calculate 1-minute Turtle breakout system"""
        if len(self.close) < 20:
            self.df['turtle_signal'] = 0
            self._results['turtle'] = IndicatorResult('Turtle', 0, 0, 0)
            return

        # 20-period high/low breakout
        high_20 = self.high.rolling(window=20).max()
        low_20 = self.low.rolling(window=20).min()

        turtle_signal = 0
        if self.close.iloc[-1] > high_20.iloc[-1]:
            turtle_signal = 1  # Breakout long
        elif self.close.iloc[-1] < low_20.iloc[-1]:
            turtle_signal = -1  # Breakout short

        self.df['turtle_signal'] = turtle_signal
        self._results['turtle'] = IndicatorResult(
            name='1-Min Turtle',
            value=turtle_signal,
            signal=turtle_signal,
            strength=0.7 if turtle_signal != 0 else 0
        )

    # ==================== QUANTUM PATTERNS ====================

    def _calculate_quantum_patterns(self):
        """Calculate advanced binary-style patterns"""

        # 60-step Volatility Breakout
        if len(self.close) >= 60:
            recent_vol = self.close.tail(60).std()
            prev_vol = self.close.iloc[-121:-61].std() if len(self.close) >= 121 else recent_vol
            vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1

            breakout_signal = 1 if vol_ratio > 1.5 else (-1 if vol_ratio < 0.7 else 0)
            self._results['vol_breakout'] = IndicatorResult(
                name='Vol Breakout',
                value=vol_ratio,
                signal=breakout_signal,
                strength=min(abs(vol_ratio - 1) * 2, 1.0)
            )

        # Fractal Pattern Detection
        self._detect_fractals()

        # Multi-timeframe confluence (simulated)
        self._calculate_mtf_confluence()

    def _detect_fractals(self):
        """Detect fractal high/low patterns"""
        if len(self.high) < 5:
            self._results['fractal'] = IndicatorResult('Fractal', 0, 0, 0)
            return

        # Fractal high: high surrounded by two lower highs on each side
        # Fractal low: low surrounded by two higher lows on each side
        fractal_high = 0
        fractal_low = 0

        for i in range(2, min(5, len(self.high))):
            if self.high.iloc[-i] > self.high.iloc[-i-1] and \
               self.high.iloc[-i] > self.high.iloc[-i-2] and \
               self.high.iloc[-i] > self.high.iloc[-i+1] if i > 1 else True:
                fractal_high = 1
                break

        for i in range(2, min(5, len(self.low))):
            if self.low.iloc[-i] < self.low.iloc[-i-1] and \
               self.low.iloc[-i] < self.low.iloc[-i-2] and \
               self.low.iloc[-i] < self.low.iloc[-i+1] if i > 1 else True:
                fractal_low = 1
                break

        fractal_signal = fractal_high - fractal_low
        self._results['fractal'] = IndicatorResult(
            name='Fractal',
            value=fractal_signal,
            signal=fractal_signal,
            strength=0.6 if fractal_signal != 0 else 0
        )

    def _calculate_mtf_confluence(self):
        """Simulate multi-timeframe confluence"""
        # In production, this would aggregate M15, H1 data
        # Here we simulate with different lookback periods

        # Fast (M1 equivalent)
        fast_ema = ta.ema(self.close, length=8).iloc[-1]

        # Medium (M15 equivalent - downsampled)
        medium_ema = ta.ema(self.close, length=15 * 8).iloc[-1]

        # Slow (H1 equivalent - downsampled)
        slow_ema = ta.ema(self.close, length= 60 * 8).iloc[-1] if len(self.close) >= 480 else fast_ema

        # Confluence: all aligned
        bullish = fast_ema > medium_ema > slow_ema
        bearish = fast_ema < medium_ema < slow_ema

        mtf_signal = 1 if bullish else (-1 if bearish else 0)
        self._results['mtf'] = IndicatorResult(
            name='MTF Confluence',
            value=1.0 if mtf_signal != 0 else 0,
            signal=mtf_signal,
            strength=0.9 if mtf_signal != 0 else 0.3
        )

    # ==================== PUBLIC METHODS ====================

    def get_all_indicators(self) -> Dict[str, IndicatorResult]:
        """Get all calculated indicator results"""
        return self._results.copy()

    def get_signals_summary(self) -> Dict:
        """Get summary of all signals"""
        bullish = sum(1 for r in self._results.values() if r.signal == 1)
        bearish = sum(1 for r in self._results.values() if r.signal == -1)
        neutral = sum(1 for r in self._results.values() if r.signal == 0)

        total = bullish + bearish + neutral

        return {
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'bullish_pct': bullish / total if total > 0 else 0,
            'bearish_pct': bearish / total if total > 0 else 0,
            'net_signal': bullish - bearish
        }

    def get_feature_vector(self) -> np.ndarray:
        """Get numerical feature vector for ML model"""
        features = []
        for result in self._results.values():
            features.append(result.value)
            features.append(result.signal)
            features.append(result.strength)
        return np.array(features, dtype=np.float32)

    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe with all indicators"""
        return self.df.copy()
