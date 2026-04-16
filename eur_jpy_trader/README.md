# EUR/JPY 1-Minute Live MAI-Trader

> **Autonomous, Humanoid-AI Trading System for EUR/JPY**

A complete algorithmic trading system that monitors 1-minute EUR/JPY charts, analyzes multiple indicator families using a neural network decision engine, and executes trades with automated risk management.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Features

### 📊 Comprehensive Analysis
- **15+ Technical Indicators** across 5 families:
  - Trend: EMA Cross, SuperTrend, ADX, ATR
  - Momentum: RSI, Stochastic, MACD, CCI, 3-Point Wave
  - Volatility: Bollinger Bands, Keltner Channel, WMA Range
  - Oscillators: VWMA Slope, Elder Ray, TEMA Divergence, Turtle
  - Patterns: Volatility Breakout, Fractals, MTF Confluence

### 🧠 AI-Powered Decisions
- **LSTM + Attention Neural Network** for pattern recognition
- **Probability Output** with calibrated confidence scores
- **Expected Return** estimation in pips and percentage
- **65% Win Rate Target** with adaptive filtering

### ⚡ Real-Time Execution
- **1-Minute Candle** monitoring at every tick
- **Auto-Entry** at exact candle close
- **ATR Trailing Stop** for risk management
- **5-Minute Holding Period** with time-based exit

### 📈 Portfolio Management
- **Demo Balance** ($50,000 default)
- **2% Risk Per Trade** maximum
- **5% Daily Loss Limit** with auto-stop
- **Position Sizing** based on ATR volatility

### 📸 Audit & Compliance
- **CSV + JSON Logging** of every trade
- **Scheduled Screenshots** (3:30-5:30 PM daily)
- **Trade History** with win rate and P/L tracking
- **Performance Metrics** (Sharpe, Drawdown, Profit Factor)

---

## Quick Start

### Install Dependencies

```bash
cd eur_jpy_trader
pip install -r backend/requirements.txt
```

### Run the GUI

```bash
python main.py gui
```

### Run Backtest

```bash
python main.py backtest
```

### Run Tests

```bash
python test_trading_system.py
```

---

## Project Structure

```
eur_jpy_trader/
├── backend/
│   ├── data/
│   │   └── data_feed.py         # Real-time data streaming
│   ├── indicators/
│   │   └── indicator_engine.py  # 15+ technical indicators
│   ├── models/
│   │   ├── decision_engine.py   # Neural network model
│   │   └── trainer.py           # Model training pipeline
│   ├── trading/
│   │   ├── portfolio_manager.py # Risk management
│   │   └── trading_engine.py    # Main orchestrator
│   ├── utils/
│   │   └── logger.py            # Logging & screenshots
│   └── backtester.py            # Backtesting engine
├── frontend/
│   └── main_window.py           # PySide6 GUI
├── docker/
│   └── docker-compose.yml       # Docker deployment
├── main.py                      # Entry point
├── test_trading_system.py       # Test suite
├── DEPLOYMENT.md                # Deployment guide
└── USER_MANUAL.md               # User documentation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GUI (PySide6)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Candle Chart│  │ Action Btns │  │ Balance / History   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Trading Engine                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Data Feed    │  │ Decision     │  │ Portfolio        │   │
│  │ (1-min)      │  │ Engine (NN)  │  │ Manager          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                  Indicator Engine                           │
│  EMA | SuperTrend | RSI | MACD | BBands | Keltner | etc.   │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

```
Input: (60 timesteps × 60 features)
    ↓
LSTM (64 units, dropout 0.2)
    ↓
LSTM (32 units, dropout 0.2)
    ↓
Dense (128, ReLU)
    ↓
Attention Block (4 heads)
    ↓
Global Average Pooling
    ↓
Dense (64, ReLU) → Dense (32, ReLU)
    ↓
Output: [Buy_prob, Sell_prob, Wait_prob] (Softmax)
```

---

## Configuration

Edit `.env` file:

```bash
# Account
DEMO_BALANCE=50000
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05

# Trading
ATR_STOP_MULTIPLIER=2.0
HOLDING_PERIOD_MINUTES=5

# Screenshot Window
SCREENSHOT_START_HOUR=15
SCREENSHOT_END_HOUR=17
```

---

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Win Rate | ≥65% | Percentage of winning trades |
| Profit Factor | ≥1.5 | Gross wins ÷ Gross losses |
| Sharpe Ratio | ≥1.0 | Risk-adjusted returns |
| Max Drawdown | <10% | Maximum peak-to-trough loss |
| Avg R:R | ≥1.2 | Average reward-to-risk ratio |

---

## Docker Deployment

```bash
cd docker
docker-compose up -d --build
```

View logs:
```bash
docker-compose logs -f
```

---

## API Usage

### Programmatic Access

```python
from backend.trading.trading_engine import TradingEngine

engine = TradingEngine(
    initial_balance=50000,
    risk_per_trade=0.02
)

# Get current state
state = engine.get_full_state()
print(f"Balance: ${state['balance']:,.2f}")
print(f"P/L: ${state['total_pnl']:,.2f}")

# Get trade history
history = engine.get_trade_history()
```

### Backtesting

```python
from backend.backtester import Backtester, generate_synthetic_data

# Generate test data
df = generate_synthetic_data(days=30)

# Run backtest
backtester = Backtester()
results = backtester.run_backtest(df)

print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Total P/L: ${results['total_pnl_usd']:,.2f}")
```

---

## Testing

Run the full test suite:

```bash
python test_trading_system.py
```

Tests cover:
- Data feed and candle management
- Indicator calculations
- Portfolio management
- Decision engine predictions
- Backtesting logic

---

## Disclaimer

**This software is for educational and research purposes only.**

- Trading foreign exchange carries substantial risk
- Past performance does not guarantee future results
- Always test thoroughly before considering live deployment
- Never trade with money you cannot afford to lose

---

## License

MIT License - See LICENSE file for details

---

## Documentation

- [Deployment Guide](DEPLOYMENT.md) - Installation and configuration
- [User Manual](USER_MANUAL.md) - GUI usage and trading logic

---

## Version

1.0.0 - Initial Release
