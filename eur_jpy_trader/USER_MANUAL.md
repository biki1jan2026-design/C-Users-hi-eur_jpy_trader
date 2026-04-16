# EUR/JPY 1-Minute Live MAI-Trader - User Manual

## Overview

This is an autonomous AI-powered trading system for EUR/JPY on 1-minute timeframes. The system:

- Monitors real-time price action
- Analyzes 15+ technical indicators
- Makes Buy/Sell/Wait decisions using a neural network
- Manages risk with ATR-based stops
- Logs all trades and captures screenshots

---

## Getting Started

### First Launch

1. **Run the application**:
   ```bash
   python main.py gui
   ```

2. **Wait for initialization** (30-60 seconds):
   - Historical data loads
   - Neural network initializes
   - Charts render

3. **Verify the display**:
   - Top: Candlestick chart with volume
   - Bottom: Action buttons
   - Right panel: Account info

---

## GUI Layout

### Main Chart (Top Left)

```
┌─────────────────────────────────────────┐
│         EUR/JPY - 1 Minute              │
│  ╭───╮     ╭─╮                          │
│  │   │     │ │     ╭─╮                  │
│  │   ╰─────╯ │     │ │                  │
│  │           │     │ │                  │
├─────────────────────────────────────────┤
│  Volume                                 │
│  ███  ████  █   ██  ███                │
└─────────────────────────────────────────┘
```

- **Green candles**: Close > Open (bullish)
- **Red candles**: Close < Open (bearish)
- **Volume bars**: Trading volume at bottom

### Action Buttons (Bottom Center)

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│  BUY    │  │  SELL   │  │  WAIT   │
└─────────┘  └─────────┘  └─────────┘
```

- **Highlighted button** = AI recommendation
- **Green highlight** = Strong recommendation
- **Yellow highlight** = Wait recommended

### Probability Display (Bottom Left)

```
BUY:  ████████░░ 35.2%
SELL: ████░░░░░░ 18.5%
WAIT: ████████████ 46.3%

Expected: +2.4 pips
```

Shows:
- Probability distribution for each action
- Expected return in pips

### Account Panel (Right)

```
┌─────────────────────────┐
│ Account                 │
│ Balance: $50,234.56     │
│ Total P/L: +$234.56 (+0.47%) │
│ Daily P/L: +$123.45     │
├─────────────────────────┤
│ Current Position        │
│ BUY 1.2 lots @ 162.500  │
│ Stop: 162.400           │
│ PnL: +12.5 pips ($81.25)│
├─────────────────────────┤
│ Performance             │
│ Win Rate: 68.2%         │
│ Total Trades: 47        │
└─────────────────────────┘
```

### Trade History Tab

Shows all completed trades:

| ID | Time | Side | Entry | Exit | PnL | Win Prob |
|----|------|------|-------|------|-----|----------|
| TRD-000001 | 15:32 | BUY | 162.500 | 162.545 | +4.5 | 72% |

### Screenshot Tab

Lists captured screenshots from 3:30 PM - 5:30 PM sessions.

---

## Trading Logic

### How Decisions Are Made

The AI analyzes these indicator families:

| Category | Indicators |
|----------|------------|
| **Trend** | EMA Cross (8/21), SuperTrend, ADX, ATR |
| **Momentum** | RSI, Stochastic, MACD, CCI, 3-Point Wave |
| **Volatility** | Bollinger Bands, Keltner Channel, WMA Range |
| **Oscillators** | VWMA Slope, Elder Ray, TEMA Divergence, Turtle |
| **Patterns** | Volatility Breakout, Fractals, MTF Confluence |

### Decision Process

1. **Data Collection**: 60 candles of price + indicators
2. **Neural Network**: LSTM + Attention model processes sequence
3. **Output**: Probabilities for Buy/Sell/Wait
4. **Filter**: Win probability must exceed 55% threshold
5. **Execution**: Enter at candle close if conditions met

### Position Management

- **Entry**: At candle close when signal triggers
- **Stop Loss**: ATR × 2.0 from entry
- **Exit Conditions**:
  - Stop loss hit
  - 5-minute holding period elapsed

---

## Risk Management

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Balance | $50,000 | Demo starting balance |
| Risk per Trade | 2% | Maximum risk per trade |
| Max Daily Loss | 5% | Trading stops if hit |
| ATR Multiplier | 2.0× | Stop loss distance |
| Holding Period | 5 min | Time-based exit |

### Adjusting Risk Parameters

Edit `.env` file:

```bash
RISK_PER_TRADE=0.01    # Reduce to 1% per trade
MAX_DAILY_LOSS=0.03    # Stop at 3% daily loss
ATR_STOP_MULTIPLIER=1.5 # Tighter stops
```

### Position Size Calculation

```
Risk Amount = Balance × Risk per Trade
Stop Distance (pips) = ATR × Multiplier × 100
Position Size = Risk Amount / (Stop Distance × $6.50)
```

Example:
- Balance: $50,000
- Risk: 2% = $1,000
- ATR: 0.05, Multiplier: 2.0 → Stop: 10 pips
- Size: $1,000 / (10 × $6.50) = 15.4 lots

---

## Reading the Logs

### Trade Log (logs/YYYYMMDD_trades.csv)

```csv
trade_id,timestamp,side,entry_price,exit_price,pnl_pips,pnl_usd,size,win_probability
TRD-000001,2024-01-15T15:32:00,BUY,162.500,162.545,+4.5,+29.25,1.0,0.72
```

### Decision Log (logs/YYYYMMDD_decisions.json)

```json
{
  "timestamp": "2024-01-15T15:30:00",
  "action": "BUY",
  "probabilities": [0.45, 0.25, 0.30],
  "win_probability": 0.72,
  "reasoning": {
    "ema_cross": 0.005,
    "rsi": 35.2,
    "macd": 0.002
  }
}
```

---

## Screenshot Feature

### Automatic Capture

- **When**: 3:30 PM - 5:30 PM daily (London/NY overlap)
- **Frequency**: Every 60 seconds
- **Location**: `screenshots/YYYYMMDD/`

### File Naming

```
20240115_RY26_153045.png
│       │    │
│       │    └── Time (HHMMSS)
│       └── Session identifier
└── Date (YYYYMMDD)
```

### Manual Screenshot

In the GUI, screenshots are automatically logged. Files are saved to:
```
screenshots/YYYYMMDD/YYYYMMDD_RY26_HHMMSS.png
```

---

## Performance Metrics

### Understanding the Stats

| Metric | Target | Description |
|--------|--------|-------------|
| **Win Rate** | ≥65% | % of profitable trades |
| **Profit Factor** | ≥1.5 | Gross wins / Gross losses |
| **Sharpe Ratio** | ≥1.0 | Risk-adjusted return |
| **Max Drawdown** | <10% | Largest peak-to-trough decline |
| **Avg Win/Loss** | ≥1.2 | Reward-to-risk ratio |

### Improving Performance

1. **Let the AI run** - Avoid manual interference
2. **Review losing trades** - Check if patterns exist
3. **Adjust risk** - Lower risk if drawdown > 10%
4. **Retrain model** - Weekly retraining recommended

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `B` | Manual Buy |
| `S` | Manual Sell |
| `W` | Force Wait |
| `R` | Reset display |
| `Esc` | Close position |

---

## FAQ

**Q: Why is no trade being executed?**
A: The AI may be waiting for high-probability setups. This is normal - not every candle should produce a trade.

**Q: The balance changed but I didn't see a trade?**
A: Check the Trade History tab. Trades may close via time exit (5 minutes) without hitting stops.

**Q: How do I reset the demo balance?**
A: Edit `.env` and set `DEMO_BALANCE=50000`, then restart.

**Q: Can I trade other pairs?**
A: The system is optimized for EUR/JPY. Other pairs would require retraining and parameter adjustment.

**Q: Why are screenshots only between 3:30-5:30 PM?**
A: This is the London/NY overlap - highest liquidity and volatility period for EUR/JPY.

---

## Best Practices

1. **Start with demo** - Run at least 1 week before live trading
2. **Monitor daily** - Check logs and performance each session
3. **Don't override** - Trust the AI unless system malfunction
4. **Keep backups** - Save trade logs for tax/accounting
5. **Set alerts** - Configure notifications for drawdown limits

---

## Support Commands

### Check System Status
```bash
python -c "from backend.trading.trading_engine import TradingEngine; e = TradingEngine(); print(e.get_full_state())"
```

### Export Trade History
```bash
python -c "
from backend.utils.logger import TradeLogger
logger = TradeLogger()
df = logger.get_trade_history(days=30)
df.to_csv('trade_history_export.csv', index=False)
"
```

### View Performance Stats
```bash
python -c "
from backend.utils.logger import TradeLogger
logger = TradeLogger()
stats = logger.get_aggregate_stats(days=30)
for k, v in stats.items():
    print(f'{k}: {v}')
"
```
