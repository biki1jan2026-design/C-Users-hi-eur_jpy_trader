# EUR/JPY 1-Minute Live MAI-Trader - Deployment Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or higher
- **RAM**: 8 GB (16 GB recommended for model training)
- **Storage**: 5 GB free space
- **GPU**: Optional (NVIDIA CUDA for faster model training)

### For Docker Deployment
- Docker Desktop (Windows/macOS) or Docker Engine 20.10+ (Linux)
- Docker Compose 2.0+

---

## Quick Start

### Option 1: Direct Python Installation

```bash
# Clone or navigate to the project directory
cd eur_jpy_trader

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the GUI
python main.py gui

# Or run backtest first to validate installation
python main.py backtest
```

### Option 2: Docker Deployment

```bash
# Navigate to docker directory
cd docker

# Build and start containers
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
# Demo Trading Configuration
DEMO_BALANCE=50000          # Starting demo balance
RISK_PER_TRADE=0.02         # 2% risk per trade
MAX_DAILY_LOSS=0.05         # 5% max daily loss

# Trading Parameters
TIMEFRAME=1m                # 1-minute candles
SYMBOL=EUR/JPY
ATR_PERIOD=14
ATR_STOP_MULTIPLIER=2.0
HOLDING_PERIOD_MINUTES=5

# Screenshot Schedule (24-hour format)
SCREENSHOT_START_HOUR=15
SCREENSHOT_START_MINUTE=30
SCREENSHOT_END_HOUR=17
SCREENSHOT_END_MINUTE=30
```

### Model Configuration

The neural network model will be automatically initialized on first run. For custom training:

```bash
# Train model on historical data
python -c "
from backend.models.trainer import ModelTrainer
from backend.models.decision_engine import TradeDecisionModel
from backend.backtester import generate_synthetic_data

# Generate training data
df = generate_synthetic_data(days=30)

# Create and train model
model = TradeDecisionModel()
trainer = ModelTrainer(model)
results = trainer.train(df, epochs=50)

# Save model
model.save('backend/models/trained_model.h5')
"
```

---

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| trader-backend | 8541 | Trading engine with WebSocket API |
| trader-gui | 3000 | Web-based GUI (if using web frontend) |

### Docker Volumes

- `./logs` - Trade logs and daily reports
- `./data` - Historical data cache
- `./screenshots` - Captured chart screenshots

---

## File Structure

```
eur_jpy_trader/
├── backend/
│   ├── data/
│   │   └── data_feed.py        # Real-time data handling
│   ├── indicators/
│   │   └── indicator_engine.py # Technical indicators
│   ├── models/
│   │   ├── decision_engine.py  # Neural network model
│   │   └── trainer.py          # Model training
│   ├── trading/
│   │   ├── portfolio_manager.py # Risk management
│   │   └── trading_engine.py   # Main orchestrator
│   ├── utils/
│   │   └── logger.py           # Logging & screenshots
│   ├── backtester.py           # Backtesting engine
│   └── requirements.txt
├── frontend/
│   └── main_window.py          # PySide6 GUI
├── docker/
│   └── docker-compose.yml
├── logs/                        # Generated logs
├── data/                        # Generated data files
├── screenshots/                 # Captured screenshots
├── main.py                      # Entry point
├── test_trading_system.py       # Test suite
├── .env.example
└── DEPLOYMENT.md
```

---

## Running Components

### 1. GUI Application (Recommended for Desktop)

```bash
python main.py gui
```

This launches the full trading interface with:
- Real-time candlestick chart
- Action buttons (Buy/Sell/Wait)
- Probability display
- Account balance and P/L
- Trade history

### 2. Backend Only (Server Mode)

```bash
python main.py backend
```

Runs the trading engine without GUI. Suitable for:
- Headless servers
- Remote deployment
- API-only access

### 3. Backtesting Mode

```bash
python main.py backtest
```

Runs a 30-day simulation and outputs:
- Win rate statistics
- Total P/L
- Sharpe ratio
- Maximum drawdown

### 4. Test Suite

```bash
python test_trading_system.py
```

Validates all components before live deployment.

---

## Troubleshooting

### Common Issues

**1. TensorFlow Import Error**
```
pip install tensorflow --upgrade
# Or for CPU-only:
pip install tensorflow-cpu
```

**2. PySide6 Display Error (Linux)**
```bash
export QT_QPA_PLATFORM=xcb
# Or install platform libraries:
sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0
```

**3. Docker Permission Denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again
```

**4. Model Loading Error**
- Delete `backend/models/*.h5` files
- Restart the application (model will regenerate)

### Log Files

Check `logs/YYYYMMDD.log` for detailed runtime logs.

---

## Production Deployment

### For Live Trading

1. **Configure Broker API**
   - Add OANDA API credentials to `.env`
   - Test with demo account first

2. **Enable Model Training**
   - Run initial training on 30+ days of data
   - Validate win rate meets 65% target

3. **Set Up Monitoring**
   - Configure alerting for drawdown limits
   - Set up log aggregation

4. **Security Hardening**
   - Use secrets management for API keys
   - Enable TLS for network communication
   - Restrict network access to necessary ports

### Scaling Considerations

- **Single Instance**: Sufficient for EUR/JPY 1-minute trading
- **Multiple Pairs**: Deploy separate container per pair
- **High Frequency**: Consider C++ or Rust for indicator calculations

---

## Support

For issues or questions:
1. Check log files in `logs/`
2. Run test suite: `python test_trading_system.py`
3. Review indicator calculations with debug logging:
   ```bash
   LOG_LEVEL=DEBUG python main.py gui
   ```
