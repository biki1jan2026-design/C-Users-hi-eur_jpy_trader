#!/usr/bin/env python3
"""
EUR/JPY 1-Minute Live MAI-Trader
Main Entry Point

Run this file to start the trading system with GUI.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)


def run_gui():
    """Run the GUI application"""
    from frontend.main_window import main
    main()


def run_backend_only():
    """Run backend without GUI (for server deployment)"""
    from backend.trading.trading_engine import TradingEngine

    engine = TradingEngine(
        initial_balance=50000,
        risk_per_trade=0.02
    )

    async def run():
        await engine.start()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        engine.stop()
        logger.info("Backend stopped")


def run_backtest():
    """Run backtest"""
    from backend.backtester import run_backtest_demo
    run_backtest_demo()


def main():
    """Main entry point with command line arguments"""

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "gui":
            run_gui()
        elif command == "backend":
            run_backend_only()
        elif command == "backtest":
            run_backtest()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [gui|backend|backtest]")
            sys.exit(1)
    else:
        # Default: run GUI
        run_gui()


if __name__ == "__main__":
    main()
