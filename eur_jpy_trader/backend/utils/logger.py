"""
Logging and Data Persistence Module
Handles CSV/JSON logging and screenshot management
"""

import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs trades to CSV and JSON formats.
    Manages daily folders and screenshot tracking.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        data_dir: str = "./data",
        screenshot_dir: str = "./screenshots"
    ):
        self.log_dir = Path(log_dir)
        self.data_dir = Path(data_dir)
        self.screenshot_dir = Path(screenshot_dir)

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Current log files
        self._current_csv: Optional[Path] = None
        self._current_json: Optional[Path] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._json_file: Optional[Any] = None

        self._initialize_daily_files()

    def _initialize_daily_files(self):
        """Initialize log files for current day"""
        date_str = datetime.now().strftime("%Y%m%d")

        self._current_csv = self.log_dir / f"{date_str}_trades.csv"
        self._current_json = self.log_dir / f"{date_str}_trades.json"

        # Initialize CSV with headers if new file
        initialize_csv = not self._current_csv.exists()

        if initialize_csv:
            fieldnames = [
                'trade_id', 'timestamp', 'side', 'entry_price', 'exit_price',
                'pnl_pips', 'pnl_usd', 'size', 'win_probability',
                'expected_return', 'exit_reason', 'indicators'
            ]
            with open(self._current_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                self._csv_writer = writer

        logger.info(f"Daily log files initialized: {self._current_csv}, {self._current_json}")

    def _check_new_day(self):
        """Check if we need to create new daily files"""
        date_str = datetime.now().strftime("%Y%m%d")
        if self._current_csv and date_str not in str(self._current_csv):
            self._initialize_daily_files()

    def log_trade(self, trade_data: Dict):
        """Log a completed trade"""
        self._check_new_day()

        # Prepare row
        row = {
            'trade_id': trade_data.get('id', ''),
            'timestamp': trade_data.get('exit_time', datetime.now().isoformat()),
            'side': trade_data.get('side', ''),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price', 0),
            'pnl_pips': trade_data.get('pnl_pips', 0),
            'pnl_usd': trade_data.get('pnl_usd', 0),
            'size': trade_data.get('size', 0),
            'win_probability': trade_data.get('decision_probability', 0),
            'expected_return': trade_data.get('expected_return', 0),
            'exit_reason': trade_data.get('status', ''),
            'indicators': json.dumps(trade_data.get('trigger_indicators', {}))
        }

        # Write to CSV
        with open(self._current_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        # Append to JSON
        json_path = self._current_json
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = []

        data.append(row)

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Trade logged: {row['trade_id']}, PnL={row['pnl_pips']:.2f} pips")

    def log_decision(self, decision_data: Dict):
        """Log a decision (even if no trade was taken)"""
        self._check_new_day()

        json_path = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}_decisions.json"

        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = []

        data.append({
            'timestamp': datetime.now().isoformat(),
            'action': decision_data.get('action', 'WAIT'),
            'probabilities': decision_data.get('probabilities', []),
            'win_probability': decision_data.get('win_probability', 0),
            'expected_return_pips': decision_data.get('expected_return_pips', 0),
            'reasoning': decision_data.get('reasoning', {})
        })

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_daily_trades(self, date_str: str = None) -> List[Dict]:
        """Get trades for a specific date"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        csv_path = self.log_dir / f"{date_str}_trades.csv"
        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        return df.to_dict('records')

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history for last N days"""
        all_trades = []

        for i in range(days):
            date = datetime.now() - pd.Timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            csv_path = self.log_dir / f"{date_str}_trades.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_trades.append(df)

        if all_trades:
            return pd.concat(all_trades, ignore_index=True)
        return pd.DataFrame()

    def get_screenshot_log(self, date_str: str = None) -> List[str]:
        """Get list of screenshots for a date"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        log_file = self.screenshot_dir / f"{date_str}_screenshots.log"
        if not log_file.exists():
            return []

        with open(log_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def log_screenshot(self, filename: str):
        """Log a screenshot capture"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.screenshot_dir / f"{date_str}_screenshots.log"

        with open(log_file, 'a') as f:
            f.write(f"{filename}\n")

        logger.info(f"Screenshot logged: {filename}")

    def get_aggregate_stats(self, days: int = 30) -> Dict:
        """Get aggregate trading statistics"""
        df = self.get_trade_history(days)

        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl_pips': 0,
                'total_pnl_usd': 0,
                'avg_win_pips': 0,
                'avg_loss_pips': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        wins = df[df['pnl_pips'] > 0]
        losses = df[df['pnl_pips'] <= 0]

        return {
            'total_trades': len(df),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'total_pnl_pips': df['pnl_pips'].sum(),
            'total_pnl_usd': df['pnl_usd'].sum(),
            'avg_win_pips': wins['pnl_pips'].mean() if len(wins) > 0 else 0,
            'avg_loss_pips': losses['pnl_pips'].mean() if len(losses) > 0 else 0,
            'best_trade': df['pnl_pips'].max(),
            'worst_trade': df['pnl_pips'].min()
        }


class ScreenshotManager:
    """
    Manages screenshot capture and organization.
    Captures between 3:30 PM and 5:30 PM daily.
    """

    def __init__(
        self,
        screenshot_dir: str = "./screenshots",
        start_hour: int = 15,
        start_minute: int = 30,
        end_hour: int = 17,
        end_minute: int = 30
    ):
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.start_hour = start_hour
        self.start_minute = start_minute
        self.end_hour = end_hour
        self.end_minute = end_minute

        self._capture_interval = 60  # Capture every 60 seconds during window
        self._last_capture: Optional[datetime] = None

    def is_in_capture_window(self, current_time: datetime = None) -> bool:
        """Check if current time is within screenshot window"""
        if current_time is None:
            current_time = datetime.now()

        start = current_time.replace(
            hour=self.start_hour,
            minute=self.start_minute,
            second=0,
            microsecond=0
        )
        end = current_time.replace(
            hour=self.end_hour,
            minute=self.end_minute,
            second=0,
            microsecond=0
        )

        return start <= current_time <= end

    def should_capture(self, current_time: datetime = None) -> bool:
        """Check if we should capture a screenshot now"""
        if not self.is_in_capture_window(current_time):
            return False

        if current_time is None:
            current_time = datetime.now()

        if self._last_capture is None:
            return True

        elapsed = (current_time - self._last_capture).total_seconds()
        return elapsed >= self._capture_interval

    def generate_filename(self) -> str:
        """Generate screenshot filename"""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        return f"{date_str}_RY26_{time_str}.png"

    def get_daily_folder(self) -> Path:
        """Get or create daily folder"""
        date_str = datetime.now().strftime("%Y%m%d")
        folder = self.screenshot_dir / date_str
        folder.mkdir(exist_ok=True)
        return folder

    def save_screenshot(self, image_data: bytes, filename: str = None) -> str:
        """Save screenshot to daily folder"""
        if filename is None:
            filename = self.generate_filename()

        folder = self.get_daily_folder()
        filepath = folder / filename

        with open(filepath, 'wb') as f:
            f.write(image_data)

        self._last_capture = datetime.now()

        # Log the screenshot
        trade_logger = TradeLogger(screenshot_dir=str(self.screenshot_dir))
        trade_logger.log_screenshot(str(filepath))

        logger.info(f"Screenshot saved: {filepath}")
        return str(filepath)

    def get_screenshots(self, date_str: str = None) -> List[str]:
        """Get list of screenshots for a date"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        folder = self.screenshot_dir / date_str
        if not folder.exists():
            return []

        return sorted([f.name for f in folder.glob("*.png")])

    def get_all_screenshots(self, days: int = 7) -> Dict[str, List[str]]:
        """Get screenshots grouped by date"""
        result = {}

        for i in range(days):
            date = datetime.now() - pd.Timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            screenshots = self.get_screenshots(date_str)

            if screenshots:
                result[date_str] = screenshots

        return result
