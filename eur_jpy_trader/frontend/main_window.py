"""
Main GUI Window for EUR/JPY Trading System
PySide6-based interface with real-time charting
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QTabWidget, QTextEdit, QProgressBar, QGroupBox, QSplitter,
    QHeaderView, QTableWidget, QTableWidgetItem, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen

import pyqtgraph as pg
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.data.data_feed import DataFeed, Candle
from backend.indicators.indicator_engine import IndicatorEngine
from backend.trading.portfolio_manager import PortfolioManager


class CandlestickWidget(pg.GraphicsLayoutWidget):
    """
    Custom candlestick chart widget with volume
    """

    def __init__(self):
        super().__init__()
        self.candles = []
        self.setup_chart()

    def setup_chart(self):
        """Initialize the candlestick chart"""
        # Main price chart
        self.price_plot = self.addPlot(row=0, col=0)
        self.price_plot.setTitle("EUR/JPY - 1 Minute", color='w', size='14pt')
        self.price_plot.setLabel('left', 'Price', color='w')
        self.price_plot.setLabel('bottom', 'Time', color='w')
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setBackground('k')

        # Volume chart
        self.volume_plot = self.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', 'Volume', color='w')
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setBackground('k')
        self.volume_plot.setXLink(self.price_plot)

        # Style
        self.price_plot.getAxis('left').setPen('w')
        self.price_plot.getAxis('bottom').setPen('w')
        self.volume_plot.getAxis('left').setPen('w')
        self.volume_plot.getAxis('bottom').setPen('w')

        # Candle data storage
        self.candle_bars = []
        self.volume_bars = []

    def update_candles(self, candles: List[Dict]):
        """Update chart with new candle data"""
        self.candles = candles

        if not candles:
            return

        # Convert to arrays for plotting
        n = len(candles)
        x = np.arange(n)

        opens = np.array([c['open'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])

        # Determine colors
        colors_up = np.where(closes >= opens, 1, 0)
        colors = ['#00ff00' if c >= o else '#ff0000' for o, c in zip(opens, closes)]

        # Clear and redraw
        self.price_plot.clear()
        self.volume_plot.clear()

        # Draw candlesticks
        for i in range(n):
            # Wick
            wick = pg.PlotCurveItem(
                x=[i, i],
                y=[lows[i], highs[i]],
                pen=pg.mkPen(colors[i], width=1)
            )
            self.price_plot.addItem(wick)

            # Body
            open_price = opens[i]
            close_price = closes[i]
            height = close_price - open_price

            if abs(height) < 0.001:
                height = 0.001 if close_price >= open_price else -0.001

            rect = pg.BarGraphItem(
                x=i,
                height=height,
                width=0.8,
                y=open_price,
                brush=colors[i]
            )
            self.price_plot.addItem(rect)

        # Volume bars
        vol_colors = ['#00aa00' if c >= o else '#aa0000' for o, c in zip(opens, closes)]
        vol_bars = pg.BarGraphItem(
            x=x,
            height=volumes,
            width=0.8,
            brushes=[pg.mkBrush(c) for c in vol_colors]
        )
        self.volume_plot.addItem(vol_bars)

        # Auto-range
        self.price_plot.setYRange(lows.min() - 0.05, highs.max() + 0.05)
        self.volume_plot.setYRange(0, volumes.max() * 1.5)


class ActionButton(QPushButton):
    """
    Action button that highlights when recommended
    """

    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        self.setFixedHeight(60)
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.normal_style = """
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """
        self.highlighted_style = """
            QPushButton {
                background-color: #00aa00;
                color: #ffffff;
                border: 3px solid #00ff00;
                border-radius: 8px;
                font-weight: bold;
            }
        """
        self.wait_highlighted = """
            QPushButton {
                background-color: #666600;
                color: #ffffff;
                border: 3px solid #ffff00;
                border-radius: 8px;
            }
        """
        self.setStyleSheet(self.normal_style)
        self.is_highlighted = False

    def highlight(self, is_recommended: bool, action_type: str = 'action'):
        """Highlight button if recommended"""
        if is_recommended:
            if action_type == 'wait':
                self.setStyleSheet(self.wait_highlighted)
            else:
                self.setStyleSheet(self.highlighted_style)
            self.is_highlighted = True
        else:
            self.setStyleSheet(self.normal_style)
            self.is_highlighted = False


class ProbabilityBar(QWidget):
    """
    Custom probability bar widget
    """

    def __init__(self):
        super().__init__()
        self.buy_prob = 0.0
        self.sell_prob = 0.0
        self.wait_prob = 0.0
        self.expected_return = 0.0
        self.setFixedHeight(80)

    def update_probs(self, probs: List[float], expected_return: float):
        """Update probability display"""
        if len(probs) >= 3:
            self.buy_prob = probs[0]
            self.sell_prob = probs[1]
            self.wait_prob = probs[2]
        self.expected_return = expected_return
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw probability bars
        bar_height = 30
        y_offset = 10

        # Buy bar (green)
        buy_width = int(width * self.buy_prob)
        painter.setBrush(QBrush(QColor('#00aa00')))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, y_offset, buy_width, bar_height)

        # Sell bar (red)
        painter.setBrush(QBrush(QColor('#aa0000')))
        sell_width = int(width * self.sell_prob)
        painter.drawRect(0, y_offset + bar_height + 5, sell_width, bar_height)

        # Wait bar (yellow)
        painter.setBrush(QBrush(QColor('#aaaa00')))
        wait_width = int(width * self.wait_prob)
        painter.drawRect(0, y_offset + (bar_height + 5) * 2, wait_width, bar_height)

        # Labels
        painter.setPen(QColor('white'))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(5, y_offset + 20, f"BUY: {self.buy_prob:.1%}")
        painter.drawText(5, y_offset + bar_height + 25, f"SELL: {self.sell_prob:.1%}")
        painter.drawText(5, y_offset + (bar_height + 5) * 2 + 20, f"WAIT: {self.wait_prob:.1%}")

        # Expected return
        painter.setPen(QColor('#00ffff'))
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        return_text = f"Expected: {self.expected_return:+.2f} pips"
        painter.drawText(5, height - 10, return_text)


class TradeHistoryTable(QTableWidget):
    """
    Table widget for trade history display
    """

    def __init__(self):
        super().__init__()
        self.setup_table()

    def setup_table(self):
        """Configure the table"""
        headers = [
            'ID', 'Time', 'Side', 'Entry', 'Exit',
            'PnL (pips)', 'PnL ($)', 'Win Prob', 'Status'
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)

        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a2e;
                color: #ffffff;
                gridline-color: #333333;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2a2a4e;
                color: #ffffff;
                padding: 5px;
                border: none;
            }
        """)

    def update_trades(self, trades: List[Dict]):
        """Update table with trade data"""
        self.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            self.setItem(i, 0, QTableWidgetItem(trade.get('id', 'N/A')))

            exit_time = trade.get('exit_time', '')
            if exit_time:
                try:
                    dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    exit_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            self.setItem(i, 1, QTableWidgetItem(exit_time))

            side = trade.get('side', '')
            side_item = QTableWidgetItem(side)
            side_item.setForeground(QColor('#00ff00' if side == 'BUY' else '#ff0000'))
            self.setItem(i, 2, side_item)

            self.setItem(i, 3, QTableWidgetItem(f"{trade.get('entry_price', 0):.3f}"))
            self.setItem(i, 4, QTableWidgetItem(f"{trade.get('exit_price', 0):.3f}"))

            pnl_pips = trade.get('pnl_pips', 0)
            pnl_item = QTableWidgetItem(f"{pnl_pips:+.2f}")
            pnl_item.setForeground(QColor('#00ff00' if pnl_pips > 0 else '#ff0000'))
            self.setItem(i, 5, pnl_item)

            pnl_usd = trade.get('pnl_usd', 0)
            self.setItem(i, 6, QTableWidgetItem(f"${pnl_usd:+.2f}"))

            self.setItem(i, 7, QTableWidgetItem(f"{trade.get('decision_probability', 0):.1%}"))

            status = trade.get('status', '')
            self.setItem(i, 8, QTableWidgetItem(status))


class MainWindow(QMainWindow):
    """
    Main application window
    """

    def __init__(self):
        super().__init__()

        # Trading components
        self.data_feed = DataFeed()
        self.portfolio = PortfolioManager(initial_balance=50000.0)

        # Setup UI
        self.setup_ui()

        # Timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second

        # Simulated data
        self.simulated_candles = []
        self.current_price = 162.500

        # Start simulation
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulate_tick)
        self.sim_timer.start(500)

    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("EUR/JPY 1-Minute Live MAI-Trader")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a1a;
            }
        """)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel (chart + controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_layout) if False else QVBoxLayout(left_panel)

        # Chart
        self.candle_chart = CandlestickWidget()
        self.candle_chart.setMinimumHeight(400)
        left_layout.addWidget(self.candle_chart)

        # Action buttons
        action_frame = QFrame()
        action_frame.setFrameStyle(QFrame.StyledPanel)
        action_frame.setStyleSheet("QFrame { background-color: #1a1a2e; border-radius: 10px; }")
        action_layout = QHBoxLayout(action_frame)

        self.buy_btn = ActionButton("BUY")
        self.buy_btn.clicked.connect(lambda: self.manual_action('BUY'))
        action_layout.addWidget(self.buy_btn)

        self.sell_btn = ActionButton("SELL")
        self.sell_btn.clicked.connect(lambda: self.manual_action('SELL'))
        action_layout.addWidget(self.sell_btn)

        self.wait_btn = ActionButton("WAIT")
        self.wait_btn.clicked.connect(lambda: self.manual_action('WAIT'))
        action_layout.addWidget(self.wait_btn)

        left_layout.addWidget(action_frame)

        # Probability display
        self.prob_bar = ProbabilityBar()
        left_layout.addWidget(self.prob_bar)

        splitter.addWidget(left_panel)

        # Right panel (info + history)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tabs
        tabs = QTabWidget()

        # Status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        # Balance display
        balance_group = QGroupBox("Account")
        balance_layout = QVBoxLayout(balance_group)

        self.balance_label = QLabel("Balance: $50,000.00")
        self.balance_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.balance_label.setStyleSheet("color: #00ff00;")
        balance_layout.addWidget(self.balance_label)

        self.pnl_label = QLabel("Total P/L: $0.00 (0.00%)")
        self.pnl_label.setFont(QFont("Arial", 14))
        self.pnl_label.setStyleSheet("color: #ffffff;")
        balance_layout.addWidget(self.pnl_label)

        self.daily_pnl_label = QLabel("Daily P/L: $0.00")
        self.daily_pnl_label.setFont(QFont("Arial", 12))
        balance_layout.addWidget(self.daily_pnl_label)

        status_layout.addWidget(balance_group)

        # Position display
        position_group = QGroupBox("Current Position")
        position_layout = QVBoxLayout(position_group)

        self.position_label = QLabel("No open position")
        self.position_label.setFont(QFont("Arial", 12))
        position_layout.addWidget(self.position_label)

        self.risk_label = QLabel("Risk/Trade: 2.00%")
        position_layout.addWidget(self.risk_label)

        status_layout.addWidget(position_group)

        # Performance stats
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout(perf_group)

        self.winrate_label = QLabel("Win Rate: N/A")
        perf_layout.addWidget(self.winrate_label)

        self.total_trades_label = QLabel("Total Trades: 0")
        perf_layout.addWidget(self.total_trades_label)

        status_layout.addWidget(perf_group)

        tabs.addTab(status_tab, "Status")

        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        self.trade_history = TradeHistoryTable()
        history_layout.addWidget(self.trade_history)

        tabs.addTab(history_tab, "Trade History")

        # Screenshots tab
        screenshots_tab = QWidget()
        ss_layout = QVBoxLayout(screenshots_tab)

        self.ss_log = QTextEdit()
        self.ss_log.setReadOnly(True)
        self.ss_log.setPlaceholderText("Screenshots captured between 3:30 PM - 5:30 PM will appear here")
        ss_layout.addWidget(self.ss_log)

        tabs.addTab(screenshots_tab, "Screenshots")

        right_layout.addWidget(tabs)

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def simulate_tick(self):
        """Simulate price movement for demo"""
        # Random walk
        change = np.random.normal(0, 0.003)
        self.current_price += change

        # Create or update current candle
        now = datetime.now()

        if not self.simulated_candles or \
           self.simulated_candles[-1]['minute'] != now.minute:
            # New candle
            self.simulated_candles.append({
                'timestamp': now.isoformat(),
                'open': self.current_price,
                'high': self.current_price,
                'low': self.current_price,
                'close': self.current_price,
                'volume': 0,
                'minute': now.minute
            })
        else:
            # Update current candle
            candle = self.simulated_candles[-1]
            candle['close'] = self.current_price
            candle['high'] = max(candle['high'], self.current_price)
            candle['low'] = min(candle['low'], self.current_price)
            candle['volume'] += np.random.uniform(1, 10)

        # Keep last 50 candles
        if len(self.simulated_candles) > 50:
            self.simulated_candles = self.simulated_candles[-50:]

    @Slot()
    def update_display(self):
        """Update all display elements"""
        # Update chart
        self.candle_chart.update_candles(self.simulated_candles)

        # Simulate AI decision
        if self.simulated_candles:
            # Generate fake probabilities for demo
            buy_prob = np.random.uniform(0.1, 0.4)
            sell_prob = np.random.uniform(0.1, 0.4)
            wait_prob = 1 - buy_prob - sell_prob

            probs = [buy_prob, sell_prob, wait_prob]
            expected = np.random.uniform(-3, 3)

            self.prob_bar.update_probs(probs, expected)

            # Highlight recommended action
            max_idx = np.argmax(probs)
            actions = ['buy', 'sell', 'wait']
            self.buy_btn.highlight(max_idx == 0, actions[max_idx])
            self.sell_btn.highlight(max_idx == 1, actions[max_idx])
            self.wait_btn.highlight(max_idx == 2, actions[max_idx])

        # Update balance (simulate small changes)
        current_balance = 50000 + np.random.uniform(-100, 100)
        self.balance_label.setText(f"Balance: ${current_balance:,.2f}")

        pnl = current_balance - 50000
        pnl_pct = pnl / 50000 * 100
        pnl_color = '#00ff00' if pnl >= 0 else '#ff0000'
        self.pnl_label.setText(f"Total P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.pnl_label.setStyleSheet(f"color: {pnl_color};")

        self.daily_pnl_label.setText(f"Daily P/L: ${pnl:+.2f}")

        # Update position info
        self.winrate_label.setText(f"Win Rate: {np.random.uniform(0.55, 0.75):.1%}")
        self.total_trades_label.setText(f"Total Trades: {np.random.randint(5, 50)}")

    def manual_action(self, action: str):
        """Handle manual action button click"""
        print(f"Manual action: {action}")
        # In production, this would trigger actual trades


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set dark palette
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QColor('#0a0a1a'))
    palette.setColor(palette.WindowText, QColor('#ffffff'))
    palette.setColor(palette.Base, QColor('#1a1a2e'))
    palette.setColor(palette.AlternateBase, QColor('#2a2a4e'))
    palette.setColor(palette.Text, QColor('#ffffff'))
    palette.setColor(palette.Button, QColor('#1a1a2e'))
    palette.setColor(palette.ButtonText, QColor('#ffffff'))
    palette.setColor(palette.Highlight, QColor('#0066cc'))
    palette.setColor(palette.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
