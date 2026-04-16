"""
Portfolio and Risk Management Module
Handles position management, balance tracking, and risk controls
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"


@dataclass
class Trade:
    """Represents a single trade position"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    size: float  # Position size in lots
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr_stop: Optional[float] = None

    # Closed trade data
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_pips: Optional[float] = None
    pnl_usd: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN

    # Metadata
    trigger_indicators: Dict = field(default_factory=dict)
    decision_probability: float = 0.0
    expected_return: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl_pips': self.pnl_pips,
            'pnl_usd': self.pnl_usd,
            'status': self.status.value,
            'trigger_indicators': self.trigger_indicators,
            'decision_probability': self.decision_probability,
            'expected_return': self.expected_return
        }


class PortfolioManager:
    """
    Manages demo balance, positions, and risk controls.
    """

    def __init__(
        self,
        initial_balance: float = 50000.0,
        risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
        atr_multiplier: float = 2.0
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.atr_multiplier = atr_multiplier

        # Position tracking
        self.open_positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self._trade_counter = 0

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._last_reset_date = datetime.now().date()

        # EUR/JPY pip value (~$6.50 per standard lot per pip)
        self.pip_value_per_lot = 6.50

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        atr: float = 0.05
    ) -> float:
        """
        Calculate position size based on risk parameters.
        Max 2% of balance per trade.
        """
        if stop_loss_price is None:
            # Use ATR-based stop if not provided
            stop_distance = atr * self.atr_multiplier
        else:
            stop_distance = abs(entry_price - stop_loss_price)

        # Risk amount in USD
        risk_amount = self.balance * self.risk_per_trade

        # Convert stop distance to pips
        stop_pips = stop_distance * 100  # JPY pairs: 0.01 = 1 pip

        # Calculate lot size
        if stop_pips > 0:
            lot_size = risk_amount / (stop_pips * self.pip_value_per_lot)
        else:
            lot_size = 0

        # Cap at reasonable size
        max_lot = self.balance * 0.1 / (100 * self.pip_value_per_lot)  # Max 10% in margin
        lot_size = min(lot_size, max_lot)

        return round(lot_size, 2)

    def open_position(
        self,
        side: str,
        entry_price: float,
        atr: float,
        trigger_indicators: Dict = None,
        decision_probability: float = 0.0,
        expected_return: float = 0.0
    ) -> Optional[Trade]:
        """
        Open a new trading position.
        """

        # Check daily loss limit
        if self.daily_pnl < -self.balance * self.max_daily_loss:
            logger.warning("Daily loss limit reached - cannot open new position")
            return None

        # Check for existing position (only one at a time for this system)
        if self.open_positions:
            logger.warning("Position already open - cannot open another")
            return None

        self._trade_counter += 1
        trade_id = f"TRD-{self._trade_counter:06d}"

        # Calculate ATR-based stop loss
        if side == 'BUY':
            stop_loss = entry_price - (atr * self.atr_multiplier)
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier)

        # Calculate position size
        size = self.calculate_position_size(entry_price, stop_loss, atr)

        if size <= 0:
            logger.warning("Invalid position size calculated")
            return None

        trade = Trade(
            id=trade_id,
            symbol='EUR/JPY',
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=size,
            stop_loss=stop_loss,
            atr_stop=stop_loss,
            trigger_indicators=trigger_indicators or {},
            decision_probability=decision_probability,
            expected_return=expected_return
        )

        self.open_positions.append(trade)
        self.daily_trades += 1

        logger.info(
            f"Opened {side} position: {trade_id}, size={size:.2f} lots, "
            f"entry={entry_price:.3f}, stop={stop_loss:.3f}"
        )

        return trade

    def update_position(
        self,
        current_price: float,
        current_time: datetime
    ) -> List[Trade]:
        """
        Update open positions with current price.
        Check for stop loss hits and time-based exits.
        Returns list of trades to close.
        """
        trades_to_close = []

        for trade in self.open_positions:
            should_close = False
            exit_reason = None

            # Check stop loss
            if trade.side == 'BUY' and current_price <= trade.atr_stop:
                should_close = True
                exit_reason = 'stop_loss'
            elif trade.side == 'SELL' and current_price >= trade.atr_stop:
                should_close = True
                exit_reason = 'stop_loss'

            # Check time-based exit (5 minute hold)
            holding_time = (current_time - trade.entry_time).total_seconds() / 60
            if holding_time >= 5:
                should_close = True
                exit_reason = 'time_exit'

            if should_close:
                trade.exit_price = current_price
                trade.exit_time = current_time

                # Calculate PnL
                if trade.side == 'BUY':
                    pnl_pips = (current_price - trade.entry_price) * 100
                else:
                    pnl_pips = (trade.entry_price - current_price) * 100

                trade.pnl_pips = pnl_pips
                trade.pnl_usd = pnl_pips * trade.size * self.pip_value_per_lot
                trade.status = TradeStatus.STOPPED if exit_reason == 'stop_loss' else TradeStatus.CLOSED

                trades_to_close.append(trade)

        return trades_to_close

    def close_position(self, trade: Trade, reason: str = 'manual'):
        """Close a position and update balance"""
        if trade in self.open_positions:
            self.open_positions.remove(trade)
            self.closed_trades.append(trade)

            # Update balance
            if trade.pnl_usd is not None:
                self.balance += trade.pnl_usd
                self.daily_pnl += trade.pnl_usd

            logger.info(
                f"Closed {trade.id}: PnL={trade.pnl_pips:.2f} pips "
                f"(${trade.pnl_usd:.2f}), reason={reason}"
            )

    def get_open_position(self) -> Optional[Trade]:
        """Get current open position if any"""
        return self.open_positions[0] if self.open_positions else None

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio state"""
        open_trade = self.get_open_position()

        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.balance - self.initial_balance,
            'total_pnl_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_balance * 100,
            'open_positions': len(self.open_positions),
            'total_trades': len(self.closed_trades),
            'daily_trades': self.daily_trades,
            'current_trade': open_trade.to_dict() if open_trade else None,
            'risk_per_trade_pct': self.risk_per_trade * 100,
            'max_daily_loss_pct': self.max_daily_loss * 100
        }

    def reset_daily_counters(self):
        """Reset daily tracking (call at start of each trading day)"""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self._last_reset_date = today
            logger.info("Daily counters reset")

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        recent_trades = self.closed_trades[-limit:]
        return [t.to_dict() for t in reversed(recent_trades)]

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        if not self.closed_trades:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'avg_win_pips': 0,
                'avg_loss_pips': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        wins = [t for t in self.closed_trades if t.pnl_pips > 0]
        losses = [t for t in self.closed_trades if t.pnl_pips <= 0]

        win_pips = [t.pnl_pips for t in wins]
        loss_pips = [t.pnl_pips for t in losses]

        return {
            'win_rate': len(wins) / len(self.closed_trades) if self.closed_trades else 0,
            'total_trades': len(self.closed_trades),
            'avg_win_pips': sum(win_pips) / len(win_pips) if win_pips else 0,
            'avg_loss_pips': sum(loss_pips) / len(loss_pips) if loss_pips else 0,
            'best_trade': max(t.pnl_pips for t in self.closed_trades),
            'worst_trade': min(t.pnl_pips for t in self.closed_trades),
            'profit_factor': abs(sum(win_pips) / sum(loss_pips)) if loss_pips and sum(loss_pips) != 0 else float('inf')
        }
