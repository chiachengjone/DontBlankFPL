"""
Backtesting Framework - Historical strategy validation and performance analysis.
Tests optimization strategies against past seasons to measure effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestMetrics:
    """Performance metrics from backtesting."""
    total_points: int
    average_gw_points: float
    best_gw_points: int
    worst_gw_points: int
    std_gw_points: float
    
    # Cumulative
    cumulative_rank: int
    final_rank: int
    
    # Efficiency metrics
    bench_points_wasted: int
    captain_efficiency: float  # Ratio of captain points to best possible
    transfer_cost: int
    net_points: int  # Total - transfer cost
    
    # Risk metrics
    max_drawdown: float  # Worst rank drop
    volatility: float  # Std of weekly rank changes
    sharpe_ratio: float
    
    # Strategy stats
    total_transfers: int
    wildcard_count: int
    chip_efficiency: float
    
    # Comparative
    percentile_finish: float  # Top X%
    points_vs_average: float


@dataclass
class BacktestResult:
    """Complete backtesting result."""
    strategy_name: str
    season: str
    metrics: BacktestMetrics
    gameweek_history: pd.DataFrame
    transfer_history: List[Dict]
    squad_history: List[List[int]]
    chip_usage: Dict[str, int]
    
    def summary(self) -> str:
        """Generate summary report."""
        m = self.metrics
        return f"""
=== Backtest Summary: {self.strategy_name} ({self.season}) ===
Total Points: {m.total_points}
Net Points: {m.net_points} (after {m.transfer_cost} in transfer costs)
Final Rank: {m.final_rank:,}
Percentile: Top {m.percentile_finish:.1f}%

Performance:
- Average GW: {m.average_gw_points:.1f} pts
- Best GW: {m.best_gw_points} pts
- Volatility: {m.std_gw_points:.1f}
- Sharpe Ratio: {m.sharpe_ratio:.2f}

Efficiency:
- Captain Efficiency: {m.captain_efficiency*100:.1f}%
- Bench Waste: {m.bench_points_wasted} pts
- Transfers: {m.total_transfers} made

Risk:
- Max Drawdown: {m.max_drawdown:,.0f} ranks
- Volatility: {m.volatility:.2f}
"""


class HistoricalDataLoader:
    """Load and prepare historical FPL data for backtesting."""
    
    def __init__(self, season: str = "2024-25"):
        self.season = season
        self.historical_data = {}
    
    def load_season_data(self, data_path: Optional[str] = None) -> Dict:
        """
        Load historical season data.
        In production, this would load from files/database.
        For now, generates synthetic historical data.
        """
        # Generate synthetic historical data (38 gameweeks)
        n_players = 600
        n_gameweeks = 38
        
        # Player master list
        players = []
        for i in range(1, n_players + 1):
            players.append({
                'id': i,
                'web_name': f'Player_{i}',
                'team': (i % 20) + 1,
                'position': ['GKP', 'DEF', 'MID', 'FWD'][(i % 4)],
                'now_cost': np.random.uniform(4.0, 13.0)
            })
        
        # Gameweek performance (actual points scored)
        gw_data = []
        for gw in range(1, n_gameweeks + 1):
            for player in players:
                # Simulate points with some realism
                base_pts = np.random.poisson(3) if np.random.random() > 0.3 else 0
                bonus = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
                
                gw_data.append({
                    'gameweek': gw,
                    'player_id': player['id'],
                    'points': base_pts + bonus,
                    'minutes': np.random.choice([0, 15, 30, 60, 90], p=[0.15, 0.05, 0.05, 0.15, 0.6]),
                    'goals': np.random.poisson(0.2) if base_pts > 0 else 0,
                    'assists': np.random.poisson(0.15) if base_pts > 0 else 0,
                    'clean_sheet': 1 if np.random.random() > 0.65 else 0
                })
        
        self.historical_data = {
            'players': pd.DataFrame(players),
            'gameweek_data': pd.DataFrame(gw_data)
        }
        
        return self.historical_data
    
    def get_gameweek_data(self, gw: int) -> pd.DataFrame:
        """Get actual player points for a specific gameweek."""
        gw_df = self.historical_data['gameweek_data']
        return gw_df[gw_df['gameweek'] == gw].copy()


class BacktestEngine:
    """
    Backtesting engine for FPL strategies.
    Simulates a season using historical data and a selection strategy.
    """
    
    def __init__(
        self,
        historical_data: Dict,
        initial_budget: float = 100.0,
        enable_chips: bool = True
    ):
        self.historical_data = historical_data
        self.initial_budget = initial_budget
        self.enable_chips = enable_chips
        
        self.players_df = historical_data['players'].copy()
        self.gw_data = historical_data['gameweek_data'].copy()
        
        # State tracking
        self.reset_state()
    
    def reset_state(self):
        """Reset backtest state for new run."""
        self.current_squad = []
        self.bank = 0.0
        self.free_transfers = 1
        self.total_transfers = 0
        self.total_transfer_cost = 0
        
        self.gameweek_scores = []
        self.transfer_log = []
        self.squad_history = []
        self.chip_usage = {
            'wildcard': 0,
            'free_hit': 0,
            'bench_boost': 0,
            'triple_captain': 0
        }
        
        self.current_gw = 1
    
    def run_backtest(
        self,
        strategy_func: Callable,
        strategy_name: str = "Custom Strategy",
        start_gw: int = 1,
        end_gw: int = 38
    ) -> BacktestResult:
        """
        Run complete backtest using a strategy function.
        
        Args:
            strategy_func: Function(gw, current_squad, data) -> (squad, captain, chip)
            strategy_name: Name of the strategy
            start_gw: Starting gameweek
            end_gw: Ending gameweek
        
        Returns:
            BacktestResult with performance metrics
        """
        self.reset_state()
        
        # Initial squad selection (GW1)
        squad, captain, chip = strategy_func(1, [], self.historical_data)
        self.current_squad = squad
        self.squad_history.append(squad.copy())
        
        # Run through each gameweek
        for gw in range(start_gw, end_gw + 1):
            self.current_gw = gw
            
            # Get this week's actual results
            gw_results = self.gw_data[self.gw_data['gameweek'] == gw]
            
            # Calculate squad points
            gw_points = self._calculate_gw_points(squad, captain, chip, gw_results)
            
            # Log results
            self.gameweek_scores.append({
                'gameweek': gw,
                'points': gw_points['total'],
                'captain': captain,
                'chip': chip,
                'bench_points': gw_points['bench'],
                'transfer_cost': gw_points['transfer_cost']
            })
            
            # Update for next week (if not final GW)
            if gw < end_gw:
                # Get strategy's decisions for next GW
                new_squad, new_captain, new_chip = strategy_func(
                    gw + 1,
                    self.current_squad,
                    self.historical_data
                )
                
                # Calculate transfers
                transfers = self._calculate_transfers(self.current_squad, new_squad)
                
                # Apply transfer costs
                transfer_cost = self._apply_transfer_cost(len(transfers), new_chip)
                self.total_transfer_cost += transfer_cost
                
                # Update state
                self.current_squad = new_squad
                squad = new_squad
                captain = new_captain
                chip = new_chip
                
                self.squad_history.append(new_squad.copy())
                
                if len(transfers) > 0:
                    self.transfer_log.append({
                        'gameweek': gw + 1,
                        'transfers': transfers,
                        'cost': transfer_cost
                    })
                
                # Update chips
                if new_chip and new_chip != 'none':
                    self.chip_usage[new_chip] = self.chip_usage.get(new_chip, 0) + 1
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Build result
        gw_history_df = pd.DataFrame(self.gameweek_scores)
        
        return BacktestResult(
            strategy_name=strategy_name,
            season=self.historical_data.get('season', 'Unknown'),
            metrics=metrics,
            gameweek_history=gw_history_df,
            transfer_history=self.transfer_log,
            squad_history=self.squad_history,
            chip_usage=self.chip_usage
        )
    
    def _calculate_gw_points(
        self,
        squad: List[int],
        captain: int,
        chip: Optional[str],
        gw_results: pd.DataFrame
    ) -> Dict:
        """Calculate points for a gameweek."""
        # Get points for each player
        squad_points = []
        for pid in squad[:11]:  # Starting 11
            player_gw = gw_results[gw_results['player_id'] == pid]
            pts = player_gw['points'].iloc[0] if not player_gw.empty else 0
            
            # Captain multiplier
            if pid == captain:
                if chip == 'triple_captain':
                    pts *= 3
                else:
                    pts *= 1.25  # 2025/26 rule
            
            squad_points.append(pts)
        
        # Bench points (4 players)
        bench_points = []
        for pid in squad[11:]:
            player_gw = gw_results[gw_results['player_id'] == pid]
            pts = player_gw['points'].iloc[0] if not player_gw.empty else 0
            bench_points.append(pts)
        
        total_starting = sum(squad_points)
        total_bench = sum(bench_points)
        
        # Bench boost
        if chip == 'bench_boost':
            total = total_starting + total_bench
        else:
            total = total_starting
        
        return {
            'total': total,
            'starting': total_starting,
            'bench': total_bench,
            'transfer_cost': 0  # Calculated separately
        }
    
    def _calculate_transfers(self, old_squad: List[int], new_squad: List[int]) -> List[Tuple[int, int]]:
        """Calculate transfers between two squads."""
        old_set = set(old_squad)
        new_set = set(new_squad)
        
        transfers_out = old_set - new_set
        transfers_in = new_set - old_set
        
        return list(zip(transfers_out, transfers_in))
    
    def _apply_transfer_cost(self, n_transfers: int, chip: Optional[str] = None) -> int:
        """Calculate transfer cost."""
        if chip in ['wildcard', 'free_hit']:
            return 0  # Free transfers
        
        if n_transfers <= self.free_transfers:
            # Reset free transfers
            self.free_transfers = 1
            return 0
        else:
            cost = (n_transfers - self.free_transfers) * 4
            self.free_transfers = 1
            self.total_transfers += n_transfers
            return cost
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        gw_df = pd.DataFrame(self.gameweek_scores)
        
        total_points = gw_df['points'].sum()
        avg_gw = gw_df['points'].mean()
        best_gw = gw_df['points'].max()
        worst_gw = gw_df['points'].min()
        std_gw = gw_df['points'].std()
        
        bench_wasted = gw_df['bench_points'].sum()
        net_points = total_points - self.total_transfer_cost
        
        # Captain efficiency (simplified)
        captain_efficiency = 0.85  # Placeholder
        
        # Sharpe ratio
        risk_free = 50  # Average GW points baseline
        sharpe = (avg_gw - risk_free) / std_gw if std_gw > 0 else 0
        
        # Rank simulation (simplified)
        # Assume normal distribution of managers
        avg_season_points = 2000
        std_season_points = 300
        from scipy import stats
        percentile = stats.norm.cdf(total_points, loc=avg_season_points, scale=std_season_points)
        total_managers = 10_000_000
        final_rank = int(total_managers * (1 - percentile))
        # percentile_pct represents "Top X%" - lower is better
        percentile_pct = (1 - percentile) * 100
        
        # Max drawdown (simulate rank volatility)
        cumulative = gw_df['points'].cumsum()
        rolling_max = cumulative.expanding().max()
        drawdown = (rolling_max - cumulative).max()
        
        return BacktestMetrics(
            total_points=int(total_points),
            average_gw_points=avg_gw,
            best_gw_points=int(best_gw),
            worst_gw_points=int(worst_gw),
            std_gw_points=std_gw,
            cumulative_rank=final_rank,
            final_rank=final_rank,
            bench_points_wasted=int(bench_wasted),
            captain_efficiency=captain_efficiency,
            transfer_cost=self.total_transfer_cost,
            net_points=int(net_points),
            max_drawdown=drawdown,
            volatility=std_gw,
            sharpe_ratio=sharpe,
            total_transfers=self.total_transfers,
            wildcard_count=self.chip_usage.get('wildcard', 0),
            chip_efficiency=0.8,  # Placeholder
            percentile_finish=percentile_pct,
            points_vs_average=total_points - avg_season_points
        )
    
    def compare_strategies(
        self,
        strategies: Dict[str, Callable],
        start_gw: int = 1,
        end_gw: int = 38
    ) -> pd.DataFrame:
        """
        Compare multiple strategies head-to-head.
        
        Args:
            strategies: Dict of {name: strategy_function}
            start_gw: Start gameweek
            end_gw: End gameweek
        
        Returns:
            DataFrame comparing all strategies
        """
        results = []
        
        for name, strategy_func in strategies.items():
            print(f"Backtesting: {name}...")
            result = self.run_backtest(strategy_func, name, start_gw, end_gw)
            
            m = result.metrics
            results.append({
                'Strategy': name,
                'Total Points': m.total_points,
                'Net Points': m.net_points,
                'Avg GW': round(m.average_gw_points, 1),
                'Final Rank': f"{m.final_rank:,}",
                'Percentile': f"{m.percentile_finish:.1f}%",
                'Sharpe': round(m.sharpe_ratio, 2),
                'Transfers': m.total_transfers,
                'Transfer Cost': m.transfer_cost,
                'Bench Waste': m.bench_points_wasted,
                'Volatility': round(m.volatility, 1)
            })
        
        df = pd.DataFrame(results).sort_values('Net Points', ascending=False)
        return df


# Pre-built strategy examples

def greedy_strategy(gw: int, current_squad: List[int], data: Dict) -> Tuple[List[int], int, Optional[str]]:
    """
    Simple greedy strategy: always pick highest expected points.
    """
    players_df = data['players'].copy()
    
    # Sort by estimated EP (using ID as proxy)
    players_df['estimated_ep'] = np.random.uniform(2, 8, len(players_df))
    players_df = players_df.sort_values('estimated_ep', ascending=False)
    
    # Build legal squad (simplified)
    squad = players_df['id'].head(15).tolist()
    captain = squad[0]
    chip = 'wildcard' if gw == 1 else None
    
    return squad, captain, chip


def balanced_strategy(gw: int, current_squad: List[int], data: Dict) -> Tuple[List[int], int, Optional[str]]:
    """
    Balanced strategy: mix of value and expected points.
    """
    players_df = data['players'].copy()
    
    players_df['estimated_ep'] = np.random.uniform(2, 8, len(players_df))
    players_df['value'] = players_df['estimated_ep'] / players_df['now_cost']
    players_df['score'] = players_df['estimated_ep'] * 0.6 + players_df['value'] * 0.4
    
    players_df = players_df.sort_values('score', ascending=False)
    
    squad = players_df['id'].head(15).tolist()
    captain = squad[0]
    chip = None
    
    return squad, captain, chip


def create_backtest_engine(historical_data: Dict) -> BacktestEngine:
    """Factory function to create backtest engine."""
    return BacktestEngine(historical_data)
