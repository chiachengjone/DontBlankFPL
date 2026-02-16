"""
Monte Carlo Simulation Engine - Uncertainty quantification for FPL decisions.
Simulates thousands of possible season outcomes using stochastic modeling.
Imports are deferred to avoid slow startup.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import (
    MC_STD_EP_WEIGHT, MC_STD_FORM_BASELINE, MC_STD_FORM_WEIGHT,
    MC_STD_ROTATION_WEIGHT, MC_STD_MIN, MC_EPSILON,
)

# Lazy import for scipy
_scipy_stats = None

def _get_scipy_stats():
    global _scipy_stats
    if _scipy_stats is None:
        from scipy import stats
        _scipy_stats = stats
    return _scipy_stats


@dataclass
class SimulationResult:
    """Result from Monte Carlo simulation."""
    player_id: int
    mean_points: float
    median_points: float
    std_points: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    probability_exceeds_threshold: float
    value_at_risk_95: float  # VaR: worst case in 95% of simulations
    expected_shortfall_95: float  # CVaR: average of worst 5%
    sharpe_ratio: float  # Risk-adjusted return
    samples: np.ndarray = None  # raw simulation samples for portfolio use


@dataclass
class PortfolioSimulation:
    """Result from portfolio (squad) simulation."""
    iterations: int
    mean_total_points: float
    median_total_points: float
    std_total_points: float
    best_case_points: float
    worst_case_points: float
    probability_top_10k: float
    probability_top_100k: float
    expected_rank: int
    player_contributions: Dict[int, SimulationResult]


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for FPL uncertainty quantification.
    Uses various probability distributions to model player performance.
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        n_simulations: int = 10000,
        random_seed: int = None
    ):
        self.players_df = players_df.copy()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)
        self._prepare_distributions()
    
    def _prepare_distributions(self):
        """Prepare probability distributions for each player."""
        df = self.players_df.copy()
        
        # Ensure numeric columns
        df['form'] = pd.to_numeric(df.get('form', 2), errors='coerce').fillna(2.0)
        df['ep_next'] = pd.to_numeric(df.get('expected_points', df.get('ep_next', 2)), errors='coerce').fillna(2.0)
        df['total_points'] = pd.to_numeric(df.get('total_points', 0), errors='coerce').fillna(0)
        df['minutes'] = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
        
        # Calculate historical variance (for std estimation)
        df['points_per_game'] = np.where(
            df['starts'] > 0,
            df['total_points'] / df['starts'],
            0
        )
        
        # Estimate standard deviation based on consistency
        # High form = lower variance (more predictable)
        # Low minutes = higher variance (rotation risk)
        df['estimated_std'] = (
            df['ep_next'] * MC_STD_EP_WEIGHT +
            (MC_STD_FORM_BASELINE - df['form']) * MC_STD_FORM_WEIGHT +
            (90 - df['minutes'].clip(0, 90*38) / 38) / 90 * MC_STD_ROTATION_WEIGHT
        ).clip(lower=MC_STD_MIN)
        
        self.distributions = df[['id', 'ep_next', 'estimated_std', 'form', 'minutes']].to_dict('records')
    
    def _simulate_player_gamma(self, player_id: int, ep: float, std: float) -> np.ndarray:
        """
        Simulate player points using Gamma distribution.
        Gamma is ideal for modeling positive-only values with right skew.
        """
        # Prevent division by zero
        if std <= 0:
            std = MC_STD_MIN
        
        # Gamma parameters: shape (k) and scale (theta)
        # mean = k * theta, var = k * theta^2
        theta = (std ** 2) / max(ep, MC_EPSILON)
        k = max(ep, MC_EPSILON) / theta
        
        # Generate samples
        samples = self.rng.gamma(k, theta, size=self.n_simulations)
        return samples
    
    def _simulate_player_normal_truncated(
        self,
        player_id: int,
        ep: float,
        std: float
    ) -> np.ndarray:
        """
        Simulate using truncated normal (no negative points).
        Good for high-ownership premiums with consistent returns.
        """
        # Guard against invalid std
        if std <= 0:
            std = MC_STD_MIN
        # Generate normal samples
        samples = self.rng.normal(ep, std, size=self.n_simulations)
        # Truncate at 0 (no negative points)
        samples = np.maximum(samples, 0)
        return samples
    
    def _simulate_player_poisson_bonus(
        self,
        player_id: int,
        ep: float,
        std: float,
        bonus_prob: float = 0.3
    ) -> np.ndarray:
        """
        Simulate using Bivariate Poisson for base points + bonus points.
        Models correlated scoring events (goals, assists) using Holgate's model.
        """
        # Get player data for lambda estimation
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return self.rng.poisson(max(ep * 0.7, 0.5), size=self.n_simulations).astype(float)
        
        player = player_row.iloc[0]
        pos = player.get('position', 'MID')
        
        # ── Step 1: Extract or Estimate Lambdas ──
        # Prefer pre-calculated lambdas from the Poisson engine if available
        # These are usually stored in session or as engineered features
        lam_g = float(player.get('lambda_attack', 0))
        lam_a = float(player.get('lambda_creative', 0))
        
        # Fallback: estimate from ep_next if not available
        if lam_g <= 0 and lam_a <= 0:
            # Simple heuristic mapping from EP to lambdas
            from config import GOAL_POINTS, ASSIST_POINTS
            gpt = GOAL_POINTS.get(pos, 4)
            # Rough split: 60% of non-appearance points from goals for FWDs, 30% for MIDs
            att_ratio = 0.6 if pos == 'FWD' else 0.4 if pos == 'MID' else 0.1
            non_app_ep = max(ep - 2.0, 0.1)
            lam_g = (non_app_ep * att_ratio) / gpt
            lam_a = (non_app_ep * (1 - att_ratio)) / ASSIST_POINTS

        # ── Step 2: Bivariate Simulation (Holgate's Model) ──
        # X_g = Y_g + Z, X_a = Y_a + Z where Z ~ Pois(rho)
        from config import TEAM_ATTACK_CORRELATION
        rho = max(0.0, min(TEAM_ATTACK_CORRELATION, lam_g, lam_a))
        
        lam_g_ind = max(lam_g - rho, 0.0)
        lam_a_ind = max(lam_a - rho, 0.0)
        
        z = self.rng.poisson(rho, size=self.n_simulations)
        y_g = self.rng.poisson(lam_g_ind, size=self.n_simulations)
        y_a = self.rng.poisson(lam_a_ind, size=self.n_simulations)
        
        goals = y_g + z
        assists = y_a + z
        
        # ── Step 3: Calculate Points ──
        from config import GOAL_POINTS, ASSIST_POINTS, CLEAN_SHEET_POINTS
        gpt = GOAL_POINTS.get(pos, 4)
        apt = ASSIST_POINTS
        
        # Base scoring
        base_points = (goals * gpt + assists * apt).astype(float)
        
        # Appearance + CS + CBIT (simulated as binary/binomial)
        # We simplify these for MC to keep it fast
        p_plays = float(player.get('starts_pct', 0.8))
        plays = self.rng.random(size=self.n_simulations) < p_plays
        
        # Clean Sheet
        p_cs = float(player.get('p_cs', 0.3))
        cs_pts = CLEAN_SHEET_POINTS.get(pos, 0)
        cs = (self.rng.random(size=self.n_simulations) < p_cs) * cs_pts
        
        # CBIT
        p_cbit = float(player.get('p_cbit', 0.2))
        from config import CBIT_BONUS_POINTS
        cbit = (self.rng.random(size=self.n_simulations) < p_cbit) * CBIT_BONUS_POINTS
        
        # Bonus Points (simplified based on returns)
        # Hauls increase bonus probability
        bonus = np.zeros(self.n_simulations)
        has_return = (goals + assists) > 0
        haul = (goals + assists) >= 2
        
        # Probability of bonus given a return
        bonus[has_return & ~haul] = self.rng.choice([0, 1, 2], size=np.sum(has_return & ~haul), p=[0.4, 0.4, 0.2])
        bonus[haul] = self.rng.choice([1, 2, 3], size=np.sum(haul), p=[0.2, 0.3, 0.5])
        
        # Combine all components (only if the player "plays")
        samples = (2.0 + base_points + cs + cbit + bonus) * plays
        
        return samples
    
    def _simulate_player_mixed(
        self,
        player_id: int,
        ep: float,
        std: float,
        form: float
    ) -> np.ndarray:
        """
        Mixed distribution: combine multiple scenarios.
        
        - Good game: Normal(ep * 1.5, std)
        - Average game: Normal(ep, std * 0.8)
        - Bad game: Uniform(0, ep * 0.5)
        
        Probabilities based on form.
        """
        n = self.n_simulations
        
        # Scenario probabilities (form-dependent)
        form_normalized = min(max(form, 1), 9) / 9  # 0 to 1
        p_good = 0.2 + form_normalized * 0.3  # 0.2 to 0.5
        p_bad = 0.3 - form_normalized * 0.2   # 0.1 to 0.3
        p_avg = 1 - p_good - p_bad
        
        # Number of samples for each scenario
        n_good = int(n * p_good)
        n_avg = int(n * p_avg)
        n_bad = n - n_good - n_avg
        
        # Generate samples
        good_samples = self.rng.normal(ep * 1.5, std, size=n_good).clip(min=0)
        avg_samples = self.rng.normal(ep, std * 0.8, size=n_avg).clip(min=0)
        bad_samples = self.rng.uniform(0, ep * 0.5, size=n_bad)
        
        # Combine and shuffle
        samples = np.concatenate([good_samples, avg_samples, bad_samples])
        self.rng.shuffle(samples)
        
        return samples
    
    def simulate_player(
        self,
        player_id: int,
        n_gameweeks: int = 1,
        method: str = 'mixed'
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a single player.
        
        Args:
            player_id: FPL player ID
            n_gameweeks: Number of gameweeks to simulate
            method: 'gamma', 'normal', 'poisson', 'mixed'
        
        Returns:
            SimulationResult with statistics
        """
        # Get player data
        player_data = next((p for p in self.distributions if p['id'] == player_id), None)
        
        if player_data is None:
            # Default values
            ep, std, form = 2.0, 1.0, 5.0
        else:
            ep = player_data['ep_next']
            std = player_data['estimated_std']
            form = player_data['form']
        
        # Simulate based on method
        if method == 'gamma':
            samples = self._simulate_player_gamma(player_id, ep, std)
        elif method == 'normal':
            samples = self._simulate_player_normal_truncated(player_id, ep, std)
        elif method == 'poisson':
            samples = self._simulate_player_poisson_bonus(player_id, ep, std)
        else:  # mixed (default)
            samples = self._simulate_player_mixed(player_id, ep, std, form)
        
        # Scale for multiple gameweeks
        if n_gameweeks > 1:
            # Sum of independent draws
            multi_gw_samples = np.zeros(self.n_simulations)
            for _ in range(n_gameweeks):
                if method == 'mixed':
                    gw_samples = self._simulate_player_mixed(player_id, ep, std, form)
                elif method == 'gamma':
                    gw_samples = self._simulate_player_gamma(player_id, ep, std)
                elif method == 'normal':
                    gw_samples = self._simulate_player_normal_truncated(player_id, ep, std)
                else:
                    gw_samples = self._simulate_player_poisson_bonus(player_id, ep, std)
                
                multi_gw_samples += gw_samples
            
            samples = multi_gw_samples
        
        # Calculate statistics
        mean_pts = np.mean(samples)
        median_pts = np.median(samples)
        std_pts = np.std(samples)
        
        p5 = np.percentile(samples, 5)
        p25 = np.percentile(samples, 25)
        p75 = np.percentile(samples, 75)
        p95 = np.percentile(samples, 95)
        
        # Probability of exceeding expected points
        prob_exceed = np.mean(samples > ep * n_gameweeks)
        
        # Value at Risk (VaR): 5th percentile (worst case in 95% of scenarios)
        var_95 = p5
        
        # Conditional Value at Risk (CVaR): Expected Shortfall
        # Average of worst 5% outcomes
        worst_5_percent = samples[samples <= p5]
        cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else p5
        
        # Sharpe Ratio (risk-adjusted return)
        # (Expected Return - Risk-Free Rate) / Std Dev
        # Using expected points as baseline (risk-free = 2 points)
        sharpe = (mean_pts - 2) / std_pts if std_pts > 0 else 0
        
        return SimulationResult(
            player_id=player_id,
            mean_points=mean_pts,
            median_points=median_pts,
            std_points=std_pts,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            probability_exceeds_threshold=prob_exceed,
            value_at_risk_95=var_95,
            expected_shortfall_95=cvar_95,
            sharpe_ratio=sharpe,
            samples=samples
        )
    
    def simulate_portfolio(
        self,
        squad_ids: List[int],
        captain_id: int,
        bench_ids: List[int] = None,
        n_gameweeks: int = 1,
        method: str = 'mixed'
    ) -> PortfolioSimulation:
        """
        Simulate entire squad performance with correlations.
        
        Args:
            squad_ids: List of 11 starting player IDs
            captain_id: Captain ID (1.25x multiplier)
            bench_ids: Bench player IDs (optional)
            n_gameweeks: Number of gameweeks
            method: Simulation method
        
        Returns:
            PortfolioSimulation with squad-level statistics
        """
        # Simulate each player
        player_results = {}
        squad_samples = np.zeros(self.n_simulations)
        
        for pid in squad_ids:
            result = self.simulate_player(pid, n_gameweeks, method)
            player_results[pid] = result
            
            # Add per-simulation samples to squad total (not the scalar mean)
            multiplier = 1.25 if pid == captain_id else 1.0
            if result.samples is not None and len(result.samples) == self.n_simulations:
                squad_samples += result.samples * multiplier
            else:
                squad_samples += result.mean_points * multiplier
        
        # Calculate squad statistics
        mean_total = np.mean(squad_samples)
        median_total = np.median(squad_samples)
        std_total = np.std(squad_samples)
        best_case = np.percentile(squad_samples, 95)
        worst_case = np.percentile(squad_samples, 5)
        
        # Estimate finish probabilities (simplified model)
        # Top 10k typically needs ~2300+ points per season
        # Approximate per gameweek: ~60 points
        threshold_top_10k = 60 * n_gameweeks
        threshold_top_100k = 50 * n_gameweeks
        
        prob_top_10k = np.mean(squad_samples >= threshold_top_10k)
        prob_top_100k = np.mean(squad_samples >= threshold_top_100k)
        
        # Expected rank (very rough estimate)
        # Assuming normal distribution of FPL managers
        # Total managers ~10M, we estimate percentile from points
        managers_total = 10_000_000
        percentile = _get_scipy_stats().norm.cdf(mean_total, loc=50, scale=15)  # Rough distribution
        expected_rank = int(managers_total * (1 - percentile))
        
        return PortfolioSimulation(
            iterations=self.n_simulations,
            mean_total_points=mean_total,
            median_total_points=median_total,
            std_total_points=std_total,
            best_case_points=best_case,
            worst_case_points=worst_case,
            probability_top_10k=prob_top_10k,
            probability_top_100k=prob_top_100k,
            expected_rank=expected_rank,
            player_contributions=player_results
        )
    
    def sensitivity_analysis(
        self,
        player_id: int,
        param_ranges: Dict[str, Tuple[float, float]],
        n_steps: int = 10
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on input parameters.
        
        Args:
            player_id: Player to analyze
            param_ranges: Dict of parameter names to (min, max) tuples
                         e.g., {'ep_next': (2.0, 8.0), 'form': (1.0, 9.0)}
            n_steps: Number of steps in parameter sweep
        
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            for val in np.linspace(min_val, max_val, n_steps):
                # Update parameter temporarily
                original_data = next((p for p in self.distributions if p['id'] == player_id), None)
                
                if original_data:
                    temp_data = original_data.copy()
                    temp_data[param_name] = val
                    
                    # Replace in distributions
                    idx = next(i for i, p in enumerate(self.distributions) if p['id'] == player_id)
                    self.distributions[idx] = temp_data
                    
                    # Simulate
                    result = self.simulate_player(player_id, n_gameweeks=1, method='mixed')
                    
                    results.append({
                        'parameter': param_name,
                        'value': val,
                        'mean_points': result.mean_points,
                        'std_points': result.std_points,
                        'sharpe_ratio': result.sharpe_ratio
                    })
                    
                    # Restore original
                    self.distributions[idx] = original_data
        
        return pd.DataFrame(results)
    
    def compare_transfers(
        self,
        transfer_out_id: int,
        transfer_in_candidates: List[int],
        n_gameweeks: int = 5,
        method: str = 'mixed'
    ) -> pd.DataFrame:
        """
        Compare multiple transfer options using Monte Carlo.
        
        Returns:
            DataFrame ranking transfer candidates by various metrics
        """
        results = []
        
        # Simulate player being transferred out
        out_result = self.simulate_player(transfer_out_id, n_gameweeks, method)
        
        for candidate_id in transfer_in_candidates:
            in_result = self.simulate_player(candidate_id, n_gameweeks, method)
            
            # Calculate differential metrics
            points_diff = in_result.mean_points - out_result.mean_points
            risk_diff = in_result.std_points - out_result.std_points
            sharpe_diff = in_result.sharpe_ratio - out_result.sharpe_ratio
            
            # Probability new player outperforms
            prob_better = in_result.probability_exceeds_threshold
            
            # Get player info
            player_info = self.players_df[self.players_df['id'] == candidate_id].iloc[0] \
                          if candidate_id in self.players_df['id'].values else {}
            
            results.append({
                'player_id': candidate_id,
                'player_name': player_info.get('web_name', 'Unknown'),
                'position': player_info.get('position', '?'),
                'cost': player_info.get('now_cost', 0),
                'mean_points': in_result.mean_points,
                'points_gain': points_diff,
                'std_points': in_result.std_points,
                'risk_change': risk_diff,
                'sharpe_ratio': in_result.sharpe_ratio,
                'sharpe_improvement': sharpe_diff,
                'upside_95': in_result.percentile_95,
                'downside_5': in_result.percentile_5,
                'prob_outperform': prob_better
            })
        
        df = pd.DataFrame(results)
        
        # Score transfers (higher is better)
        if len(df) > 0:
            df['transfer_score'] = (
                df['points_gain'] * 10 +
                df['sharpe_improvement'] * 5 -
                df['risk_change'] * 2 +
                df['prob_outperform'] * 5
            )
            df = df.sort_values('transfer_score', ascending=False)
        
        return df


def create_monte_carlo_engine(
    players_df: pd.DataFrame,
    n_simulations: int = 10000
) -> MonteCarloEngine:
    """Factory function to create Monte Carlo engine."""
    return MonteCarloEngine(players_df, n_simulations)
