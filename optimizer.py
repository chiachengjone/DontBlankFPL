"""
Optimizer Module - Multi-Objective Optimization using PuLP for FPL Strategy Engine.
Implements the Knapsack Problem solver for 2025/26 rules.
PuLP is lazy-loaded to avoid slow startup.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config import (
    MAX_BUDGET, MAX_PLAYERS_PER_TEAM, SQUAD_SIZE, STARTING_XI,
    MAX_FREE_TRANSFERS, CAPTAIN_MULTIPLIER, POSITION_CONSTRAINTS,
)

# Lazy PuLP import
_pulp_loaded = False

def _ensure_pulp():
    global _pulp_loaded, LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, PULP_CBC_CMD, value
    if not _pulp_loaded:
        from pulp import (
            LpProblem as _LpProblem, LpMaximize as _LpMaximize,
            LpVariable as _LpVariable, LpBinary as _LpBinary,
            lpSum as _lpSum, LpStatus as _LpStatus,
            PULP_CBC_CMD as _PULP_CBC_CMD, value as _value
        )
        LpProblem = _LpProblem
        LpMaximize = _LpMaximize
        LpVariable = _LpVariable
        LpBinary = _LpBinary
        lpSum = _lpSum
        LpStatus = _LpStatus
        PULP_CBC_CMD = _PULP_CBC_CMD
        value = _value
        _pulp_loaded = True


class OptimizationMode(Enum):
    """Different optimization objectives."""
    MAX_POINTS = "maximize_points"
    DIFFERENTIAL = "maximize_differential"
    VALUE = "maximize_value"
    BALANCED = "balanced"


@dataclass
class OptimizationConstraints:
    """Constraints for the optimization problem."""
    budget: float = MAX_BUDGET
    existing_squad: List[int] = None
    free_transfers: int = 1
    bank: float = 0.0
    max_transfers: int = None  # If None, full squad optimization
    must_include: List[int] = None
    must_exclude: List[int] = None
    min_games_in_window: int = 0  # Minimum games to play in window
    

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    success: bool
    status: str
    selected_players: List[int]
    starting_xi: List[int]
    bench: List[int]
    captain: int
    vice_captain: int
    total_expected_points: float
    transfers_out: List[int]
    transfers_in: List[int]
    transfer_cost: int
    budget_remaining: float
    details: Dict


class FPLOptimizer:
    """
    Multi-Objective Optimizer for FPL squad selection.
    Uses PuLP to solve the Knapsack Problem with FPL-specific constraints.
    """
    
    def __init__(self, players_df: pd.DataFrame, weeks_ahead: int = 5):
        """
        Initialize optimizer with player data.
        
        Args:
            players_df: DataFrame with player info and expected points
            weeks_ahead: Number of weeks for the optimization horizon
        """
        _ensure_pulp()
        self.players_df = players_df.copy()
        self.weeks_ahead = weeks_ahead
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and validate player data for optimization."""
        # Ensure required columns exist
        required_cols = ['id', 'web_name', 'position', 'team', 'now_cost']
        for col in required_cols:
            if col not in self.players_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Fill missing expected points - ensure numeric
        if 'expected_points' not in self.players_df.columns:
            if 'ep_next' in self.players_df.columns:
                self.players_df['expected_points'] = pd.to_numeric(
                    self.players_df['ep_next'], errors='coerce'
                ).fillna(2.0)
            else:
                self.players_df['expected_points'] = 2.0
        
        # Ensure all key columns are numeric
        self.players_df['expected_points'] = pd.to_numeric(
            self.players_df['expected_points'], errors='coerce'
        ).fillna(2.0)
        
        self.players_df['now_cost'] = pd.to_numeric(
            self.players_df['now_cost'], errors='coerce'
        ).fillna(5.0)
        
        if 'blank_count' in self.players_df.columns:
            self.players_df['blank_count'] = pd.to_numeric(
                self.players_df['blank_count'], errors='coerce'
            ).fillna(0).astype(int)
        else:
            self.players_df['blank_count'] = 0
        
        if 'differential_score' in self.players_df.columns:
            self.players_df['differential_score'] = pd.to_numeric(
                self.players_df['differential_score'], errors='coerce'
            ).fillna(0.0)
        
        # Create player lookup
        self.player_ids = self.players_df['id'].tolist()
        self.player_lookup = self.players_df.set_index('id').to_dict('index')
    
    def _calculate_decay_weights(self, weeks: int) -> List[float]:
        """Calculate decay weights for multi-week optimization."""
        return [0.95 ** i for i in range(weeks)]
    
    def _get_weighted_ep(self, player_id: int) -> float:
        """
        Calculate weighted expected points over the horizon.
        Applies decay and accounts for blank gameweeks ("Ghost Points" logic).
        """
        player = self.player_lookup.get(player_id, {})
        base_ep = player.get('expected_points', 2.0)
        blank_count = player.get('blank_count', 0)
        
        # Apply Ghost Points penalty for blanks
        active_weeks = self.weeks_ahead - blank_count
        decay_weights = self._calculate_decay_weights(self.weeks_ahead)
        
        # If player has blanks, reduce their weighted EP
        if blank_count > 0:
            # Assume blanks are at the end of the window (conservative)
            effective_weights = decay_weights[:active_weeks]
            weight_sum = sum(effective_weights)
        else:
            weight_sum = sum(decay_weights)
        
        return base_ep * weight_sum
    
    def optimize_squad(
        self,
        mode: OptimizationMode = OptimizationMode.MAX_POINTS,
        constraints: OptimizationConstraints = None
    ) -> OptimizationResult:
        """
        Optimize full squad selection using Integer Linear Programming.
        
        Objective: Maximize Σ(Expected Points × Decay Factor) over N weeks
        
        Constraints:
        - Budget limit
        - Max 3 players per team
        - Position requirements
        - 5 Free Transfer cap (2025/26 rule)
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Create the optimization problem
        prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
        
        # Decision variables: 1 if player is selected, 0 otherwise
        squad_vars = {
            pid: LpVariable(f"squad_{pid}", cat=LpBinary)
            for pid in self.player_ids
        }
        
        # Starting XI variables
        start_vars = {
            pid: LpVariable(f"start_{pid}", cat=LpBinary)
            for pid in self.player_ids
        }
        
        # Captain variable
        captain_vars = {
            pid: LpVariable(f"captain_{pid}", cat=LpBinary)
            for pid in self.player_ids
        }
        
        # === OBJECTIVE FUNCTION ===
        weighted_eps = {
            pid: self._get_weighted_ep(pid) for pid in self.player_ids
        }
        
        if mode == OptimizationMode.MAX_POINTS:
            # Maximize total expected points
            # Starting XI get full points, captain gets 1.25x (2025/26 rule)
            prob += lpSum([
                start_vars[pid] * weighted_eps[pid] + 
                captain_vars[pid] * weighted_eps[pid] * (CAPTAIN_MULTIPLIER - 1)
                for pid in self.player_ids
            ]), "Total_Expected_Points"
            
        elif mode == OptimizationMode.DIFFERENTIAL:
            # Maximize differential score
            diff_scores = {
                pid: self.player_lookup.get(pid, {}).get('differential_score', 0)
                for pid in self.player_ids
            }
            prob += lpSum([
                start_vars[pid] * diff_scores[pid]
                for pid in self.player_ids
            ]), "Total_Differential_Score"
            
        elif mode == OptimizationMode.VALUE:
            # Maximize value (points per cost)
            value_scores = {
                pid: weighted_eps[pid] / max(self.player_lookup.get(pid, {}).get('now_cost', 4.5), 4.0)
                for pid in self.player_ids
            }
            prob += lpSum([
                start_vars[pid] * value_scores[pid]
                for pid in self.player_ids
            ]), "Total_Value_Score"
            
        else:  # BALANCED
            # Balanced objective
            prob += lpSum([
                start_vars[pid] * (
                    weighted_eps[pid] * 0.6 +
                    self.player_lookup.get(pid, {}).get('differential_score', 0) * 0.2 +
                    (5 - self.player_lookup.get(pid, {}).get('blank_count', 0)) * 0.2
                )
                for pid in self.player_ids
            ]), "Balanced_Score"
        
        # === CONSTRAINTS ===
        
        # 1. Budget constraint
        prob += lpSum([
            squad_vars[pid] * self.player_lookup.get(pid, {}).get('now_cost', 10)
            for pid in self.player_ids
        ]) <= constraints.budget, "Budget_Constraint"
        
        # 2. Squad size
        prob += lpSum([squad_vars[pid] for pid in self.player_ids]) == SQUAD_SIZE, "Squad_Size"
        
        # 3. Starting XI size
        prob += lpSum([start_vars[pid] for pid in self.player_ids]) == STARTING_XI, "Starting_XI_Size"
        
        # 4. Can only start players in squad
        for pid in self.player_ids:
            prob += start_vars[pid] <= squad_vars[pid], f"Start_In_Squad_{pid}"
        
        # 5. Captain must be in starting XI
        prob += lpSum([captain_vars[pid] for pid in self.player_ids]) == 1, "One_Captain"
        for pid in self.player_ids:
            prob += captain_vars[pid] <= start_vars[pid], f"Captain_Starts_{pid}"
        
        # 6. Max 3 players per team
        teams = self.players_df['team'].unique()
        for team in teams:
            team_players = self.players_df[self.players_df['team'] == team]['id'].tolist()
            prob += lpSum([squad_vars[pid] for pid in team_players]) <= MAX_PLAYERS_PER_TEAM, f"Max_Team_{team}"
        
        # 7. Position constraints (squad)
        for pos, limits in POSITION_CONSTRAINTS.items():
            pos_players = self.players_df[self.players_df['position'] == pos]['id'].tolist()
            prob += lpSum([squad_vars[pid] for pid in pos_players]) >= limits['min'], f"Min_{pos}"
            prob += lpSum([squad_vars[pid] for pid in pos_players]) <= limits['max'], f"Max_{pos}"
            prob += lpSum([start_vars[pid] for pid in pos_players]) >= limits['min_start'], f"Min_Start_{pos}"
            prob += lpSum([start_vars[pid] for pid in pos_players]) <= limits['max_start'], f"Max_Start_{pos}"
        
        # 8. Must include players
        if constraints.must_include:
            for pid in constraints.must_include:
                if pid in squad_vars:
                    prob += squad_vars[pid] == 1, f"Must_Include_{pid}"
        
        # 9. Must exclude players
        if constraints.must_exclude:
            for pid in constraints.must_exclude:
                if pid in squad_vars:
                    prob += squad_vars[pid] == 0, f"Must_Exclude_{pid}"
        
        # 10. Minimum games constraint (Ghost Points protection)
        if constraints.min_games_in_window > 0:
            for pid in self.player_ids:
                games = self.weeks_ahead - self.player_lookup.get(pid, {}).get('blank_count', 0)
                if games < constraints.min_games_in_window:
                    prob += start_vars[pid] == 0, f"Min_Games_{pid}"
        
        # === SOLVE ===
        solver = PULP_CBC_CMD(msg=0, timeLimit=60)
        prob.solve(solver)
        
        # === EXTRACT RESULTS ===
        if LpStatus[prob.status] != 'Optimal':
            return OptimizationResult(
                success=False,
                status=LpStatus[prob.status],
                selected_players=[],
                starting_xi=[],
                bench=[],
                captain=None,
                vice_captain=None,
                total_expected_points=0,
                transfers_out=[],
                transfers_in=[],
                transfer_cost=0,
                budget_remaining=0,
                details={'error': 'No optimal solution found'}
            )
        
        # Extract selected players
        selected = [pid for pid in self.player_ids if value(squad_vars[pid]) == 1]
        starting = [pid for pid in self.player_ids if value(start_vars[pid]) == 1]
        bench = [pid for pid in selected if pid not in starting]
        captain = [pid for pid in self.player_ids if value(captain_vars[pid]) == 1][0]
        
        # Find vice captain (highest EP in starting XI after captain)
        starting_eps = [(pid, weighted_eps[pid]) for pid in starting if pid != captain]
        starting_eps.sort(key=lambda x: x[1], reverse=True)
        vice_captain = starting_eps[0][0] if starting_eps else captain
        
        # Calculate total cost and remaining budget
        total_cost = sum(self.player_lookup.get(pid, {}).get('now_cost', 0) for pid in selected)
        
        # Calculate expected points
        total_ep = sum(
            weighted_eps[pid] * (CAPTAIN_MULTIPLIER if pid == captain else 1.0)
            for pid in starting
        )
        
        # Calculate transfers if existing squad provided
        transfers_out = []
        transfers_in = []
        transfer_cost = 0
        
        if constraints.existing_squad:
            existing_set = set(constraints.existing_squad)
            selected_set = set(selected)
            transfers_out = list(existing_set - selected_set)
            transfers_in = list(selected_set - existing_set)
            
            # Calculate transfer cost
            num_transfers = len(transfers_in)
            free_transfers = min(constraints.free_transfers, MAX_FREE_TRANSFERS)
            paid_transfers = max(0, num_transfers - free_transfers)
            transfer_cost = paid_transfers * 4  # 4 points per extra transfer
        
        return OptimizationResult(
            success=True,
            status='Optimal',
            selected_players=selected,
            starting_xi=starting,
            bench=bench,
            captain=captain,
            vice_captain=vice_captain,
            total_expected_points=total_ep,
            transfers_out=transfers_out,
            transfers_in=transfers_in,
            transfer_cost=transfer_cost,
            budget_remaining=constraints.budget - total_cost,
            details={
                'objective_value': value(prob.objective),
                'weeks_ahead': self.weeks_ahead,
                'mode': mode.value
            }
        )
    
    def optimize_transfers(
        self,
        current_squad: List[int],
        free_transfers: int = 1,
        bank: float = 0.0,
        max_transfers: int = None,
        mode: OptimizationMode = OptimizationMode.MAX_POINTS
    ) -> OptimizationResult:
        """
        Optimize transfer decisions given a current squad.
        Respects the 5 Free Transfer rollover cap (2025/26 rule).
        
        Args:
            current_squad: List of player IDs in current squad
            free_transfers: Number of free transfers available (max 5)
            bank: Money in the bank
            max_transfers: Maximum transfers to suggest (None = unlimited)
            mode: Optimization mode
        """
        # Calculate current squad value
        current_value = sum(
            self.player_lookup.get(pid, {}).get('now_cost', 0)
            for pid in current_squad
        )
        
        # Total budget = current value + bank
        total_budget = current_value + bank
        
        # Apply 5 transfer cap
        effective_free_transfers = min(free_transfers, MAX_FREE_TRANSFERS)
        
        constraints = OptimizationConstraints(
            budget=total_budget,
            existing_squad=current_squad,
            free_transfers=effective_free_transfers,
            bank=bank
        )
        
        result = self.optimize_squad(mode=mode, constraints=constraints)
        
        # If max_transfers specified, enforce it
        if max_transfers is not None and result.success:
            num_transfers = len(result.transfers_in)
            if num_transfers > max_transfers:
                # Re-optimize with constraint to keep more players
                min_keep = SQUAD_SIZE - max_transfers
                constraints.must_include = current_squad[:min_keep]  # Keep first N players
                result = self.optimize_squad(mode=mode, constraints=constraints)
        
        return result
    
    def suggest_captain(
        self,
        starting_xi: List[int],
        include_differential: bool = False
    ) -> Tuple[int, int, Dict]:
        """
        Suggest optimal captain and vice-captain picks.
        Applies 1.25x multiplier (2025/26 rule).
        
        Returns: (captain_id, vice_captain_id, analysis_dict)
        """
        captain_scores = []
        
        for pid in starting_xi:
            player = self.player_lookup.get(pid, {})
            ep = self._get_weighted_ep(pid)
            ownership = float(player.get('selected_by_percent', 50) or 50)
            
            # Base captain score
            base_score = ep * CAPTAIN_MULTIPLIER
            
            # Differential factor (captaining low-owned players is higher risk/reward)
            if include_differential and ownership < 10:
                diff_boost = (10 - ownership) / 10 * 0.2  # Up to 20% boost
                base_score *= (1 + diff_boost)
            
            captain_scores.append({
                'id': pid,
                'name': player.get('web_name', 'Unknown'),
                'expected_points': ep,
                'captain_points': ep * CAPTAIN_MULTIPLIER,
                'ownership': ownership,
                'score': base_score
            })
        
        # Sort by score
        captain_scores.sort(key=lambda x: x['score'], reverse=True)
        
        captain = captain_scores[0]['id']
        vice_captain = captain_scores[1]['id'] if len(captain_scores) > 1 else captain
        
        return captain, vice_captain, {
            'rankings': captain_scores,
            'captain_boost': CAPTAIN_MULTIPLIER,
            'top_3': captain_scores[:3]
        }
    
    def analyze_chip_timing(
        self,
        current_squad: List[int],
        chips_available: List[str]
    ) -> Dict:
        """
        Analyze optimal timing for chips (Wildcard, Free Hit, etc.)
        """
        analysis = {}
        
        # Get fixture difficulty across next 10 weeks
        all_fixture_ease = []
        for pid in current_squad:
            player = self.player_lookup.get(pid, {})
            ease = player.get('fixture_ease', 0.5)
            blanks = player.get('blank_count', 0)
            all_fixture_ease.append({'ease': ease, 'blanks': blanks})
        
        avg_ease = np.mean([p['ease'] for p in all_fixture_ease])
        total_blanks = sum(p['blanks'] for p in all_fixture_ease)
        
        # Wildcard recommendation
        if 'wildcard' in chips_available:
            wc_score = (1 - avg_ease) + (total_blanks / len(current_squad))
            analysis['wildcard'] = {
                'urgency': wc_score,
                'recommendation': 'High' if wc_score > 0.6 else 'Medium' if wc_score > 0.3 else 'Low',
                'reason': f"Team fixture ease: {avg_ease:.2f}, Blank exposure: {total_blanks}"
            }
        
        # Free Hit recommendation (good for blank gameweeks)
        if 'free_hit' in chips_available:
            # Free Hit is best when many players have blanks
            blank_severity = total_blanks / (len(current_squad) * self.weeks_ahead)
            analysis['free_hit'] = {
                'urgency': blank_severity,
                'recommendation': 'Consider' if blank_severity > 0.2 else 'Hold',
                'reason': f"{total_blanks} blank fixtures detected in your squad"
            }
        
        # Bench Boost recommendation
        if 'bench_boost' in chips_available:
            analysis['bench_boost'] = {
                'urgency': avg_ease,
                'recommendation': 'Good' if avg_ease > 0.7 else 'Wait',
                'reason': f"Fixture congestion score: {avg_ease:.2f}"
            }
        
        # Triple Captain recommendation
        if 'triple_captain' in chips_available:
            # Find best captaincy opportunity
            best_ep = max(
                self._get_weighted_ep(pid) 
                for pid in current_squad
            )
            tc_score = best_ep / 10  # Normalize
            analysis['triple_captain'] = {
                'urgency': tc_score,
                'recommendation': 'Strong' if tc_score > 0.8 else 'Hold',
                'reason': f"Best captain EP: {best_ep:.1f}"
            }
        
        return analysis


class DefConEngine:
    """
    The DefCon Engine - Calculates CBIT (Clearances, Blocks, Interceptions, Tackles)
    propensity scores for defenders under 2025/26 rules.
    
    10 CBIT = +2 bonus points
    """
    
    def __init__(self, players_df: pd.DataFrame):
        self.players_df = players_df.copy()
        self.CBIT_THRESHOLD = 10
        self.CBIT_BONUS = 2
    
    def calculate_cbit_score(self, player_id: int) -> Dict:
        """
        Calculate comprehensive CBIT analysis for a player.
        """
        player = self.players_df[self.players_df['id'] == player_id]
        if player.empty:
            return {'error': 'Player not found'}
        
        player = player.iloc[0]
        
        # Get CBIT metrics (using available proxies)
        cbit_per_90 = player.get('cbit_per_90', 0)
        cbit_propensity = player.get('cbit_propensity', 0)
        
        # Probability of hitting 10 CBIT threshold
        # Using a simple model based on historical distribution
        hit_probability = min(cbit_propensity * 1.2, 1.0)  # Adjusted for realism
        
        # Expected bonus points from CBIT
        expected_cbit_bonus = hit_probability * self.CBIT_BONUS
        
        # Value adjustment for defenders
        minutes_per_game = player.get('minutes_per_game', 0)
        playing_regularity = min(minutes_per_game / 90, 1.0)
        
        return {
            'player_id': player_id,
            'name': player.get('web_name', 'Unknown'),
            'position': player.get('position', 'Unknown'),
            'cbit_per_90': cbit_per_90,
            'cbit_propensity': cbit_propensity,
            'threshold_hit_probability': hit_probability,
            'expected_cbit_bonus': expected_cbit_bonus,
            'playing_regularity': playing_regularity,
            'adjusted_value': expected_cbit_bonus * playing_regularity,
            'defcon_rating': self._calculate_defcon_rating(
                cbit_propensity, hit_probability, playing_regularity
            )
        }
    
    def _calculate_defcon_rating(
        self, 
        propensity: float, 
        hit_prob: float, 
        regularity: float
    ) -> str:
        """
        Calculate DefCon rating (1-5 scale, like threat levels).
        DefCon 1 = Highest value defender for CBIT points
        """
        score = (propensity * 0.4 + hit_prob * 0.4 + regularity * 0.2)
        
        if score >= 0.8:
            return "DefCon 1 - Elite"
        elif score >= 0.6:
            return "DefCon 2 - Premium"
        elif score >= 0.4:
            return "DefCon 3 - Solid"
        elif score >= 0.2:
            return "DefCon 4 - Average"
        else:
            return "DefCon 5 - Limited"
    
    def rank_defenders_by_cbit(self) -> pd.DataFrame:
        """
        Rank all defenders by their CBIT potential.
        """
        defenders = self.players_df[self.players_df['position'] == 'DEF'].copy()
        
        cbit_data = []
        for _, player in defenders.iterrows():
            analysis = self.calculate_cbit_score(player['id'])
            if 'error' not in analysis:
                cbit_data.append(analysis)
        
        df = pd.DataFrame(cbit_data)
        df = df.sort_values('adjusted_value', ascending=False)
        
        return df


def create_optimizer(players_df: pd.DataFrame, weeks_ahead: int = 5) -> FPLOptimizer:
    """Factory function to create an optimizer instance."""
    return FPLOptimizer(players_df, weeks_ahead)


def quick_optimize(
    players_df: pd.DataFrame,
    budget: float = 100.0,
    weeks_ahead: int = 5,
    mode: str = 'points'
) -> OptimizationResult:
    """
    Quick optimization function for simple use cases.
    
    Args:
        players_df: Player data with expected points
        budget: Available budget
        weeks_ahead: Optimization horizon
        mode: 'points', 'differential', 'value', or 'balanced'
    """
    mode_map = {
        'points': OptimizationMode.MAX_POINTS,
        'differential': OptimizationMode.DIFFERENTIAL,
        'value': OptimizationMode.VALUE,
        'balanced': OptimizationMode.BALANCED
    }
    
    optimizer = FPLOptimizer(players_df, weeks_ahead)
    constraints = OptimizationConstraints(budget=budget)
    
    return optimizer.optimize_squad(
        mode=mode_map.get(mode, OptimizationMode.MAX_POINTS),
        constraints=constraints
    )
