"""
Poisson-Based Expected Points Engine for FPL 2025/26.

Professional-grade probabilistic model using:
- Poisson distribution for goals/assists (match-specific λ)
- Binomial distribution for CBIT threshold attainment
- Poisson-Gamma framework for clean sheet probability
- Handles Double Gameweeks via fixture iteration

Based on methodology used by FPL Review and similar top-tier solvers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson, binom

logger = logging.getLogger(__name__)

# ── FPL 2025/26 Scoring Rules ──
GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_PTS = 3
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
APPEARANCE_PTS = {60: 2, 1: 1}  # 60+ mins = 2pts, 1-59 = 1pt
CBIT_THRESHOLD = {"GKP": 12, "DEF": 10, "MID": 12, "FWD": 12}
CBIT_BONUS = 2
GOALS_CONCEDED_PER_2 = -1  # GKP/DEF lose 1pt per 2 goals conceded
SAVE_PTS = 3  # 3 saves = 1pt (for goalkeepers)

# Home/Away multipliers (based on historical EPL data)
HOME_ADVANTAGE_ATTACK = 1.15  # ~15% more goals at home
HOME_ADVANTAGE_DEFENSE = 0.90  # ~10% fewer goals conceded at home
AWAY_MULTIPLIER_ATTACK = 0.87
AWAY_MULTIPLIER_DEFENSE = 1.10

# League average benchmarks (updated each season)
LEAGUE_AVG_XG_PER_MATCH = 1.35  # ~2.7 goals per game split between teams
LEAGUE_AVG_XGA_PER_MATCH = 1.35

# Maximum k for Poisson calculations (player rarely scores 5+ in a match)
MAX_K_GOALS = 5
MAX_K_ASSISTS = 4


class PoissonEPEngine:
    """
    Professional-grade Expected Points calculator using Poisson distributions.
    
    Key features:
    - Match-specific λ (lambda) based on player xG/90 × opponent defensive weakness
    - Handles Double Gameweeks by summing independent fixture outcomes
    - Includes CBIT (2025/26 defensive actions) bonus probability
    - Clean sheet probability using Poisson model for opponent scoring
    """
    
    def __init__(self, team_stats: Optional[pd.DataFrame] = None):
        """
        Initialize engine with optional team-level xG/xGA stats.
        
        Args:
            team_stats: DataFrame with columns: team_name, xG_for_per90, xGA_per90
        """
        self.team_stats = team_stats if team_stats is not None else pd.DataFrame()
        self._league_avg_xg = LEAGUE_AVG_XG_PER_MATCH
        self._league_avg_xga = LEAGUE_AVG_XGA_PER_MATCH
        
        if not self.team_stats.empty and 'xGA_per90' in self.team_stats.columns:
            self._league_avg_xga = self.team_stats['xGA_per90'].mean()
        if not self.team_stats.empty and 'xG_for_per90' in self.team_stats.columns:
            self._league_avg_xg = self.team_stats['xG_for_per90'].mean()
    
    def calculate_attack_lambda(
        self,
        player_xg_per90: float,
        opponent_xga_per90: float,
        is_home: bool,
        minutes_expected: float = 90.0
    ) -> float:
        """
        Calculate match-specific λ for attacking returns (goals).
        
        λ_attack = (Player xG/90) × (Opponent xGA / League xGA) × Home/Away × (Mins/90)
        
        Args:
            player_xg_per90: Player's expected goals per 90 minutes
            opponent_xga_per90: Opponent's expected goals against per 90
            is_home: Whether player is at home
            minutes_expected: Expected minutes (for rotation risk)
        
        Returns:
            Lambda value for Poisson distribution
        """
        if player_xg_per90 <= 0:
            return 0.0
        
        # Opponent defensive weakness factor
        opp_factor = opponent_xga_per90 / self._league_avg_xga if self._league_avg_xga > 0 else 1.0
        
        # Home/away adjustment
        venue_mult = HOME_ADVANTAGE_ATTACK if is_home else AWAY_MULTIPLIER_ATTACK
        
        # Minutes adjustment (for rotation risk)
        mins_factor = min(minutes_expected / 90.0, 1.0)
        
        return player_xg_per90 * opp_factor * venue_mult * mins_factor
    
    def calculate_creative_lambda(
        self,
        player_xa_per90: float,
        opponent_defensive_quality: float,
        is_home: bool,
        minutes_expected: float = 90.0
    ) -> float:
        """
        Calculate match-specific λ for creative returns (assists).
        
        λ_creative = (Player xA/90) × (Opponent xG / League xG) × Home/Away × (Mins/90)
        
        Note: We use opponent xG as proxy for how much attacking play 
        they'll force the player's team into (better opponents = more chances).
        """
        if player_xa_per90 <= 0:
            return 0.0
        
        # Creative factor based on opponent defensive style
        creative_factor = opponent_defensive_quality / self._league_avg_xg if self._league_avg_xg > 0 else 1.0
        creative_factor = max(0.7, min(creative_factor, 1.4))  # Bound the adjustment
        
        venue_mult = HOME_ADVANTAGE_ATTACK if is_home else AWAY_MULTIPLIER_ATTACK
        mins_factor = min(minutes_expected / 90.0, 1.0)
        
        return player_xa_per90 * creative_factor * venue_mult * mins_factor
    
    def poisson_expected_points(
        self,
        lambda_val: float,
        points_per_return: int,
        max_k: int = MAX_K_GOALS
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate expected points using Poisson PMF.
        
        xP = Σ P(k) × k × PointsPerReturn for k ∈ {0, 1, 2, ..., max_k}
        
        Returns:
            Tuple of (expected_points, probability_distribution)
        """
        if lambda_val <= 0:
            return 0.0, np.zeros(max_k + 1)
        
        # Calculate probability mass for each k
        k_values = np.arange(max_k + 1)
        probabilities = poisson.pmf(k_values, lambda_val)
        
        # Expected returns: Σ P(k) × k
        expected_returns = np.sum(probabilities * k_values)
        
        # Expected points
        xp = expected_returns * points_per_return
        
        return xp, probabilities
    
    def calculate_clean_sheet_probability(
        self,
        opponent_xg_per90: float,
        is_home: bool,
        team_defensive_quality: float = 1.0
    ) -> float:
        """
        Calculate clean sheet probability using Poisson model.
        
        P(CS) = P(opponent scores 0) = e^(-λ_opponent)
        
        Args:
            opponent_xg_per90: Opponent's attacking xG per 90
            is_home: Whether defending team is at home
            team_defensive_quality: Team's defensive multiplier (xGA / league avg)
        
        Returns:
            Probability of keeping a clean sheet
        """
        # Opponent's expected goals in this match
        venue_mult = HOME_ADVANTAGE_DEFENSE if is_home else AWAY_MULTIPLIER_DEFENSE
        lambda_opponent = opponent_xg_per90 * team_defensive_quality * venue_mult
        
        # P(0 goals) = e^(-λ)
        p_cs = poisson.pmf(0, max(lambda_opponent, 0.01))
        
        return min(p_cs, 0.6)  # Cap at 60% - even best teams rarely > 50%
    
    def calculate_goals_conceded_penalty(
        self,
        opponent_xg_per90: float,
        is_home: bool,
        team_defensive_quality: float = 1.0,
        position: str = "DEF"
    ) -> float:
        """
        Calculate expected penalty from goals conceded (GKP/DEF only).
        
        E[penalty] = Σ P(k goals) × floor(k/2) × (-1)
        """
        if position not in ("GKP", "DEF"):
            return 0.0
        
        venue_mult = HOME_ADVANTAGE_DEFENSE if is_home else AWAY_MULTIPLIER_DEFENSE
        lambda_opponent = opponent_xg_per90 * team_defensive_quality * venue_mult
        
        # Calculate expected goals conceded
        k_values = np.arange(7)  # 0-6 goals
        probabilities = poisson.pmf(k_values, max(lambda_opponent, 0.01))
        
        # Penalty: -1 per 2 goals conceded
        penalties = np.floor(k_values / 2) * GOALS_CONCEDED_PER_2
        expected_penalty = np.sum(probabilities * penalties)
        
        return expected_penalty
    
    def calculate_cbit_probability(
        self,
        player_tackles_per90: float,
        player_interceptions_per90: float,
        player_clearances_per90: float,
        position: str,
        minutes_expected: float = 90.0
    ) -> float:
        """
        Calculate probability of hitting CBIT threshold using Binomial model.
        
        CBIT = Tackles + Interceptions + Clearances + Blocks
        For 2025/26: DEF needs 10+, others need 12+ for +2 bonus
        
        Models as sum of Binomial trials:
        P(CBIT ≥ threshold) using normal approximation for sum of independent counts
        """
        threshold = CBIT_THRESHOLD.get(position, 12)
        
        # Estimate total defensive actions per 90
        total_actions_per90 = player_tackles_per90 + player_interceptions_per90 + player_clearances_per90
        
        # Adjust for expected minutes
        mins_factor = min(minutes_expected / 90.0, 1.0)
        expected_actions = total_actions_per90 * mins_factor
        
        if expected_actions <= 0:
            return 0.0
        
        # Use Poisson approximation for counting process
        # P(actions >= threshold)
        p_hit = 1 - poisson.cdf(threshold - 1, expected_actions)
        
        return min(max(p_hit, 0.0), 1.0)
    
    def calculate_appearance_probability(
        self,
        minutes_per_start: float,
        starts_pct: float,
        chance_of_playing: float = 100.0,
    ) -> Tuple[float, float]:
        """
        Calculate probability of appearance and 60+ minutes.
        
        Args:
            minutes_per_start: Average minutes per start
            starts_pct: Fraction of team games the player starts (0-1)
            chance_of_playing: FPL chance_of_playing_next_round (0-100)
        
        Returns:
            Tuple of (P(plays), P(60+ mins | plays))
        """
        # Use FPL's granular chance_of_playing percentage
        availability = max(0.0, min(float(chance_of_playing), 100.0)) / 100.0
        
        # Probability of playing
        p_plays = starts_pct * availability
        
        # Probability of 60+ minutes given playing
        if minutes_per_start >= 80:
            p_60_plus = 0.95
        elif minutes_per_start >= 60:
            p_60_plus = 0.85
        elif minutes_per_start >= 45:
            p_60_plus = 0.50
        else:
            p_60_plus = 0.25
        
        return p_plays, p_60_plus
    
    def calculate_bonus_expected(
        self,
        xg_lambda: float,
        xa_lambda: float,
        position: str
    ) -> float:
        """
        Calculate expected bonus points using haul probability.
        
        Bonus points are non-linear - a player with 2+ returns in a match
        has high probability of getting 2-3 bonus points.
        
        Uses joint probability of multi-return outcomes.
        """
        # Probabilities of each outcome
        p_0g = poisson.pmf(0, xg_lambda) if xg_lambda > 0 else 1.0
        p_1g = poisson.pmf(1, xg_lambda) if xg_lambda > 0 else 0.0
        p_2g_plus = 1 - p_0g - p_1g if xg_lambda > 0 else 0.0
        
        p_0a = poisson.pmf(0, xa_lambda) if xa_lambda > 0 else 1.0
        p_1a = poisson.pmf(1, xa_lambda) if xa_lambda > 0 else 0.0
        p_2a_plus = 1 - p_0a - p_1a if xa_lambda > 0 else 0.0
        
        # Haul scenarios and expected bonus
        # 1 goal only: ~0.5 avg bonus
        # 1 assist only: ~0.3 avg bonus  
        # 2+ goals: ~2.5 avg bonus
        # 1g + 1a: ~2.0 avg bonus
        # 2+ goals + assist(s): ~3.0 avg bonus (max)
        
        # Position adjustments (forwards need more for bonus)
        pos_mult = {"GKP": 1.5, "DEF": 1.3, "MID": 1.0, "FWD": 0.85}.get(position, 1.0)
        
        expected_bonus = (
            # Single goal scenarios
            p_1g * p_0a * 0.5 * pos_mult +
            # Single assist only
            p_0g * p_1a * 0.3 * pos_mult +
            # Goal + assist  
            p_1g * p_1a * 2.0 * pos_mult +
            # Multiple goals
            p_2g_plus * p_0a * 2.5 * pos_mult +
            # Multiple goals + assist
            p_2g_plus * (1 - p_0a) * 3.0 * pos_mult +
            # Multiple assists + goal
            p_1g * p_2a_plus * 2.2 * pos_mult
        )
        
        return min(expected_bonus, 3.0)  # Cap at 3 bonus points

    def calculate_single_fixture_xp(
        self,
        player: Dict,
        opponent_stats: Dict,
        is_home: bool
    ) -> Dict[str, float]:
        """
        Calculate expected points for a single fixture.
        
        Args:
            player: Dict with player stats (xG_per90, xA_per90, position, etc.)
            opponent_stats: Dict with opponent team stats (xGA_per90, xG_per90)
            is_home: Whether player is at home
        
        Returns:
            Dict with breakdown of expected points components
        """
        position = player.get('position', 'MID')
        
        # Get player stats with fallbacks
        xg_per90 = float(player.get('us_xG_per90', player.get('expected_goals_per_90', 0)) or 0)
        xa_per90 = float(player.get('us_xA_per90', player.get('expected_assists_per_90', 0)) or 0)
        minutes_per_game = float(player.get('minutes_per_game', 70) or 70)
        starts_pct = float(player.get('starts_pct', 0.8) or 0.8)
        
        # Defensive action rates (estimated from available data)
        tackles_per90 = float(player.get('tackles_per90', 0) or 0)
        interceptions_per90 = float(player.get('interceptions_per90', 0) or 0)
        clearances_per90 = float(player.get('clearances_per90', 0) or 0)
        
        # Fallback: estimate CBIT from clean sheets for defenders
        if position in ('GKP', 'DEF') and tackles_per90 == 0:
            cs_ratio = float(player.get('clean_sheets', 0) or 0) / max(float(player.get('starts', 1) or 1), 1)
            tackles_per90 = 3.0 if cs_ratio > 0.3 else 2.0
            interceptions_per90 = 2.5 if cs_ratio > 0.3 else 1.5
            clearances_per90 = 4.0 if position == 'DEF' else 1.0
        
        injury_status = player.get('status', 'a')
        
        # Use granular chance_of_playing if available, else map from status
        cop_raw = player.get('chance_of_playing_next_round', None)
        if cop_raw is not None and not (isinstance(cop_raw, float) and np.isnan(cop_raw)):
            cop = float(cop_raw)
        else:
            injury_cop = {"a": 100, "d": 50, "i": 10, "s": 0, "n": 0, "u": 0}
            cop = injury_cop.get(injury_status, 80)
        
        # Opponent stats
        opp_xga = float(opponent_stats.get('xGA_per90', self._league_avg_xga) or self._league_avg_xga)
        opp_xg = float(opponent_stats.get('xG_per90', self._league_avg_xg) or self._league_avg_xg)
        team_def_quality = float(player.get('team_xga_ratio', 1.0) or 1.0)
        
        # ── Appearance Points ──
        p_plays, p_60_plus = self.calculate_appearance_probability(
            minutes_per_game, starts_pct, cop
        )
        xp_appearance = p_plays * (p_60_plus * 2 + (1 - p_60_plus) * 1)
        
        # ── Attacking Points (Goals) ──
        lambda_attack = self.calculate_attack_lambda(
            xg_per90, opp_xga, is_home, minutes_per_game
        )
        xp_goals, goal_probs = self.poisson_expected_points(
            lambda_attack, GOAL_PTS.get(position, 4), MAX_K_GOALS
        )
        
        # ── Creative Points (Assists) ──
        lambda_creative = self.calculate_creative_lambda(
            xa_per90, opp_xg, is_home, minutes_per_game
        )
        xp_assists, assist_probs = self.poisson_expected_points(
            lambda_creative, ASSIST_PTS, MAX_K_ASSISTS
        )
        
        # ── Clean Sheet Points ──
        cs_pts = CS_PTS.get(position, 0)
        if cs_pts > 0:
            p_cs = self.calculate_clean_sheet_probability(
                opp_xg, is_home, team_def_quality
            )
            xp_cs = p_cs * cs_pts * p_plays * p_60_plus  # Must play 60+ for CS
        else:
            xp_cs = 0.0
            p_cs = 0.0
        
        # ── Goals Conceded Penalty ──
        xp_gc_penalty = self.calculate_goals_conceded_penalty(
            opp_xg, is_home, team_def_quality, position
        ) * p_plays * p_60_plus
        
        # ── CBIT Bonus (2025/26) ──
        p_cbit = self.calculate_cbit_probability(
            tackles_per90, interceptions_per90, clearances_per90,
            position, minutes_per_game
        )
        xp_cbit = p_cbit * CBIT_BONUS * p_plays
        
        # ── Bonus Points ──
        xp_bonus = self.calculate_bonus_expected(
            lambda_attack, lambda_creative, position
        ) * p_plays
        
        # ── Total Expected Points ──
        total_xp = (
            xp_appearance +
            xp_goals * p_plays +
            xp_assists * p_plays +
            xp_cs +
            xp_gc_penalty +
            xp_cbit +
            xp_bonus
        )
        
        return {
            'xp_total': round(total_xp, 2),
            'xp_appearance': round(xp_appearance, 2),
            'xp_goals': round(xp_goals * p_plays, 2),
            'xp_assists': round(xp_assists * p_plays, 2),
            'xp_cs': round(xp_cs, 2),
            'xp_gc_penalty': round(xp_gc_penalty, 2),
            'xp_cbit': round(xp_cbit, 2),
            'xp_bonus': round(xp_bonus, 2),
            'lambda_attack': round(lambda_attack, 3),
            'lambda_creative': round(lambda_creative, 3),
            'p_cs': round(p_cs, 3),
            'p_cbit': round(p_cbit, 3),
            'p_plays': round(p_plays, 3),
            'goal_probs': goal_probs.tolist(),
            'assist_probs': assist_probs.tolist(),
        }
    
    def calculate_gameweek_xp(
        self,
        player: Dict,
        fixtures: List[Dict],
        team_stats_map: Dict[int, Dict]
    ) -> Dict[str, float]:
        """
        Calculate expected points for a gameweek (handles DGW).
        
        Sums independent Poisson outcomes for each fixture.
        
        Args:
            player: Player data dict
            fixtures: List of fixture dicts for this GW
            team_stats_map: Map of team_id -> team stats
        
        Returns:
            Combined expected points for the gameweek
        """
        if not fixtures:
            return {'xp_total': 0.0}
        
        player_team_id = player.get('team', 0)
        
        total_breakdown = {
            'xp_total': 0.0, 'xp_appearance': 0.0, 'xp_goals': 0.0,
            'xp_assists': 0.0, 'xp_cs': 0.0, 'xp_gc_penalty': 0.0,
            'xp_cbit': 0.0, 'xp_bonus': 0.0, 'fixture_count': len(fixtures)
        }
        
        for fixture in fixtures:
            is_home = fixture.get('team_h') == player_team_id
            opponent_id = fixture.get('team_a') if is_home else fixture.get('team_h')
            opponent_stats = team_stats_map.get(opponent_id, {})
            
            fx_xp = self.calculate_single_fixture_xp(player, opponent_stats, is_home)
            
            # Sum components
            for key in total_breakdown:
                if key in fx_xp and key != 'fixture_count':
                    total_breakdown[key] += fx_xp[key]
        
        return total_breakdown


def build_team_stats_from_understat(ustat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level Understat data to team-level stats.
    
    NOTE: This is a *fallback* used only when real team-level data from
    ``build_team_stats_df`` (in understat_api.py) is unavailable.
    Player-level aggregation cannot produce xGA (goals conceded) so
    we approximate using league-average symmetry.
    
    Returns DataFrame with: team_name, xG_for_per90, xGA_per90
    """
    if ustat_df.empty:
        return pd.DataFrame()
    
    team_stats = ustat_df.groupby('team_title').agg({
        'xG': 'sum',
        'xA': 'sum',
        'games': 'max',  # Use max as proxy for team games
    }).reset_index()
    
    team_stats['xG_for_per90'] = team_stats['xG'] / team_stats['games'].clip(lower=1)
    
    # xGA approximation: each team's xGA = league total xG / num_teams
    # This is symmetric – every goal expected *for* one team is expected *against* another
    total_xg = team_stats['xG'].sum()
    num_teams = len(team_stats)
    avg_matches = team_stats['games'].mean() if num_teams > 0 else 1
    if num_teams > 0 and avg_matches > 0:
        team_stats['xGA_per90'] = (total_xg / num_teams) / avg_matches
    else:
        team_stats['xGA_per90'] = LEAGUE_AVG_XGA_PER_MATCH
    
    return team_stats.rename(columns={'team_title': 'team_name'})


def calculate_poisson_ep_for_dataframe(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    current_gw: int,
    team_stats: Optional[pd.DataFrame] = None,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Main entry point: Calculate Poisson-based EP for all players.
    
    Uses **real Understat team-level xGA** when available (passed as
    ``team_stats``).  Falls back to the old FDR proxy only when
    Understat team data is missing.
    
    DGW/Multi-GW is handled properly by iterating each fixture independently
    through ``calculate_single_fixture_xp()``.
    
    Args:
        df: Player DataFrame with xG/xA stats
        fixtures_df: Fixtures DataFrame
        current_gw: Current gameweek number
        team_stats: DataFrame from ``build_team_stats_df()`` with real
                    per-match xGA / xG columns, or None for fallback.
        horizon: Number of gameweeks to look ahead (summing all fixtures)
    
    Returns:
        DataFrame with new expected_points_poisson column and breakdown
    """
    engine = PoissonEPEngine(team_stats)
    
    # ── Build FPL team_id → opponent xGA/xG map using Understat data ──
    # Maps each FPL team_id to {xGA_per90, xG_per90} for the real Understat stats
    team_xga_map: Dict[int, Dict[str, float]] = {}
    fdr_fallback_teams: List[str] = []   # track which teams lacked real data
    
    if team_stats is not None and not team_stats.empty and 'understat_team' in team_stats.columns:
        # Validate required columns exist
        required_cols = {'understat_team', 'xGA_per90', 'xG_for_per90'}
        missing_cols = required_cols - set(team_stats.columns)
        if missing_cols:
            logger.warning(
                "team_stats DataFrame missing columns %s; "
                "falling back to FDR for all teams", missing_cols
            )
        else:
            # Reverse the FPL->Understat name mapping
            try:
                from understat_api import TEAM_NAME_MAP
                ustat_to_fpl = {v: k for k, v in TEAM_NAME_MAP.items()}
            except ImportError:
                logger.warning("Could not import TEAM_NAME_MAP from understat_api")
                ustat_to_fpl = {}
            
            # Build Understat team name -> stats lookup with NaN guards
            ustat_stats: Dict[str, Dict[str, float]] = {}
            for _, row in team_stats.iterrows():
                xga_val = float(row.get('xGA_per90', 0) or 0)
                xg_val = float(row.get('xG_for_per90', 0) or 0)
                # Skip teams with invalid/zero stats
                if xga_val <= 0 or xg_val <= 0:
                    logger.debug(
                        "Skipping team '%s': invalid stats (xGA=%.3f, xG=%.3f)",
                        row['understat_team'], xga_val, xg_val
                    )
                    continue
                ustat_stats[row['understat_team']] = {
                    'xGA_per90': xga_val,
                    'xG_per90': xg_val,
                }
            
            # Map FPL team_name -> stats via reverse name mapping
            # Also try case-insensitive fallback for promoted teams not in MAP
            fpl_name_stats: Dict[str, Dict[str, float]] = {}
            ustat_names_lower = {n.lower(): n for n in ustat_stats}
            for ustat_name, stats in ustat_stats.items():
                fpl_name = ustat_to_fpl.get(ustat_name, ustat_name)
                fpl_name_stats[fpl_name] = stats
            
            # Finally map FPL team_id -> stats using the df's team/team_name columns
            if 'team_name' in df.columns:
                name_to_id = (
                    df[['team', 'team_name']]
                    .drop_duplicates()
                    .set_index('team_name')['team']
                    .to_dict()
                )
                for name, tid in name_to_id.items():
                    if name in fpl_name_stats:
                        team_xga_map[tid] = fpl_name_stats[name]
                    else:
                        # Fuzzy fallback: try case-insensitive substring match
                        # for promoted teams whose names may differ slightly
                        name_lower = name.lower()
                        matched = False
                        for ustat_lower, ustat_orig in ustat_names_lower.items():
                            if (name_lower in ustat_lower
                                    or ustat_lower in name_lower):
                                team_xga_map[tid] = ustat_stats[ustat_orig]
                                logger.info(
                                    "Fuzzy-matched FPL '%s' -> Understat '%s'",
                                    name, ustat_orig
                                )
                                matched = True
                                break
                        if not matched:
                            fdr_fallback_teams.append(name)
            
            total_teams = df['team'].nunique()
            mapped_count = len(team_xga_map)
            logger.info(
                "Poisson engine: mapped real xGA for %d / %d teams",
                mapped_count, total_teams,
            )
            if fdr_fallback_teams:
                logger.warning(
                    "Poisson engine: FDR fallback for %d team(s): %s",
                    len(fdr_fallback_teams), ', '.join(fdr_fallback_teams),
                )
    else:
        reason = (
            "team_stats is None" if team_stats is None
            else "team_stats is empty" if team_stats.empty
            else "missing 'understat_team' column"
        )
        logger.warning(
            "Poisson engine: no Understat team data (%s); "
            "using FDR proxy for ALL teams", reason,
        )
    
    # ── Get fixtures per team for the horizon ──
    # Select all fixtures from next GW up to next GW + horizon - 1
    horizon_fixtures = fixtures_df[
        (fixtures_df['event'] >= current_gw + 1) & 
        (fixtures_df['event'] < current_gw + 1 + horizon)
    ]
    
    # Build id→name lookup for logging FDR fallback teams
    id_to_name: Dict[int, str] = {}
    if 'team_name' in df.columns:
        id_to_name = (
            df[['team', 'team_name']]
            .drop_duplicates()
            .set_index('team')['team_name']
            .to_dict()
        )
    
    fdr_fixture_count = 0
    real_fixture_count = 0
    
    team_fixture_list: Dict[int, List[Dict]] = {}  # team_id -> list of fixture dicts
    for team_id in df['team'].unique():
        team_fixtures = horizon_fixtures[
            (horizon_fixtures['team_h'] == team_id) |
            (horizon_fixtures['team_a'] == team_id)
        ]
        
        fixtures_list: List[Dict] = []
        for _, fx in team_fixtures.iterrows():
            is_home = fx['team_h'] == team_id
            opp_id = fx['team_a'] if is_home else fx['team_h']
            
            # Use REAL Understat xGA; fall back to FDR proxy only if unavailable
            opp_real = team_xga_map.get(opp_id)
            if opp_real is not None:
                opp_xga = opp_real['xGA_per90']
                opp_xg = opp_real['xG_per90']
                real_fixture_count += 1
            else:
                # FDR fallback (used only when Understat team data is missing)
                fdr = (
                    fx.get('team_h_difficulty', 3)
                    if not is_home
                    else fx.get('team_a_difficulty', 3)
                )
                xga_mult = 0.5 + (fdr / 5) * 1.0
                xg_mult = 0.5 + ((6 - fdr) / 5) * 1.0
                opp_xga = LEAGUE_AVG_XGA_PER_MATCH * xga_mult
                opp_xg = LEAGUE_AVG_XG_PER_MATCH * xg_mult
                fdr_fixture_count += 1
                opp_name = id_to_name.get(opp_id, f"team_{opp_id}")
                logger.debug(
                    "FDR fallback for opponent %s (id=%d): FDR=%d -> "
                    "xGA=%.3f, xG=%.3f",
                    opp_name, opp_id, fdr, opp_xga, opp_xg,
                )
            
            fixtures_list.append({
                'is_home': is_home,
                'opp_xGA_per90': opp_xga,
                'opp_xG_per90': opp_xg,
                'opponent_id': opp_id,
            })
        
        # No fixtures found in horizon?
        if not fixtures_list:
            fixtures_list = []
        
        team_fixture_list[team_id] = fixtures_list
    
    total_fixtures = real_fixture_count + fdr_fixture_count
    if total_fixtures > 0:
        logger.info(
            "Poisson fixtures: %d real xGA, %d FDR fallback (%.0f%% real)",
            real_fixture_count, fdr_fixture_count,
            real_fixture_count / total_fixtures * 100,
        )
    
    # ── Calculate Poisson EP for each player ──
    results = []
    for _, player in df.iterrows():
        team_id = player.get('team', 0)
        player_fixtures = team_fixture_list.get(team_id, [])
        
        player_dict = player.to_dict()
        
        # ── Iterate EACH fixture independently (proper DGW handling) ──
        total_xp = 0.0
        total_xp_goals = 0.0
        total_xp_assists = 0.0
        total_xp_cs = 0.0
        total_xp_bonus = 0.0
        total_p_cs = 0.0
        
        if player_fixtures:
            first_opp_xga = player_fixtures[0].get('opp_xGA_per90', LEAGUE_AVG_XGA_PER_MATCH)
            first_opp_xg = player_fixtures[0].get('opp_xG_per90', LEAGUE_AVG_XG_PER_MATCH)
        else:
            first_opp_xga = LEAGUE_AVG_XGA_PER_MATCH
            first_opp_xg = LEAGUE_AVG_XG_PER_MATCH
        
        for fx in player_fixtures:
            opponent_dict = {
                'xGA_per90': fx['opp_xGA_per90'],
                'xG_per90': fx['opp_xG_per90'],
            }
            fx_result = engine.calculate_single_fixture_xp(
                player_dict, opponent_dict, fx['is_home']
            )
            total_xp += fx_result.get('xp_total', 0)
            total_xp_goals += fx_result.get('xp_goals', 0)
            total_xp_assists += fx_result.get('xp_assists', 0)
            total_xp_cs += fx_result.get('xp_cs', 0)
            total_xp_bonus += fx_result.get('xp_bonus', 0)
            total_p_cs += fx_result.get('p_cs', 0)
        
        # Average p_cs across fixtures (for CBIT calculations)
        avg_p_cs = total_p_cs / len(player_fixtures) if player_fixtures else 0.0
        
        results.append({
            'player_id': player.get('id', 0),
            'xp_total': round(total_xp, 2),
            'xp_goals': round(total_xp_goals, 2),
            'xp_assists': round(total_xp_assists, 2),
            'xp_cs': round(total_xp_cs, 2),
            'xp_bonus': round(total_xp_bonus, 2),
            'p_cs': round(avg_p_cs, 3),
            'fixture_count': len(player_fixtures),
            'opp_xGA_per90': round(first_opp_xga, 3),
            'opp_xG_per90': round(first_opp_xg, 3),
        })
    
    # ── Merge results back to DataFrame ──
    ep_df = pd.DataFrame(results)
    result_df = df.copy()
    
    if not ep_df.empty and 'player_id' in ep_df.columns:
        ep_df = ep_df.set_index('player_id')
        for col in ['xp_total', 'xp_goals', 'xp_assists', 'xp_cs', 'xp_bonus',
                     'lambda_attack', 'lambda_creative', 'p_cs', 'p_cbit',
                     'opp_xGA_per90', 'opp_xG_per90']:
            if col in ep_df.columns:
                result_df[f'poisson_{col}'] = result_df['id'].map(ep_df[col]).fillna(0)
        
        result_df['expected_points_poisson'] = result_df['id'].map(
            ep_df['xp_total']
        ).fillna(0)
    
    return result_df
