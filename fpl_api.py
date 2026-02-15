"""
FPL API Module - Data fetching and processing for the 2025/26 FPL Strategy Engine.
Handles all interactions with the official FPL API and The Odds API.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

# FPL API Base URLs
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# 2025/26 Season Constants
CBIT_BONUS_THRESHOLD = 10  # CBIT points needed for +2 bonus
CBIT_BONUS_POINTS = 2
MAX_FREE_TRANSFERS = 5  # New 2025/26 rollover cap
CAPTAIN_MULTIPLIER = 1.25  # New 2025/26 captaincy boost


class FPLDataFetcher:
    """Handles all FPL API data fetching operations."""
    
    def __init__(self, odds_api_key: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-Strategy-Engine/2.0',
            'Accept-Charset': 'utf-8'
        })
        self.odds_api_key = odds_api_key
        self._cache = {}
        self._cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
    def _get_cached(self, key: str) -> Optional[dict]:
        """Retrieve cached data if not expired."""
        if key in self._cache:
            if time.time() < self._cache_expiry.get(key, 0):
                return self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: dict):
        """Cache data with expiration."""
        self._cache[key] = data
        self._cache_expiry[key] = time.time() + self.cache_duration
    
    def get_bootstrap_static(self) -> Dict:
        """
        Fetch general player/team info from bootstrap-static endpoint.
        Returns all players, teams, events (gameweeks), and game settings.
        """
        cached = self._get_cached('bootstrap')
        if cached:
            return cached
            
        try:
            response = self.session.get(f"{FPL_BASE_URL}/bootstrap-static/")
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            self._set_cache('bootstrap', data)
            return data
        except requests.RequestException as e:
            raise FPLAPIError(f"Failed to fetch bootstrap data: {e}")
    
    def get_player_summary(self, player_id: int) -> Dict:
        """
        Fetch detailed player history and fixtures.
        Returns past season data, current season history, and upcoming fixtures.
        """
        cache_key = f'player_{player_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            response = self.session.get(f"{FPL_BASE_URL}/element-summary/{player_id}/")
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            self._set_cache(cache_key, data)
            return data
        except requests.RequestException as e:
            raise FPLAPIError(f"Failed to fetch player {player_id} summary: {e}")
    
    def get_team_picks(self, team_id: int, gameweek: int) -> Dict:
        """
        Fetch a user's squad for a specific gameweek.
        Returns picks, active chip, automatic substitutions, and entry history.
        """
        cache_key = f'team_{team_id}_gw_{gameweek}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            response = self.session.get(
                f"{FPL_BASE_URL}/entry/{team_id}/event/{gameweek}/picks/"
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            self._set_cache(cache_key, data)
            return data
        except requests.RequestException as e:
            raise FPLAPIError(f"Failed to fetch team {team_id} GW{gameweek} picks: {e}")
    
    def get_team_history(self, team_id: int) -> Dict:
        """Fetch complete history for a team."""
        cache_key = f'team_history_{team_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            response = self.session.get(f"{FPL_BASE_URL}/entry/{team_id}/history/")
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            self._set_cache(cache_key, data)
            return data
        except requests.RequestException as e:
            raise FPLAPIError(f"Failed to fetch team {team_id} history: {e}")
    
    def get_fixtures(self) -> List[Dict]:
        """Fetch all fixtures for the season."""
        cached = self._get_cached('fixtures')
        if cached:
            return cached
            
        try:
            response = self.session.get(f"{FPL_BASE_URL}/fixtures/")
            response.raise_for_status()
            response.encoding = 'utf-8'
            data = response.json()
            self._set_cache('fixtures', data)
            return data
        except requests.RequestException as e:
            raise FPLAPIError(f"Failed to fetch fixtures: {e}")
    
    def get_current_gameweek(self) -> int:
        """Determine the current gameweek."""
        bootstrap = self.get_bootstrap_static()
        events = bootstrap.get('events', [])
        for event in events:
            if event.get('is_current'):
                return event['id']
        # Return next gameweek if none is current
        for event in events:
            if event.get('is_next'):
                return event['id']
        return 1


class OddsDataFetcher:
    """Handles betting odds data for probability calculations."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._dummy_data = self._generate_dummy_odds()
    
    def _generate_dummy_odds(self) -> Dict[str, Dict]:
        """Generate dummy odds data as fallback."""
        return {
            'anytime_goalscorer': {
                'Haaland': 1.50, 'Salah': 2.10, 'Watkins': 3.00,
                'Isak': 2.80, 'Palmer': 2.50, 'Saka': 3.20,
                'Son': 2.90, 'Gordon': 3.50, 'Solanke': 3.80,
                'Darwin Núñez': 2.60
            },
            'clean_sheet': {
                'Arsenal': 2.20, 'Man City': 2.00, 'Liverpool': 2.10,
                'Chelsea': 2.80, 'Tottenham': 3.20, 'Newcastle': 2.50,
                'Man Utd': 3.00, 'Aston Villa': 2.90, 'Brighton': 3.10,
                'West Ham': 3.50, 'Fulham': 3.80, 'Brentford': 3.60,
                'Crystal Palace': 3.40, 'Everton': 3.70, 'Bournemouth': 3.90,
                'Wolves': 3.50, 'Nottm Forest': 3.30, 'Leicester': 4.00,
                'Ipswich': 4.50, 'Southampton': 4.20
            }
        }
    
    def get_goalscorer_odds(self, player_name: str) -> float:
        """Get anytime goalscorer odds for a player."""
        if self.api_key:
            # Would call actual API here
            pass
        return self._dummy_data['anytime_goalscorer'].get(player_name, 5.0)
    
    def get_clean_sheet_odds(self, team_name: str) -> float:
        """Get clean sheet odds for a team."""
        if self.api_key:
            # Would call actual API here
            pass
        return self._dummy_data['clean_sheet'].get(team_name, 3.5)
    
    def odds_to_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / decimal_odds if decimal_odds > 0 else 0.0


class FPLDataProcessor:
    """Processes and engineers features from raw FPL data."""
    
    def __init__(self, fetcher: FPLDataFetcher, odds_fetcher: Optional[OddsDataFetcher] = None):
        self.fetcher = fetcher
        self.odds_fetcher = odds_fetcher or OddsDataFetcher()
        self._bootstrap_data = None
        self._players_df = None
        self._teams_df = None
        self._fixtures_df = None
    
    @property
    def bootstrap_data(self) -> Dict:
        if self._bootstrap_data is None:
            self._bootstrap_data = self.fetcher.get_bootstrap_static()
        return self._bootstrap_data
    
    @property
    def players_df(self) -> pd.DataFrame:
        if self._players_df is None:
            self._players_df = self._build_players_dataframe()
        return self._players_df
    
    @property
    def teams_df(self) -> pd.DataFrame:
        if self._teams_df is None:
            self._teams_df = pd.DataFrame(self.bootstrap_data['teams'])
        return self._teams_df
    
    @property
    def fixtures_df(self) -> pd.DataFrame:
        if self._fixtures_df is None:
            fixtures = self.fetcher.get_fixtures()
            self._fixtures_df = pd.DataFrame(fixtures)
        return self._fixtures_df
    
    def _build_players_dataframe(self) -> pd.DataFrame:
        """Build comprehensive player DataFrame with all raw stats."""
        players = self.bootstrap_data['elements']
        df = pd.DataFrame(players)
        
        # Convert price to actual value (API returns tenths, e.g. 106 = £10.6m)
        df['now_cost'] = (pd.to_numeric(df['now_cost'], errors='coerce').fillna(50) / 10.0).round(1)
        
        # Convert all string-numeric columns the API returns as strings
        _str_numeric_cols = [
            'ep_next', 'ep_this', 'form', 'points_per_game',
            'selected_by_percent', 'value_form', 'value_season',
            'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'expected_goals_per_90', 'expected_assists_per_90',
            'expected_goal_involvements_per_90',
            'expected_goals_conceded_per_90',
        ]
        for col in _str_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Add team names
        team_map = {t['id']: t['name'] for t in self.bootstrap_data['teams']}
        df['team_name'] = df['team'].map(team_map)
        
        # Add position names
        position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position'] = df['element_type'].map(position_map)
        
        # Calculate minutes per game
        df['minutes_per_game'] = np.where(
            df['starts'] > 0,
            df['minutes'] / df['starts'],
            0
        )
        
        # Calculate starts percentage (for Poisson appearance model)
        # Compare each player's starts to the most-started player on their team
        # This gives a true "nailedness" ratio (1.0 = ever-present, 0.0 = never)
        team_max_starts = df.groupby('team')['starts'].transform('max').clip(lower=1)
        df['starts_pct'] = np.where(
            df['starts'] > 0,
            np.minimum(df['starts'] / team_max_starts, 1.0),
            0.05  # Small baseline for non-starters
        )
        
        return df
    
    def calculate_cbit_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced CBIT metrics (Clearances, Blocks, Interceptions, Tackles).
        2025/26 "DefCon" scoring: DEF threshold=10, others=12 for +2 bonus.
        
        New metrics:
        - cbit_aa90: Average (estimated) actions per 90 minutes
        - cbit_dtt: Distance to threshold (negative = below, positive = above)
        - cbit_prob: Poisson probability of hitting CBIT threshold (0-1)
        - cbit_hit_rate: Season % estimate of games hitting threshold
        - cbit_floor: Defensive floor score = xCS × 4 + P(CBIT) × 2
        - cbit_matchup: Opponent givingness factor (higher = more actions expected)
        - cbit_score: Composite CBIT value metric for display
        """
        from scipy.stats import poisson
        df = df.copy()
        
        # Position-based thresholds
        threshold_map = {'GKP': 12, 'DEF': 10, 'MID': 12, 'FWD': 12}
        df['cbit_threshold'] = df['position'].map(threshold_map).fillna(12).astype(int)
        
        # Games played
        df['_games'] = (df['minutes'].fillna(0) / 90).clip(lower=0.1)
        
        # ── Estimate Average Actions per 90 (AA90) ──
        # FPL API doesn't provide tackles/interceptions/clearances/blocks directly
        # Use position-based baseline + clean sheet rate as defensive solidity proxy
        cs_rate = (df['clean_sheets'].fillna(0) / df['_games']).clip(0, 1)
        
        # Base defensive action rates by position (empirical estimates)
        base_aa90 = {
            'GKP': 4.0,   # GKs get fewer field actions but more saves
            'DEF': 12.0,  # CBs average ~12 actions, WBs slightly less
            'MID': 7.0,   # CDMs higher, CAMs lower
            'FWD': 3.0,   # Minimal defensive work
        }
        df['_base_aa90'] = df['position'].map(base_aa90).fillna(5.0)
        
        # Adjust by CS rate (good defenders do more defensive work in games they keep CS)
        # Also adjust by minutes consistency (rotation = less reliable)
        mins_consistency = (df['minutes'] / (df['_games'] * 90).clip(lower=1)).clip(0, 1)
        
        # For DEF: high CS rate correlates with solid defensive actions
        # Scaling: 0.5 CS rate → +20% actions, 0.0 CS rate → -10%
        cs_adjustment = np.where(
            df['position'].isin(['GKP', 'DEF']),
            1.0 + (cs_rate - 0.3) * 0.5,  # 30% CS is baseline
            1.0  # No adjustment for MID/FWD
        )
        
        df['cbit_aa90'] = (df['_base_aa90'] * cs_adjustment * mins_consistency).round(1)
        
        # ── Distance to Threshold (DTT) ──
        # Positive = exceeds threshold, Negative = below threshold
        # Players near 0 are "high variance" CBIT assets
        df['cbit_dtt'] = (df['cbit_aa90'] - df['cbit_threshold']).round(1)
        
        # ── CBIT Probability (Poisson) ──
        # P(actions >= threshold) = 1 - CDF(threshold - 1)
        def calc_cbit_prob(aa90, threshold):
            if aa90 <= 0:
                return 0.0
            return 1 - poisson.cdf(threshold - 1, aa90)
        
        df['cbit_prob'] = df.apply(
            lambda r: calc_cbit_prob(r['cbit_aa90'], r['cbit_threshold']), axis=1
        ).clip(0, 1).round(3)
        
        # ── Hit Rate (Season %) ──
        # Combination of probability and minutes consistency
        df['cbit_hit_rate'] = (df['cbit_prob'] * mins_consistency).round(2)
        
        # ── Clean Sheet Probability (xCS) ──
        # Prefer Poisson p_cs from fixture-based calculation, fallback to CS rate
        if 'poisson_p_cs' in df.columns:
            p_cs = pd.to_numeric(df['poisson_p_cs'], errors='coerce').fillna(cs_rate)
            df['_xcs'] = p_cs.clip(0, 0.7)  # Cap at 70%
        else:
            df['_xcs'] = cs_rate.clip(0, 0.6)  # Cap at 60%
        
        # ── Defensive Floor Score ──
        # xP_floor = xCS × 4pts + P(CBIT) × 2pts
        cs_pts = df['position'].map({'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}).fillna(0)
        df['cbit_floor'] = (df['_xcs'] * cs_pts + df['cbit_prob'] * CBIT_BONUS_POINTS).round(2)
        
        # ── Opponent Givingness / Matchup Factor ──
        # Teams with high xG create more defensive action opportunities
        # High xG opponents = more shots to block/clear/tackle
        if 'poisson_opp_xG_per90' in df.columns:
            opp_xg = pd.to_numeric(df['poisson_opp_xG_per90'], errors='coerce').fillna(1.35)
            league_avg_xg = opp_xg[opp_xg > 0].mean() or 1.35
            df['cbit_matchup'] = (opp_xg / league_avg_xg).clip(0.5, 2.0).round(2)
        else:
            df['cbit_matchup'] = 1.0
        
        # Adjust AA90 and probability by matchup
        df['cbit_aa90_adj'] = (df['cbit_aa90'] * df['cbit_matchup']).round(1)
        df['cbit_prob_adj'] = df.apply(
            lambda r: calc_cbit_prob(r['cbit_aa90_adj'], r['cbit_threshold']), axis=1
        ).clip(0, 1).round(3)
        
        # ── Composite CBIT Score (for display) ──
        # Weighted: 40% prob, 30% floor, 20% consistency, 10% matchup bonus
        df['cbit_score'] = (
            df['cbit_prob'] * 0.4 * 10 +
            df['cbit_floor'] * 0.3 +
            df['cbit_hit_rate'] * 0.2 * 10 +
            (df['cbit_matchup'] - 1) * 0.1 * 5
        ).round(2)
        
        # ── Legacy compatibility ──
        df['cbit_propensity'] = df['cbit_prob']
        df['cbit_bonus_expected'] = (df['cbit_prob'] * CBIT_BONUS_POINTS).round(2)
        df['cbit_safety_floor'] = df['cbit_floor']
        
        # Clean up temp columns
        df.drop(columns=['_games', '_base_aa90', '_xcs'], inplace=True, errors='ignore')
        
        return df
    
    def calculate_fixture_difficulty(self, player_id: int, weeks_ahead: int = 5) -> List[float]:
        """
        Calculate fixture difficulty ratings for upcoming matches.
        Returns list of FDR values for the next N gameweeks.
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return [3.0] * weeks_ahead  # Default medium difficulty
        
        team_id = player_row.iloc[0]['team']
        
        # Use team-level cache to avoid repeated per-player lookups
        return self._get_team_fdrs(team_id, weeks_ahead)
    
    def _get_team_fdrs(self, team_id: int, weeks_ahead: int = 5) -> List[float]:
        """Get FDR list for a team, with caching."""
        cache_key = (team_id, weeks_ahead)
        if not hasattr(self, '_fdr_cache'):
            self._fdr_cache = {}
        if cache_key in self._fdr_cache:
            return self._fdr_cache[cache_key]
        
        current_gw = self.fetcher.get_current_gameweek()
        
        fixtures = self.fixtures_df[
            (self.fixtures_df['event'] >= current_gw) &
            (self.fixtures_df['event'] < current_gw + weeks_ahead) &
            ((self.fixtures_df['team_h'] == team_id) | (self.fixtures_df['team_a'] == team_id))
        ].sort_values('event')
        
        fdrs = []
        for _, fixture in fixtures.iterrows():
            if fixture['team_h'] == team_id:
                fdrs.append(fixture.get('team_h_difficulty', 3))
            else:
                fdrs.append(fixture.get('team_a_difficulty', 3))
        
        while len(fdrs) < weeks_ahead:
            fdrs.append(3.0)
        
        result = fdrs[:weeks_ahead]
        self._fdr_cache[cache_key] = result
        return result
    
    def calculate_fixture_ease(self, player_id: int, weeks_ahead: int = 5) -> float:
        """
        Calculate aggregate fixture ease score.
        Higher = easier fixtures ahead.
        """
        fdrs = self.calculate_fixture_difficulty(player_id, weeks_ahead)
        if not fdrs:
            return 0.5
        
        # Invert FDR (5 = hardest, 1 = easiest) to ease score
        # Apply decay weights - nearer fixtures matter more
        decay_weights = [0.95 ** i for i in range(len(fdrs))]
        weighted_ease = sum((6 - fdr) * w for fdr, w in zip(fdrs, decay_weights))
        max_ease = sum(5 * w for w in decay_weights)
        
        return weighted_ease / max_ease if max_ease > 0 else 0.5
    
    def detect_blanks(self, player_id: int, weeks_ahead: int = 5) -> List[int]:
        """
        Detect blank gameweeks (no fixtures) for a player.
        Returns list of blank gameweek numbers.
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return []
        
        team_id = player_row.iloc[0]['team']
        current_gw = self.fetcher.get_current_gameweek()
        
        blanks = []
        for gw in range(current_gw, current_gw + weeks_ahead):
            has_fixture = not self.fixtures_df[
                (self.fixtures_df['event'] == gw) &
                ((self.fixtures_df['team_h'] == team_id) | (self.fixtures_df['team_a'] == team_id))
            ].empty
            
            if not has_fixture:
                blanks.append(gw)
        
        return blanks
    
    def calculate_expected_points(self, player_id: int, weeks_ahead: int = 5) -> Tuple[float, List[float]]:
        """
        Calculate expected points using multiple factors.
        Returns (total_ep, weekly_ep_list).
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return 0.0, [0.0] * weeks_ahead
        
        player = player_row.iloc[0]
        
        # Base expected points from FPL's own EP calculation
        base_ep = float(player.get('ep_next', 0) or 0)
        
        # Get fixture ease multiplier
        fixture_ease = self.calculate_fixture_ease(player_id, weeks_ahead)
        
        # Get blanks
        blanks = self.detect_blanks(player_id, weeks_ahead)
        current_gw = self.fetcher.get_current_gameweek()
        
        # Calculate weekly EP with decay
        weekly_eps = []
        for i in range(weeks_ahead):
            gw = current_gw + i
            if gw in blanks:
                weekly_eps.append(0.0)  # Ghost points - blank week
            else:
                # Apply decay factor (further away = less certain)
                decay = 0.95 ** i
                fdrs = self.calculate_fixture_difficulty(player_id, weeks_ahead)
                fdr_multiplier = (6 - fdrs[i]) / 5 if i < len(fdrs) else 0.6
                weekly_eps.append(base_ep * decay * fdr_multiplier)
        
        # Add CBIT bonus for defenders (if cbit_bonus_expected column exists)
        if player['position'] == 'DEF' and 'cbit_bonus_expected' in self.players_df.columns:
            cbit_bonus = self.players_df.loc[
                self.players_df['id'] == player_id, 'cbit_bonus_expected'
            ].values
            if len(cbit_bonus) > 0 and not np.isnan(cbit_bonus[0]):
                weekly_eps = [ep + cbit_bonus[0] for ep in weekly_eps]
        
        return sum(weekly_eps), weekly_eps
    
    def calculate_differential_score(self, player_id: int, weeks_ahead: int = 5) -> float:
        """
        Calculate Advanced Differential Score using Net Rank Gain + ROI.
        
        Formula:
            EP_net = EP × (1 - EO/100) -- points gained vs field
            Diff_ROI = EP / (EO + 1) -- points per % ownership
            Score = EP_net × 0.5 + Diff_ROI × 100 × 0.3 + fixture_ease × 0.2
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return 0.0
        
        player = player_row.iloc[0]
        
        # Get EP
        total_ep, _ = self.calculate_expected_points(player_id, weeks_ahead)
        
        # Get effective ownership
        eo = float(player.get('selected_by_percent', 50) or 50)
        eo = max(eo, 0.1)
        
        # Get fixture ease
        fixture_ease = self.calculate_fixture_ease(player_id, weeks_ahead)
        
        # Net Rank Gain: points you gain vs field
        ep_net = total_ep * (1 - eo / 100)
        
        # Differential ROI: points per % ownership
        diff_roi = total_ep / (eo + 1)
        
        # Combined score
        differential = ep_net * 0.5 + diff_roi * 100 * 0.3 + fixture_ease * 0.2
        
        return differential
    
    def get_engineered_features_df(self, weeks_ahead: int = 5) -> pd.DataFrame:
        """
        Build comprehensive DataFrame with all engineered features.
        Uses vectorized operations per-team instead of per-player loops.
        """
        df = self.players_df.copy()
        
        # Calculate CBIT metrics (already vectorized)
        df = self.calculate_cbit_metrics(df)
        
        # --- Vectorized: precompute per-TEAM data (20 teams, not 600 players) ---
        current_gw = self.fetcher.get_current_gameweek()
        fixtures = self.fixtures_df
        
        # Build team-level lookups once
        team_ids = df['team'].unique()
        team_fdrs = {}        # team_id -> [fdr1, fdr2, ...]
        team_blanks = {}      # team_id -> set of blank gw numbers
        team_ease = {}        # team_id -> ease score
        
        decay_weights = [0.95 ** i for i in range(weeks_ahead)]
        max_ease_denom = sum(5 * w for w in decay_weights)
        
        for tid in team_ids:
            # FDR per gameweek
            team_fixtures = fixtures[
                (fixtures['event'] >= current_gw) &
                (fixtures['event'] < current_gw + weeks_ahead) &
                ((fixtures['team_h'] == tid) | (fixtures['team_a'] == tid))
            ].sort_values('event')
            
            fdrs = []
            fixture_gws = set()
            for _, fx in team_fixtures.iterrows():
                fixture_gws.add(fx['event'])
                if fx['team_h'] == tid:
                    fdrs.append(fx.get('team_h_difficulty', 3))
                else:
                    fdrs.append(fx.get('team_a_difficulty', 3))
            
            while len(fdrs) < weeks_ahead:
                fdrs.append(3.0)
            fdrs = fdrs[:weeks_ahead]
            team_fdrs[tid] = fdrs
            
            # Blanks
            blanks = set()
            for gw in range(current_gw, current_gw + weeks_ahead):
                if gw not in fixture_gws:
                    blanks.add(gw)
            team_blanks[tid] = blanks
            
            # Fixture ease (vectorized per team)
            weighted_ease = sum((6 - fdr) * w for fdr, w in zip(fdrs, decay_weights))
            team_ease[tid] = weighted_ease / max_ease_denom if max_ease_denom > 0 else 0.5
        
        # --- Vectorized: map team-level data to all players at once ---
        df['fixture_ease'] = df['team'].map(team_ease).fillna(0.5).astype(float)
        df['blank_count'] = df['team'].map(
            lambda tid: len(team_blanks.get(tid, set()))
        ).fillna(0).astype(int)
        
        # Base EP from FPL (single GW next-match prediction)
        if 'ep_next' in df.columns:
            df['ep_next_num'] = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0.0)
        else:
            df['ep_next_num'] = 0.0
        
        # Initialize expected_points from FPL data first
        df['expected_points'] = df['ep_next_num'].copy()
        
        # Import streamlit for session state access (handle all exceptions)
        st = None
        try:
            import streamlit as st
        except Exception:
            pass
        
        # ── Understat xG/xA enrichment ──
        try:
            from understat_api import enrich_with_understat
            df = enrich_with_understat(df)
            if st is not None:
                st.session_state['_understat_active'] = True
                # Log data source quality summary
                ustat_status = st.session_state.get('_understat_status', {})
                team_count = ustat_status.get('team_stats_count', 0)
                match_rate = ustat_status.get('player_match_rate', 0)
                errs = ustat_status.get('errors', [])
                import logging as _log
                _ulog = _log.getLogger(__name__)
                _ulog.info(
                    "Understat enrichment OK: %d team stats, %.0f%% player match rate",
                    team_count, match_rate * 100,
                )
                if errs:
                    _ulog.warning("Understat enrichment warnings: %s", '; '.join(errs))
        except Exception as e:
            import logging
            import traceback
            logging.getLogger(__name__).error(
                "Understat enrichment failed: %s\n%s", e, traceback.format_exc()
            )
            if st is not None:
                st.session_state['_understat_active'] = False
                st.session_state['_understat_status'] = {
                    'players_fetched': 0, 'teams_fetched': 0,
                    'team_stats_built': False, 'team_stats_count': 0,
                    'player_match_rate': 0.0,
                    'errors': [f"top-level: {e}"],
                }
        
        # ── Poisson-based Expected Points (professional-grade model) ──
        try:
            use_poisson = st.session_state.get('use_poisson_xp', True) if st is not None else True
            
            if use_poisson:
                from poisson_ep import calculate_poisson_ep_for_dataframe
                # Pass real Understat team stats if available
                team_stats = st.session_state.get('_understat_team_stats', None) if st is not None else None
                has_real_xga = (
                    team_stats is not None
                    and isinstance(team_stats, pd.DataFrame)
                    and not team_stats.empty
                )
                import logging as _plog
                _plogger = _plog.getLogger(__name__)
                if has_real_xga:
                    _plogger.info(
                        "Poisson EP: using real Understat team xGA (%d teams)",
                        len(team_stats),
                    )
                else:
                    _plogger.warning(
                        "Poisson EP: no Understat team stats available; "
                        "FDR proxy will be used for opponent strength"
                    )
                df = calculate_poisson_ep_for_dataframe(df, fixtures, current_gw, team_stats=team_stats)
                # Use Poisson EP as primary expected_points if available
                if 'expected_points_poisson' in df.columns:
                    # Blend: 70% Poisson model, 30% FPL baseline for stability
                    poisson_ep = pd.to_numeric(df['expected_points_poisson'], errors='coerce').fillna(0)
                    base_ep = pd.to_numeric(df['ep_next_num'], errors='coerce').fillna(0)
                    df['expected_points'] = (poisson_ep * 0.7 + base_ep * 0.3).clip(lower=0)
            else:
                # Use raw FPL ep_next
                df['expected_points'] = df['ep_next_num']
        except Exception as e:
            import logging
            import traceback as _tb
            logging.getLogger(__name__).error(
                "Poisson EP calculation failed: %s\n%s", e, _tb.format_exc()
            )
        
        # Ensure expected_points is numeric and exists
        if 'expected_points' not in df.columns:
            df['expected_points'] = df['ep_next_num']
        df['expected_points'] = pd.to_numeric(df['expected_points'], errors='coerce').fillna(df['ep_next_num'])
        
        # ── Multi-week planning EP (for horizon-based features) ──
        # This is used for differential calculations and planning, NOT for display
        def _calc_horizon_ep(row):
            tid = row['team']
            base_ep = row.get('expected_points', 0)  # Use single-GW EP with safe fallback
            if pd.isna(base_ep):
                base_ep = 0
            fdrs = team_fdrs.get(tid, [3.0] * weeks_ahead)
            blanks = team_blanks.get(tid, set())
            
            total = 0.0
            for i in range(weeks_ahead):
                gw = current_gw + i
                if gw in blanks:
                    continue
                decay = decay_weights[i]
                fdr_mult = (6 - fdrs[i]) / 5 if i < len(fdrs) else 0.6
                total += base_ep * decay * fdr_mult
            
            # CBIT bonus for defenders
            if row.get('position') == 'DEF':
                cbit_bonus = row.get('cbit_bonus_expected', 0)
                if pd.notna(cbit_bonus):
                    active_weeks = weeks_ahead - len(blanks & set(range(current_gw, current_gw + weeks_ahead)))
                    total += cbit_bonus * active_weeks
            
            return total
        
        # Store horizon EP separately for planning features
        df['expected_points_horizon'] = df.apply(_calc_horizon_ep, axis=1).astype(float)
        
        # ══════════════════════════════════════════════════════════════
        # ── Relative Rank Impact (RRI) Differential Model ──
        # Score = xP × (1 − EO_top10k)   ["Effective Net Points"]
        # Tells you how many points a player gains you vs. the field.
        # ══════════════════════════════════════════════════════════════

        # General ownership (0-100)
        eo_general = pd.to_numeric(
            df.get('selected_by_percent', 50), errors='coerce'
        ).fillna(50).clip(lower=0.1)

        # ── Estimate top-10k EO from general ownership ──
        # Top-10k managers cluster heavily on template players.
        # Empirical model: EO_10k ≈ clip(a·EO^1.4 + b, 1, 99)
        #   - 70% general → ~97% top-10k (template)
        #   - 10% general → ~20% top-10k
        #   -  2% general → ~3%  top-10k
        eo_frac = eo_general / 100.0
        eo_10k = (1.5 * np.power(eo_frac, 1.4) + 0.01).clip(0.01, 0.99)
        df['eo_top10k'] = (eo_10k * 100).round(1)

        # ── Core RRI: Differential Gain ──
        # xP × (1 − EO) = points you gain vs the field per GW
        xp = df['expected_points']
        df['differential_gain'] = (xp * (1 - eo_10k)).round(2)

        # ── Differential ROI per million ──
        cost = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5).clip(lower=4.0)
        df['diff_roi'] = (df['differential_gain'] / cost).round(3)

        # ── Poisson variance (σ) for boom/bust classification ──
        # σ of Poisson EP ≈ sqrt(λ_attack × goal_pts² + λ_creative × asst_pts²)
        goal_pts_map = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
        asst_pts = 3
        gp = df['position'].map(goal_pts_map).fillna(4)
        lam_a = pd.to_numeric(df.get('poisson_lambda_attack', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        lam_c = pd.to_numeric(df.get('poisson_lambda_creative', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        df['poisson_sigma'] = np.sqrt(lam_a * gp**2 + lam_c * asst_pts**2).round(2)

        # ── Boom/Bust label ──
        # High σ → high-ceiling winger (chase rank), Low σ → floor defender (protect lead)
        sigma_median = df['poisson_sigma'].median()
        sigma_median = sigma_median if sigma_median > 0 else 1.0
        df['sigma_ratio'] = df['poisson_sigma'] / sigma_median
        df['diff_profile'] = np.select(
            [df['sigma_ratio'] > 1.3, df['sigma_ratio'] < 0.7],
            ['Boom/Bust', 'Floor'],
            default='Balanced'
        )

        # ── Ownership-adjusted variance bonus ──
        # Underperforming xG → "due" for regression upward → higher diff value
        if 'us_goal_overperf' in df.columns:
            regression_signal = (
                -df['us_goal_overperf'].fillna(0) * 0.5
                + -df.get('us_assist_overperf', pd.Series(0, index=df.index)).fillna(0) * 0.3
            ).clip(-0.3, 0.5)
        else:
            regression_signal = pd.Series(0.0, index=df.index)
        df['variance_adj_diff'] = (df['differential_gain'] * (1 + regression_signal)).round(2)

        # ── Legacy compatibility aliases ──
        df['ep_net'] = df['differential_gain']

        # ── Primary differential score (composite) ──
        # Weighted blend of RRI, ROI, variance bonus, and fixtures
        df['differential_score'] = (
            df['differential_gain'] * 0.50          # Core RRI
            + df['diff_roi'] * 100 * 0.25           # Value-per-million
            + df['variance_adj_diff'] * 0.15        # Regression bonus
            + df['fixture_ease'] * 0.10             # Fixture influence
        ).round(2)

        # ── Diff verdict labels (for UI) ──
        df['diff_verdict'] = np.select(
            [
                (eo_10k > 0.60) & (xp >= 5),
                (eo_10k < 0.15) & (xp >= 4),
                (eo_10k < 0.10) & (xp < 2.5),
            ],
            ['Template', 'Elite Diff', 'Trap'],
            default=''
        )
        
        # Ensure now_cost is numeric
        df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5.0)
        
        # Price sensitivity (EP per cost)
        df['xg_per_pound'] = df['expected_points'] / df['now_cost'].clip(lower=4.0)
        
        # ── EPPM: Effective Points Per Million (next N gameweeks) ──
        df['eppm'] = df['expected_points'] / df['now_cost'].clip(lower=4.0)
        
        # ── Threat Momentum: Weighted rolling xG/xA (recent form emphasized) ──
        # Uses Understat per-90 stats with form weighting
        if 'us_xG_per90' in df.columns:
            # Form acts as recency weight (higher form = more recent performance)
            form = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0).clip(0, 10)
            form_weight = 0.5 + (form / 20)  # 0.5 to 1.0 range
            
            df['threat_momentum'] = (
                df['us_xG_per90'].fillna(0) * form_weight +
                df['us_xA_per90'].fillna(0) * form_weight * 0.7
            )
            
            # Momentum direction: form_weight centered on 0.75 (avg form=5)
            # Above 0.75 = heating up, below = cooling off
            df['threat_direction'] = ((form_weight - 0.75) / 0.25).clip(-1, 1)
        else:
            # Fallback using FPL expected stats
            form = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0).clip(0, 10)
            xg = pd.to_numeric(df.get('expected_goals', 0), errors='coerce').fillna(0)
            xa = pd.to_numeric(df.get('expected_assists', 0), errors='coerce').fillna(0)
            minutes = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
            games = (minutes / 90).clip(lower=1)
            
            df['threat_momentum'] = ((xg + xa * 0.7) / games) * (0.5 + form / 20)
            df['threat_direction'] = 0.0
        
        # ── Matchup Quality: Player threat × Opponent defensive weakness ──
        # matchup_quality = (Player_npxG_per_90) × (Opponent_xGA / League_Avg_xGA)
        # Uses REAL Understat xGA from Poisson pipeline when available
        if 'us_npxG_per90' in df.columns:
            player_threat = df['us_npxG_per90'].fillna(0) + df['us_xA_per90'].fillna(0) * 0.5
        else:
            xg = pd.to_numeric(df.get('expected_goals', 0), errors='coerce').fillna(0)
            xa = pd.to_numeric(df.get('expected_assists', 0), errors='coerce').fillna(0)
            minutes = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
            games = (minutes / 90).clip(lower=1)
            player_threat = (xg + xa * 0.5) / games
        
        # Use real opponent xGA from Poisson pipeline if available
        if 'poisson_opp_xGA_per90' in df.columns:
            poisson_opp = pd.to_numeric(df['poisson_opp_xGA_per90'], errors='coerce').fillna(0)
            league_avg_xga = poisson_opp[poisson_opp > 0].mean()
            if league_avg_xga > 0:
                opp_weakness = poisson_opp / league_avg_xga
                # Clip to reasonable range (0.5x to 2.0x of league average)
                opp_weakness = opp_weakness.clip(0.5, 2.0)
            else:
                opp_weakness = 0.5 + df['fixture_ease'].clip(0, 1)
        else:
            # Fallback: fixture_ease as proxy (0–1 scale → 0.5–1.5 multiplier)
            opp_weakness = 0.5 + df['fixture_ease'].clip(0, 1)
        
        df['matchup_quality'] = player_threat * opp_weakness
        
        # ── Engineered Differential: RRI-based with momentum + matchup ──
        low_own_bonus = np.where(eo_general < 10, (10 - eo_general) / 10, 0)
        df['engineered_diff'] = (
            df['differential_gain'] * 0.35 +
            df['eppm'] * 0.25 +
            df['threat_momentum'] * 0.20 +
            df['matchup_quality'] * 0.10 +
            low_own_bonus * 0.10
        ).round(2)
        
        # Value score: combines EP, RRI, regression, fixture coverage
        df['value_score'] = (
            df['expected_points'] * 0.35 +
            df['differential_gain'] * 0.25 +
            df['variance_adj_diff'] * 0.10 +
            df['diff_roi'] * 10 * 0.15 +
            (5.0 - df['blank_count'].astype(float)) * 0.15
        ).round(2)
        
        return df


class TransactionLedger:
    """
    SQL-style transaction log for tracking transfers and comparing teams.
    """
    
    def __init__(self):
        self.transactions = pd.DataFrame(columns=[
            'team_id', 'gameweek', 'player_in', 'player_out',
            'transfer_cost', 'timestamp'
        ])
        self.weekly_scores = pd.DataFrame(columns=[
            'team_id', 'gameweek', 'points', 'bench_points',
            'transfer_hits', 'captain_points'
        ])
    
    def record_transfer(self, team_id: int, gameweek: int, 
                       player_in: int, player_out: int, cost: int = 0):
        """Record a transfer in the ledger."""
        new_record = pd.DataFrame([{
            'team_id': team_id,
            'gameweek': gameweek,
            'player_in': player_in,
            'player_out': player_out,
            'transfer_cost': cost,
            'timestamp': datetime.now()
        }])
        self.transactions = pd.concat([self.transactions, new_record], ignore_index=True)
    
    def record_weekly_score(self, team_id: int, gameweek: int, points: int,
                           bench_points: int = 0, transfer_hits: int = 0, 
                           captain_points: int = 0):
        """Record weekly performance."""
        new_record = pd.DataFrame([{
            'team_id': team_id,
            'gameweek': gameweek,
            'points': points,
            'bench_points': bench_points,
            'transfer_hits': transfer_hits,
            'captain_points': captain_points
        }])
        self.weekly_scores = pd.concat([self.weekly_scores, new_record], ignore_index=True)
    
    def calculate_cumulative_alpha(self, team_id_1: int, team_id_2: int) -> pd.DataFrame:
        """
        Calculate cumulative alpha (performance difference) between two teams.
        """
        team1_scores = self.weekly_scores[self.weekly_scores['team_id'] == team_id_1].copy()
        team2_scores = self.weekly_scores[self.weekly_scores['team_id'] == team_id_2].copy()
        
        if team1_scores.empty or team2_scores.empty:
            return pd.DataFrame()
        
        merged = team1_scores.merge(
            team2_scores, on='gameweek', suffixes=('_team1', '_team2')
        )
        
        merged['weekly_alpha'] = merged['points_team1'] - merged['points_team2']
        merged['cumulative_alpha'] = merged['weekly_alpha'].cumsum()
        
        return merged[['gameweek', 'points_team1', 'points_team2', 
                       'weekly_alpha', 'cumulative_alpha']]
    
    def calculate_jaccard_similarity(self, picks1: List[int], picks2: List[int]) -> float:
        """
        Calculate Jaccard similarity (overlap) between two squads.
        """
        set1 = set(picks1)
        set2 = set(picks2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class RivalScout:
    """
    Analysis tools for comparing teams and finding tactical edges.
    """
    
    def __init__(self, fetcher: FPLDataFetcher, processor: FPLDataProcessor):
        self.fetcher = fetcher
        self.processor = processor
        self.ledger = TransactionLedger()
    
    def load_team_data(self, team_id: int) -> Dict:
        """Load and validate team data."""
        try:
            history = self.fetcher.get_team_history(team_id)
            current_gw = self.fetcher.get_current_gameweek()
            
            # Get current picks
            try:
                picks = self.fetcher.get_team_picks(team_id, current_gw)
            except FPLAPIError:
                # Try previous gameweek if current not available
                picks = self.fetcher.get_team_picks(team_id, max(1, current_gw - 1))
            
            return {
                'history': history,
                'picks': picks,
                'team_id': team_id,
                'valid': True
            }
        except FPLAPIError as e:
            return {
                'team_id': team_id,
                'valid': False,
                'error': str(e)
            }
    
    def compare_teams(self, team_id_1: int, team_id_2: int) -> Dict:
        """
        Comprehensive comparison between two teams.
        """
        team1_data = self.load_team_data(team_id_1)
        team2_data = self.load_team_data(team_id_2)
        
        if not team1_data['valid'] or not team2_data['valid']:
            return {
                'valid': False,
                'error': 'One or both team IDs are invalid'
            }
        
        # Get player IDs from each team
        picks1 = [p['element'] for p in team1_data['picks']['picks']]
        picks2 = [p['element'] for p in team2_data['picks']['picks']]
        
        # Calculate Jaccard similarity
        jaccard = self.ledger.calculate_jaccard_similarity(picks1, picks2)
        
        # Find unique players
        unique_to_team1 = set(picks1) - set(picks2)
        unique_to_team2 = set(picks2) - set(picks1)
        common_players = set(picks1) & set(picks2)
        
        # Find biggest threat (rival's unique player with highest EP)
        biggest_threat = None
        max_ep = 0
        for player_id in unique_to_team2:
            ep, _ = self.processor.calculate_expected_points(player_id)
            if ep > max_ep:
                max_ep = ep
                player_info = self.processor.players_df[
                    self.processor.players_df['id'] == player_id
                ]
                if not player_info.empty:
                    biggest_threat = {
                        'id': player_id,
                        'name': player_info.iloc[0]['web_name'],
                        'expected_points': ep
                    }
        
        return {
            'valid': True,
            'jaccard_similarity': jaccard,
            'overlap_count': len(common_players),
            'unique_to_you': list(unique_to_team1),
            'unique_to_rival': list(unique_to_team2),
            'common_players': list(common_players),
            'biggest_threat': biggest_threat,
            'your_squad': picks1,
            'rival_squad': picks2,
            'team1_data': team1_data,
            'team2_data': team2_data
        }


class FPLAPIError(Exception):
    """Custom exception for FPL API errors."""
    pass


# Utility functions for external use
def create_data_pipeline(odds_api_key: Optional[str] = None) -> Tuple[FPLDataFetcher, FPLDataProcessor]:
    """Create and return the full data pipeline."""
    fetcher = FPLDataFetcher(odds_api_key)
    odds_fetcher = OddsDataFetcher(odds_api_key)
    processor = FPLDataProcessor(fetcher, odds_fetcher)
    return fetcher, processor


def get_differential_picks(processor: FPLDataProcessor, 
                          min_ep: float = 3.0, 
                          max_ownership: float = 5.0,
                          weeks_ahead: int = 5) -> pd.DataFrame:
    """
    Find differential players with high EP but low ownership.
    """
    df = processor.get_engineered_features_df(weeks_ahead)
    
    differentials = df[
        (df['expected_points'] >= min_ep) &
        (df['selected_by_percent'].astype(float) <= max_ownership)
    ].sort_values('differential_score', ascending=False)
    
    return differentials[[
        'web_name', 'team_name', 'position', 'now_cost',
        'expected_points', 'selected_by_percent', 'differential_score',
        'fixture_ease', 'blank_count'
    ]]
