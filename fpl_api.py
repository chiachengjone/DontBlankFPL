"""
FPL API Module - Data fetching and processing for the 2025/26 FPL Strategy Engine.
Handles all interactions with the official FPL API and The Odds API.
"""

import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass, field

from config import (
    FPL_BASE_URL, ODDS_API_BASE_URL, CBIT_BONUS_THRESHOLD,
    CBIT_BONUS_POINTS, MAX_FREE_TRANSFERS, CAPTAIN_MULTIPLIER,
    CACHE_DURATION,
)

logger = logging.getLogger(__name__)


def safe_numeric(series, default=0):
    """Safely convert series or scalar to numeric, replacing NaN with default."""
    try:
        if isinstance(series, (pd.Series, pd.Index, np.ndarray)):
            return pd.to_numeric(series, errors='coerce').fillna(default)
        val = pd.to_numeric(series, errors='coerce')
        return val if pd.notnull(val) else default
    except Exception:
        return default

# Maximum number of entries kept in the in-memory cache.
# Each entry is keyed by endpoint/param combo; this prevents unbounded growth
# when many unique player/team lookups are performed.
_MAX_CACHE_ENTRIES: int = 500


@dataclass
class EngineStatus:
    """Tracks the status and errors of various enrichment modules."""
    understat_active: bool = False
    understat_status: Dict = field(default_factory=dict)
    poisson_active: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class EngineeredFeaturesResult:
    """Return object for the engineered features pipeline."""
    df: pd.DataFrame
    status: EngineStatus


class FPLDataFetcher:
    """
    Handles all communication with the FPL API.
    
    This class is designed to be robust:
    1.  **Rate Limiting**: Ensures we don't send too many requests too fast.
    2.  **Retries**: Automatically tries again if the internet blips or FPL is busy.
    3.  **Caching**: Remembers answers for a while to save time.
    """

    # ── Configuration ──
    # Minimum time (seconds) to wait between requests to be polite to the server
    MIN_REQUEST_INTERVAL: float = 1.0
    
    # How many times to try again if a request fails
    MAX_RETRIES: int = 3
    
    # "Backoff" means we wait longer after each failure (2s, 4s, 8s...)
    RETRY_BACKOFF_BASE: float = 2.0
    
    # Give up if the server doesn't respond in this many seconds
    REQUEST_TIMEOUT: int = 15
    
    def __init__(self, odds_api_key: Optional[str] = None):
        self.session = requests.Session()
        # Masquerade as a browser/app to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'FPL-Strategy-Engine/2.0',
            'Accept-Charset': 'utf-8'
        })
        self.odds_api_key = odds_api_key
        self._cache: Dict[str, dict] = {}
        self._cache_expiry: Dict[str, float] = {}
        self.cache_duration = CACHE_DURATION
        self._last_request_time: float = 0.0

    # ── Rate Limiting & Retries ──

    def _rate_limit(self):
        """
        Calculates how much time has passed since the last request and sleeps
        if necessary to ensure we respect the MIN_REQUEST_INTERVAL.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(self, url: str, context: str = "") -> requests.Response:
        """
        Sends an HTTP GET request with built-in safety nets.
        
        If the server says "Too Many Requests" (429) or fails unexpectedly,
        this will wait a bit and try again up to MAX_RETRIES times.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            self._rate_limit() # Always pause before sending
            try:
                response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
                
                # Special handling for FPL's rate limit response
                if response.status_code == 429:  # Too Many Requests
                    wait = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning("FPL says to slow down! Waiting %.1fs (Context: %s)", wait, context)
                    time.sleep(wait)
                    continue
                
                response.raise_for_status() # Raise error for 404/500 codes
                response.encoding = 'utf-8'
                return response
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning("Network blip (%s). Retrying in %.1fs... Error: %s", context, wait, exc)
                    time.sleep(wait)
        
        # If we get here, we've exhausted all retries
        raise FPLAPIError(f"Gave up after {self.MAX_RETRIES} tries ({context}). Last error: {last_exc}")

    # ── Smart Caching ──

    def _get_cached(self, key: str) -> Optional[dict]:
        """Check if we have a valid (non-expired) copy of this data in memory."""
        if key in self._cache:
            if time.time() < self._cache_expiry.get(key, 0):
                return self._cache[key]
            # It's expired, so clear it out to make room for fresh data
            self._cache.pop(key, None)
            self._cache_expiry.pop(key, None)
        return None
    
    def _set_cache(self, key: str, data: dict):
        """Store data in memory. If full, throw out the oldest item."""
        # Evict oldest entries if at capacity
        if len(self._cache) >= _MAX_CACHE_ENTRIES:
            oldest_key = min(self._cache_expiry, key=self._cache_expiry.get)
            self._cache.pop(oldest_key, None)
            self._cache_expiry.pop(oldest_key, None)
        self._cache[key] = data
        self._cache_expiry[key] = time.time() + self.cache_duration
    
    def get_bootstrap_static(self) -> Dict:
        """
        Fetch the 'Big Data' from FPL.
        Contains all players, teams, gameweek schedules, and rules.
        This is the foundation for everything else.
        """
        cached = self._get_cached('bootstrap')
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/bootstrap-static/", context="bootstrap-static"
        )
        data = response.json()
        self._set_cache('bootstrap', data)
        return data
    
    def get_player_summary(self, player_id: int) -> Dict:
        """
        Fetch detailed player history and fixtures.
        Returns past season data, current season history, and upcoming fixtures.
        """
        player_id = int(player_id)
        cache_key = f'element_{player_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/element-summary/{player_id}/",
            context=f"element-summary/{player_id}",
        )
        data = response.json()
        self._set_cache(cache_key, data)
        return data
    
    def get_team_picks(self, team_id: int, gameweek: int) -> Dict:
        """
        Fetch a user's squad for a specific gameweek.
        Returns picks, active chip, automatic substitutions, and entry history.
        """
        team_id, gameweek = int(team_id), int(gameweek)
        cache_key = f'picks_{team_id}_gw{gameweek}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/entry/{team_id}/event/{gameweek}/picks/",
            context=f"entry/{team_id}/event/{gameweek}/picks",
        )
        data = response.json()
        self._set_cache(cache_key, data)
        return data
    
    def get_team_history(self, team_id: int) -> Dict:
        """Fetch complete history for a team."""
        team_id = int(team_id)
        cache_key = f'history_{team_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/entry/{team_id}/history/",
            context=f"entry/{team_id}/history",
        )
        data = response.json()
        self._set_cache(cache_key, data)
        return data

    def get_transfers(self, team_id: int) -> List[Dict]:
        """Fetch transfer history for a team."""
        team_id = int(team_id)
        cache_key = f'transfers_{team_id}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/entry/{team_id}/transfers/",
            context=f"entry/{team_id}/transfers",
        )
        data = response.json()
        self._set_cache(cache_key, data)
        return data
    
    def get_fixtures(self) -> List[Dict]:
        """Fetch all fixtures for the season."""
        cached = self._get_cached('fixtures')
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/fixtures/", context="fixtures"
        )
        data = response.json()
        self._set_cache('fixtures', data)
        return data
            
    def get_event_live(self, gameweek: int) -> Dict:
        """Fetch all player points/stats for a specific gameweek."""
        gameweek = int(gameweek)
        cache_key = f'live_gw{gameweek}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        response = self._request_with_retry(
            f"{FPL_BASE_URL}/event/{gameweek}/live/",
            context=f"event/{gameweek}/live",
        )
        data = response.json()
        self._set_cache(cache_key, data)
        return data
    
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
        
        # Add full name for better search and display
        df['full_name'] = df['first_name'] + " " + df['second_name']
        
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
    
    def calculate_cbit_metrics(self, df: pd.DataFrame, team_fdrs: Optional[Dict] = None) -> pd.DataFrame:
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
        
        # Games played and reliability
        minutes = safe_numeric(df.get('minutes', 0))
        total_gws = max(df.get('starts', pd.Series([1]*len(df))).max() + df.get('subs_bench', pd.Series([0]*len(df))).max(), 1)
        
        # Starts % and Minutes per game are better proxies for "active reliability"
        starts_pct = safe_numeric(df.get('starts_pct', 0.8)).clip(0, 1)
        mins_per_game = safe_numeric(df.get('minutes_per_game', 70)).clip(20, 90)
        
        # Base defensive action rates by position
        base_aa90 = {
            'GKP': 4.5,   # Saves + basic clearances
            'DEF': 12.5,  # CBs ~13-14, WBs ~10-11
            'MID': 7.5,   # DMs ~10, AMs ~4
            'FWD': 3.2,   # Pressing and defensive set pieces
        }
        df['_base_aa90'] = df['position'].map(base_aa90).fillna(5.0)
        
        # Adjust by Clean Sheet propensity (defensive solidity proxy)
        # Using historical CS rate or Poisson P(CS)
        if 'poisson_p_cs' in df.columns:
            p_cs = safe_numeric(df['poisson_p_cs']).clip(0, 0.7)
        else:
            historical_cs = (safe_numeric(df.get('clean_sheets', 0)) / (minutes / 90).clip(lower=1)).clip(0, 1)
            p_cs = historical_cs.clip(0, 0.6)

        # Scaling: High P(CS) (+20% actions), Low P(CS) (-10% actions)
        cs_adjustment = 0.9 + (p_cs * 0.4)
        
        # Rotation Risk Penalty
        # If a player starts < 60% of games, their per-90 actions are less likely to hit the threshold in a full match
        rotation_penalty = np.where(starts_pct < 0.6, 0.85, 1.0)
        
        df['cbit_aa90'] = (df['_base_aa90'] * cs_adjustment * rotation_penalty).round(1)
        
        # ── Distance to Threshold (DTT) ──
        # Positive = exceeds threshold, Negative = below threshold
        # Players near 0 are "high variance" CBIT assets
        df['cbit_dtt'] = (df['cbit_aa90'] - df['cbit_threshold']).round(1)
        
        # ── CBIT Probability (Modified Poisson) ──
        # Real-world defensive actions are overdispersed, so raw Poisson overpredicts 
        # consistency. We apply an empirical 15% discount for FPL realism.
        def calc_cbit_prob(aa90, threshold):
            if aa90 <= 0:
                return 0.0
            raw_prob = 1 - poisson.cdf(threshold - 1, aa90)
            return raw_prob * 0.85 
        
        df['cbit_prob'] = df.apply(
            lambda r: calc_cbit_prob(r['cbit_aa90'], r['cbit_threshold']), axis=1
        ).clip(0, 1).round(3)
        
        # ── Hit Rate (Season %) ──
        # Combination of probability and minutes consistency
        # Minutes consistency = how reliably the player gets playing time
        mins_consistency = (mins_per_game / 90).clip(0, 1)
        df['cbit_hit_rate'] = (df['cbit_prob'] * mins_consistency).round(2)
        
        # ── Clean Sheet Probability (xCS) ──
        # Prefer Poisson p_cs from fixture-based calculation, fallback to historical CS rate
        if 'poisson_p_cs' in df.columns:
            df['_xcs'] = safe_numeric(df['poisson_p_cs']).clip(0, 0.7)  # Cap at 70%
        else:
            df['_xcs'] = p_cs.clip(0, 0.6)  # Cap at 60%
        
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
        elif team_fdrs is not None:
             # Use FDR as proxy: Higher FDR = Harder Opponent = More Defensive Actions
             # Map FDR 1-5 to Matchup 0.7-1.3
             def get_matchup_from_fdr(tid):
                 try:
                     # Ensure ID is int for lookup
                     tid = int(tid)
                 except (ValueError, TypeError):
                     return 1.0
                 
                 fdrs = team_fdrs.get(tid)
                 if not fdrs:
                     return 1.0 # Default if no fixtures found
                     
                 next_fdr = fdrs[0]
                 # FDR 5 (Hard) -> 1.3 (More defensive actions expected)
                 # FDR 1 (Easy) -> 0.7 (Fewer defensive actions expected)
                 # FDR 3 (Avg)  -> 1.0 (Neutral)
                 return 1.0 + (next_fdr - 3.0) * 0.15
                 
             df['cbit_matchup'] = df['team'].map(get_matchup_from_fdr).fillna(1.0).clip(0.7, 1.4).round(2)
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
            df['cbit_prob'] * 4.0 +
            df['cbit_floor'] * 0.3 +
            starts_pct * 1.5 +
            (df['cbit_matchup'] - 1.0).clip(-0.5, 0.5) * 2.0
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
        eo = float(player.get('selected_by_percent', 0) or 0)
        
        # EP_net: Points gained vs field
        ep_net = total_ep * (1 - eo/100)
        
        # Diff_ROI: Points per % ownership
        diff_roi = total_ep / (eo + 1)
        
        # Fixture ease
        ease = self.calculate_fixture_ease(player_id, weeks_ahead)
        
        # Composite score
        score = (ep_net * 0.5) + (diff_roi * 100 * 0.3) + (ease * 0.2)
        
        return round(score, 2)

    def _get_team_fixture_metrics(self, weeks_ahead: int) -> Tuple[Dict, Dict, Dict, List[float], float]:
        """Calculates team-level fixture metrics (FDR, Blanks, Ease)."""
        current_gw = self.fetcher.get_current_gameweek()
        fixtures = self.fixtures_df
        team_ids = self.teams_df['id'].unique()
        
        team_fdrs = {}
        team_blanks = {}
        team_ease = {}
        
        decay_weights = [0.95 ** i for i in range(weeks_ahead)]
        max_ease_denom = sum(5 * w for w in decay_weights)
        
        for tid in team_ids:
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
            
            blanks = set()
            for gw in range(current_gw, current_gw + weeks_ahead):
                if gw not in fixture_gws:
                    blanks.add(gw)
            team_blanks[tid] = blanks
            
            weighted_ease = sum((6 - fdr) * w for fdr, w in zip(fdrs, decay_weights))
            team_ease[tid] = weighted_ease / max_ease_denom if max_ease_denom > 0 else 0.5
            
        return team_fdrs, team_blanks, team_ease, decay_weights, max_ease_denom

    def _enrich_with_external_data(self, df: pd.DataFrame, status: EngineStatus) -> pd.DataFrame:
        """Enriches the DataFrame with Understat data without direct st.session_state access."""
        try:
            from understat_api import enrich_with_understat
            df = enrich_with_understat(df)
            status.understat_active = True
            # In a non-Streamlit environment, we can't easily populate st.session_state
            # but we can return the status in the EngineStatus object.
        except Exception as e:
            logger.error(f"Understat enrichment failed: {e}")
            status.understat_active = False
            status.errors.append(f"Understat: {str(e)}")
        return df

    def _calculate_poisson_metrics(self, df: pd.DataFrame, weeks_ahead: int, status: EngineStatus, 
                                  team_stats: Optional[Dict] = None, use_poisson_primary: bool = True) -> pd.DataFrame:
        """Calculates Poisson-based Expected Points."""
        try:
            from poisson_ep import calculate_poisson_ep_for_dataframe
            current_gw = self.fetcher.get_current_gameweek()
            fixtures = self.fixtures_df
            
            df = calculate_poisson_ep_for_dataframe(
                df, fixtures, current_gw, team_stats=team_stats, horizon=weeks_ahead
            )
            
            status.poisson_active = True
            
            if use_poisson_primary and 'expected_points_poisson' in df.columns:
                poisson_ep = pd.to_numeric(df['expected_points_poisson'], errors='coerce').fillna(0)
                base_ep = pd.to_numeric(df.get('ep_next_num', 0), errors='coerce').fillna(0)
                df['expected_points'] = (poisson_ep * 0.7 + base_ep * 0.3).clip(lower=0)
            
        except Exception as e:
            logger.error(f"Poisson EP calculation failed: {e}")
            status.poisson_active = False
            status.errors.append(f"Poisson: {str(e)}")
        return df

    def _apply_differential_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates RRI and other differential metrics."""
        # Use safe_numeric for Series conversion
        eo_general = safe_numeric(df.get('selected_by_percent', 50)).clip(lower=0.1)
        eo_frac = eo_general / 100.0
        eo_10k = (1.5 * np.power(eo_frac, 1.4) + 0.01).clip(0.01, 0.99)
        df['eo_top10k'] = (eo_10k * 100).round(1)
        
        xp = df['expected_points']
        df['differential_gain'] = (xp * (1 - eo_10k)).round(2)
        
        cost = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5).clip(lower=4.0)
        df['diff_roi'] = (df['differential_gain'] / cost).round(3)
        
        goal_pts_map = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
        asst_pts = 3
        gp = df['position'].map(goal_pts_map).fillna(4)
        lam_a = safe_numeric(df.get('poisson_lambda_attack', pd.Series(0, index=df.index)))
        lam_c = safe_numeric(df.get('poisson_lambda_creative', pd.Series(0, index=df.index)))
        df['poisson_sigma'] = np.sqrt(lam_a * gp**2 + lam_c * asst_pts**2).round(2)
        
        sigma_median = df['poisson_sigma'].median()
        sigma_median = sigma_median if sigma_median > 0 else 1.0
        df['sigma_ratio'] = df['poisson_sigma'] / sigma_median
        df['diff_profile'] = np.select(
            [df['sigma_ratio'] > 1.3, df['sigma_ratio'] < 0.7],
            ['Boom/Bust', 'Floor'],
            default='Balanced'
        )
        
        if 'us_goal_overperf' in df.columns:
            regression_signal = (
                -df['us_goal_overperf'].fillna(0) * 0.5
                + -df.get('us_assist_overperf', 0).fillna(0) * 0.3
            ).clip(-0.3, 0.5)
        else:
            regression_signal = 0.0
            
        df['variance_adj_diff'] = (df['differential_gain'] * (1 + regression_signal)).round(2)
        df['ep_net'] = df['differential_gain']
        
        df['differential_score'] = (
            df['differential_gain'] * 0.50 +
            df['diff_roi'] * 100 * 0.25 +
            df['variance_adj_diff'] * 0.15 +
            df['fixture_ease'] * 0.10
        ).round(2)
        
        df['diff_verdict'] = np.select(
            [
                (eo_10k > 0.60) & (xp >= 5),
                (eo_10k < 0.15) & (xp >= 4),
                (eo_10k < 0.10) & (xp < 2.5),
            ],
            ['Template', 'Elite Diff', 'Trap'],
            default=''
        )
        return df

    def _apply_momentum_and_matchup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates momentum and matchup quality."""
        if 'us_xG_per90' in df.columns:
            form = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0).clip(0, 10)
            form_weight = 0.5 + (form / 20)
            df['threat_momentum'] = (
                df['us_xG_per90'].fillna(0) * form_weight +
                df['us_xA_per90'].fillna(0) * form_weight * 0.7
            )
            df['threat_direction'] = ((form_weight - 0.75) / 0.25).clip(-1, 1)
        else:
            form = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0).clip(0, 10)
            xg = pd.to_numeric(df.get('expected_goals', 0), errors='coerce').fillna(0)
            xa = pd.to_numeric(df.get('expected_assists', 0), errors='coerce').fillna(0)
            minutes = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
            games = (minutes / 90).clip(lower=1)
            df['threat_momentum'] = ((xg + xa * 0.7) / games) * (0.5 + form / 20)
            df['threat_direction'] = 0.0

        if 'us_npxG_per90' in df.columns:
            player_threat = df['us_npxG_per90'].fillna(0) + df['us_xA_per90'].fillna(0) * 0.5
        else:
            xg = pd.to_numeric(df.get('expected_goals', 0), errors='coerce').fillna(0)
            xa = pd.to_numeric(df.get('expected_assists', 0), errors='coerce').fillna(0)
            minutes = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
            games = (minutes / 90).clip(lower=1)
            player_threat = (xg + xa * 0.5) / games
            
        if 'poisson_opp_xGA_per90' in df.columns:
            poisson_opp = pd.to_numeric(df['poisson_opp_xGA_per90'], errors='coerce').fillna(0)
            league_avg_xga = poisson_opp[poisson_opp > 0].mean()
            opp_weakness = poisson_opp / league_avg_xga if league_avg_xga > 0 else 0.5 + df['fixture_ease'].clip(0, 1)
            opp_weakness = opp_weakness.clip(0.5, 2.0)
        else:
            opp_weakness = 0.5 + df['fixture_ease'].clip(0, 1)
            
        df['matchup_quality'] = player_threat * opp_weakness
        return df

    def get_engineered_features_result(self, weeks_ahead: int = 5, 
                                      team_stats: Optional[Dict] = None, 
                                      use_poisson_primary: bool = True) -> EngineeredFeaturesResult:
        """
        New modular entry point for feature engineering.
        Returns an EngineeredFeaturesResult object.
        """
        status = EngineStatus()
        df = self.players_df.copy()
        
        # 1. Team Fixture Metrics (Calculated first so they can be used for Matchup logic)
        team_fdrs, team_blanks, team_ease, decay_weights, _ = self._get_team_fixture_metrics(weeks_ahead)
        df['fixture_ease'] = df['team'].map(team_ease).fillna(0.5).astype(float)
        df['blank_count'] = df['team'].map(lambda tid: len(team_blanks.get(tid, set()))).fillna(0).astype(int)
        
        # 2. CBIT Metrics (Now uses FDR for matchup)
        df = self.calculate_cbit_metrics(df, team_fdrs=team_fdrs)
        
        # 3. Base EP Prep
        df['ep_next_num'] = pd.to_numeric(df.get('ep_next', 0), errors='coerce').fillna(0.0)
        df['expected_points'] = df['ep_next_num'].copy()
        
        # 4. Understat Enrichment
        df = self._enrich_with_external_data(df, status)
        
        # 5. Poisson EP
        df = self._calculate_poisson_metrics(df, weeks_ahead, status, team_stats, use_poisson_primary)
        
        # 6. Horizon EP (Vectorized where possible)
        # Optimized horizon EP calculation
        current_gw = self.fetcher.get_current_gameweek()
        
        # Precompute per-team horizon EP boost
        # A true multiplier. 1 normal game = 1.0. Hard game < 1.0, Easy game > 1.0. Blank = 0.
        team_horizon_mult = {}
        for tid, fdrs in team_fdrs.items():
            blanks = team_blanks.get(tid, set())
            mult = 0.0
            for i in range(weeks_ahead):
                if (current_gw + i) not in blanks:
                    # e.g FDR 3 (Neutral) = 1.0 multiplier
                    # FDR 2 (Easy) = 1.1 multiplier
                    # FDR 4 (Hard) = 0.9 multiplier
                    # Apply a slight 5% decay per week into the future for uncertainty
                    decay = 0.95 ** i
                    fdr_val = fdrs[i] if i < len(fdrs) else 3.0
                    game_multiplier = 1.0 + (3.0 - fdr_val) * 0.1
                    mult += decay * game_multiplier
            team_horizon_mult[tid] = mult
            
        df['horizon_multiplier'] = df['team'].map(team_horizon_mult).fillna(float(weeks_ahead) * 0.9)
        df['expected_points_horizon'] = df['expected_points'] * df['horizon_multiplier']
        
        # Add CBIT bonus to horizon EP
        if 'cbit_bonus_expected' in df.columns:
            def_mask = df['position'] == 'DEF'
            active_weeks = weeks_ahead - df['blank_count']
            df.loc[def_mask, 'expected_points_horizon'] += df.loc[def_mask, 'cbit_bonus_expected'] * active_weeks
            
        # 7. Differential & Momentum
        df = self._apply_differential_model(df)
        df = self._apply_momentum_and_matchup(df)
        
        # 8. Composite scores
        eo_general = pd.to_numeric(df.get('selected_by_percent', 50), errors='coerce').fillna(50)
        low_own_bonus = np.where(eo_general < 10, (10 - eo_general) / 10, 0)
        
        df['engineered_diff'] = (
            df['differential_gain'] * 0.35 +
            (df['expected_points'] / df['now_cost'].clip(lower=4.0)) * 0.25 +
            df['threat_momentum'] * 0.20 +
            df['matchup_quality'] * 0.10 +
            low_own_bonus * 0.10
        ).round(2)
        
        df['value_score'] = (
            df['expected_points'] * 0.35 +
            df['differential_gain'] * 0.25 +
            df['variance_adj_diff'] * 0.10 +
            (df['differential_gain'] / df['now_cost'].clip(lower=4.0)) * 10 * 0.15 +
            (5.0 - df['blank_count'].astype(float)) * 0.15
        ).round(2)
        
        df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5.0)
        df['xg_per_pound'] = df['expected_points'] / df['now_cost'].clip(lower=4.0)
        df['eppm'] = df['xg_per_pound']
        
        return EngineeredFeaturesResult(df=df, status=status)

    def get_engineered_features_df(self, weeks_ahead: int = 5) -> pd.DataFrame:
        """
        Backward compatible entry point. 
        Tries to handle Streamlit session state if available but doesn't require it.
        """
        # Attempt to get Streamlit specific settings if running in Streamlit
        st = None
        try:
            import streamlit as st
            team_stats = st.session_state.get('_understat_team_stats', None)
            use_poisson = st.session_state.get('use_poisson_xp', True)
        except Exception:
            team_stats = None
            use_poisson = True
            st = None
            
        result = self.get_engineered_features_result(
            weeks_ahead=weeks_ahead, 
            team_stats=team_stats, 
            use_poisson_primary=use_poisson
        )
        
        # Update Streamlit session state if we are indeed in a Streamlit context
        if st is not None:
            try:
                st.session_state['_understat_active'] = result.status.understat_active
                # Some other legacy status might need to be set here if needed
            except Exception:
                pass
                
        return result.df


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
