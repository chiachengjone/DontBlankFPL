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
            'User-Agent': 'FPL-Strategy-Engine/2.0'
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
        
        # Convert price to actual value
        df['now_cost'] = df['now_cost'] / 10.0
        
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
        
        return df
    
    def calculate_cbit_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CBIT (Clearances, Blocks, Interceptions, Tackles) metrics.
        This is the new 2025/26 "DefCon" scoring system.
        """
        df = df.copy()
        
        # Estimate CBIT from available stats (using bonus as proxy for defensive actions)
        # In real implementation, this would use Opta data
        df['estimated_cbit'] = (
            df['bonus'].fillna(0) * 0.5 +  # Proxy
            df['clean_sheets'].fillna(0) * 2 +
            df['goals_conceded'].fillna(0) * -0.5
        )
        
        # CBIT per 90 minutes
        df['cbit_per_90'] = np.where(
            df['minutes'] > 0,
            (df['estimated_cbit'] / df['minutes']) * 90,
            0
        )
        
        # CBIT Propensity Score (probability of hitting 10 CBIT threshold)
        # Normalized to 0-1 scale
        max_cbit = df['cbit_per_90'].max() if df['cbit_per_90'].max() > 0 else 1
        df['cbit_propensity'] = df['cbit_per_90'] / max_cbit
        
        # Bonus points expectation from CBIT
        df['cbit_bonus_expected'] = df['cbit_propensity'] * CBIT_BONUS_POINTS
        
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
        
        # Pad with default if fewer fixtures
        while len(fdrs) < weeks_ahead:
            fdrs.append(3.0)
        
        return fdrs[:weeks_ahead]
    
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
        Calculate Differential Finder Index.
        High EP, low ownership, easy fixtures = high differential score.
        """
        player_row = self.players_df[self.players_df['id'] == player_id]
        if player_row.empty:
            return 0.0
        
        player = player_row.iloc[0]
        
        # Get EP
        total_ep, _ = self.calculate_expected_points(player_id, weeks_ahead)
        
        # Get effective ownership
        eo = float(player.get('selected_by_percent', 50) or 50)
        eo = max(eo, 0.1)  # Avoid division by zero
        
        # Get fixture ease
        fixture_ease = self.calculate_fixture_ease(player_id, weeks_ahead)
        
        # Differential Score = (EP / EO%) × Fixture Ease
        differential = (total_ep / eo) * fixture_ease * 100
        
        return differential
    
    def get_engineered_features_df(self, weeks_ahead: int = 5) -> pd.DataFrame:
        """
        Build comprehensive DataFrame with all engineered features.
        """
        df = self.players_df.copy()
        
        # Calculate CBIT metrics
        df = self.calculate_cbit_metrics(df)
        
        # Calculate per-player metrics
        eps_total = []
        eps_weekly = []
        fixture_ease_scores = []
        differential_scores = []
        blank_counts = []
        
        for player_id in df['id']:
            total_ep, weekly_ep = self.calculate_expected_points(player_id, weeks_ahead)
            eps_total.append(total_ep)
            eps_weekly.append(weekly_ep)
            fixture_ease_scores.append(self.calculate_fixture_ease(player_id, weeks_ahead))
            differential_scores.append(self.calculate_differential_score(player_id, weeks_ahead))
            blank_counts.append(len(self.detect_blanks(player_id, weeks_ahead)))
        
        df['expected_points'] = pd.to_numeric(pd.Series(eps_total), errors='coerce').fillna(0.0).values
        df['weekly_ep_breakdown'] = eps_weekly
        df['fixture_ease'] = pd.to_numeric(pd.Series(fixture_ease_scores), errors='coerce').fillna(0.5).values
        df['differential_score'] = pd.to_numeric(pd.Series(differential_scores), errors='coerce').fillna(0.0).values
        df['blank_count'] = pd.to_numeric(pd.Series(blank_counts), errors='coerce').fillna(0).astype(int).values
        
        # Ensure now_cost is numeric
        df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(5.0)
        
        # Price sensitivity (EP per cost)
        df['xg_per_pound'] = df['expected_points'].astype(float) / df['now_cost'].clip(lower=4.0).astype(float)
        
        # Value score - ensure all values are numeric floats
        ep_vals = df['expected_points'].astype(float)
        diff_vals = df['differential_score'].astype(float)
        blank_vals = df['blank_count'].astype(float)
        
        df['value_score'] = (
            ep_vals * 0.4 +
            diff_vals * 0.3 +
            (5.0 - blank_vals) * 0.3
        )
        
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
