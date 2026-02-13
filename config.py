"""
Centralized configuration for FPL Strategy Engine.
All magic numbers, thresholds, and season constants live here.
"""

# ── 2025/26 Season Rules ──
SEASON = "2025/26"
MAX_FREE_TRANSFERS = 5
CAPTAIN_MULTIPLIER = 1.25
CBIT_BONUS_THRESHOLD = 10
CBIT_BONUS_POINTS = 2

# ── Squad Constraints ──
MAX_BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3
SQUAD_SIZE = 15
STARTING_XI = 11
POSITION_CONSTRAINTS = {
    'GKP': {'min': 2, 'max': 2, 'min_start': 1, 'max_start': 1},
    'DEF': {'min': 5, 'max': 5, 'min_start': 3, 'max_start': 5},
    'MID': {'min': 5, 'max': 5, 'min_start': 2, 'max_start': 5},
    'FWD': {'min': 3, 'max': 3, 'min_start': 1, 'max_start': 3},
}

# ── Optimization Defaults ──
DEFAULT_WEEKS_AHEAD = 5
DECAY_FACTOR = 0.95
TRANSFER_HIT_COST = 4

# ── FDR Thresholds ──
FDR_EASY = 2       # <= 2 is easy
FDR_MEDIUM = 3     # 3 is medium
FDR_HARD = 4       # >= 4 is tough
FDR_COLOR_MAP = {
    1: '#22c55e',
    2: '#77c45e',
    3: '#f59e0b',
    4: '#ef6b4e',
    5: '#ef4444',
}

# ── Ownership Tiers ──
OWNERSHIP_TEMPLATE_THRESHOLD = 25.0   # >= 25% = template
OWNERSHIP_POPULAR_THRESHOLD = 10.0    # 10-25% = popular
OWNERSHIP_DIFFERENTIAL_THRESHOLD = 5.0  # < 5% = differential
# 5-10% = enabler

# ── Rotation Risk Thresholds ──
ROTATION_SAFE_MINUTES_PCT = 0.80      # >= 80% available mins = safe
ROTATION_MODERATE_PCT = 0.60          # 60-80% = moderate risk
# < 60% = high risk

# ── Injury/Availability Status Codes ──
STATUS_AVAILABLE = 'a'
STATUS_INJURED = 'i'
STATUS_DOUBTFUL = 'd'
STATUS_SUSPENDED = 's'
STATUS_UNAVAILABLE = 'u'
UNAVAILABLE_STATUSES = {'i', 's', 'u'}

# ── Price Thresholds ──
PRICE_RISE_NET_TRANSFERS = 50_000
PRICE_FALL_NET_TRANSFERS = -50_000

# ── Dashboard Quick Pick Limits ──
DASHBOARD_TOP_PICKS = 5
DASHBOARD_INJURY_ALERTS = 8
DASHBOARD_FIXTURE_SWINGS = 6

# ── Chart Theme ──
CHART_BG = '#0a0a0b'
CHART_PLOT_BG = '#111113'
CHART_GRID = '#1e1e21'
CHART_FONT_FAMILY = 'Inter, sans-serif'
CHART_FONT_COLOR = '#6b6b6b'
CHART_FONT_SIZE = 11

POSITION_COLORS = {
    'GKP': '#3b82f6',
    'DEF': '#22c55e',
    'MID': '#f59e0b',
    'FWD': '#ef4444',
}

# ── API Config ──
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_DURATION = 300  # 5 minutes


def get_chart_layout(height=400, **overrides):
    """Return a standard Plotly layout dict for consistent styling."""
    layout = dict(
        height=height,
        template='plotly_dark',
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_PLOT_BG,
        font=dict(family=CHART_FONT_FAMILY, color=CHART_FONT_COLOR, size=CHART_FONT_SIZE),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
    )
    layout.update(overrides)
    return layout
