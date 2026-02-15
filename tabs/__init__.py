"""Tab renderers for FPL Strategy Engine."""

from tabs.dashboard import render_dashboard_tab
from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.ml_tab import render_ml_tab
from tabs.montecarlo_tab import render_monte_carlo_tab
from tabs.genetic_tab import render_genetic_tab
from tabs.captain_tab import render_captain_tab
from tabs.team_analysis_tab import render_team_analysis_tab
from tabs.price_predictor_tab import render_price_predictor_tab
from tabs.history_tab import render_history_tab
from tabs.wildcard_tab import render_wildcard_tab

__all__ = [
    'render_dashboard_tab',
    'render_strategy_tab',
    'render_optimization_tab',
    'render_rival_tab',
    'render_analytics_tab',
    'render_ml_tab',
    'render_monte_carlo_tab',
    'render_genetic_tab',
    'render_captain_tab',
    'render_team_analysis_tab',
    'render_price_predictor_tab',
    'render_history_tab',
    'render_wildcard_tab',
]
