"""Tab renderers for FPL Strategy Engine."""

from tabs.dashboard import render_dashboard_tab
from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.ml_tab import render_ml_tab
from tabs.montecarlo_tab import render_monte_carlo_tab
from tabs.genetic_tab import render_genetic_tab

__all__ = [
    'render_dashboard_tab',
    'render_strategy_tab',
    'render_optimization_tab',
    'render_rival_tab',
    'render_analytics_tab',
    'render_ml_tab',
    'render_monte_carlo_tab',
    'render_genetic_tab',
]
