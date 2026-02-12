"""Tab renderers for FPL Strategy Engine."""

from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.planning import render_planning_tab

__all__ = [
    'render_strategy_tab',
    'render_optimization_tab',
    'render_rival_tab',
    'render_analytics_tab',
    'render_planning_tab',
]
