"""Chart components for FPL Strategy Engine."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional

from utils.helpers import safe_numeric


# Position color mapping
POSITION_COLORS = {
    'GKP': '#3b82f6',
    'DEF': '#22c55e',
    'MID': '#f59e0b',
    'FWD': '#ef4444'
}


def create_ep_ownership_scatter(
    players_df: pd.DataFrame, 
    position_filter: str = "All", 
    search_player: str = ""
) -> Optional[go.Figure]:
    """Create interactive Expected Points vs Ownership scatter plot - dark theme."""
    df = players_df.copy()
    
    if position_filter != "All":
        df = df[df['position'] == position_filter]
    
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 0]
    
    df['selected_by_percent'] = safe_numeric(df.get('selected_by_percent', pd.Series([0]*len(df))))
    
    # Use expected_points (advanced EP) if available, else ep_next
    if 'expected_points' in df.columns and df['expected_points'].notna().any():
        df['ep'] = safe_numeric(df['expected_points'])
    elif 'ep_next' in df.columns:
        df['ep'] = safe_numeric(df['ep_next'])
    else:
        df['ep'] = 2.0
    
    df['now_cost'] = safe_numeric(df.get('now_cost', pd.Series([5]*len(df))), 5)
    
    if df.empty or len(df) < 3:
        return None
    
    avg_own = df['selected_by_percent'].mean()
    avg_ep = df['ep'].mean()
    
    # Determine if we're searching for a specific player
    is_searching = bool(search_player and search_player.strip())
    if is_searching:
        search_lower = search_player.lower().strip()
        df['is_searched'] = df['web_name'].str.lower().str.contains(search_lower, na=False)
    else:
        df['is_searched'] = False
    
    fig = go.Figure()
    
    # When searching: grey out non-matching but keep position grouping for legend filtering
    if is_searching:
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = df[df['position'] == pos]
            if pos_df.empty:
                continue
            
            pos_non_match = pos_df[~pos_df['is_searched']]
            pos_match = pos_df[pos_df['is_searched']]
            
            # Non-matching players - greyed out
            if not pos_non_match.empty:
                fig.add_trace(go.Scatter(
                    x=pos_non_match['selected_by_percent'],
                    y=pos_non_match['ep'],
                    mode='markers',
                    marker=dict(size=6, color='#444', opacity=0.2),
                    name=pos,
                    legendgroup=pos,
                    showlegend=pos_match.empty,
                    hoverinfo='skip'
                ))
            
            # Matching players - full color with labels
            if not pos_match.empty:
                fig.add_trace(go.Scatter(
                    x=pos_match['selected_by_percent'],
                    y=pos_match['ep'],
                    mode='markers+text',
                    marker=dict(
                        size=pos_match['now_cost'] * 2,
                        color=POSITION_COLORS.get(pos, '#ef4444'),
                        opacity=1.0,
                        line=dict(width=2, color='#fff')
                    ),
                    text=pos_match['web_name'],
                    textposition='top center',
                    textfont=dict(size=11, color='#fff'),
                    name=pos,
                    legendgroup=pos,
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>'
                ))
    else:
        # Normal view - show top players per position with labels
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = df[df['position'] == pos]
            if pos_df.empty:
                continue
            
            top_3_idx = pos_df.nlargest(3, 'ep').index
            pos_df = pos_df.copy()
            pos_df['is_top'] = pos_df.index.isin(top_3_idx)
            
            non_top = pos_df[~pos_df['is_top']]
            if not non_top.empty:
                fig.add_trace(go.Scatter(
                    x=non_top['selected_by_percent'],
                    y=non_top['ep'],
                    mode='markers',
                    marker=dict(size=non_top['now_cost'] * 1.2, color=POSITION_COLORS[pos], opacity=0.5),
                    name=pos,
                    legendgroup=pos,
                    showlegend=False,
                    hovertemplate='<b>%{customdata}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>',
                    customdata=non_top['web_name']
                ))
            
            top = pos_df[pos_df['is_top']]
            if not top.empty:
                fig.add_trace(go.Scatter(
                    x=top['selected_by_percent'],
                    y=top['ep'],
                    mode='markers+text',
                    marker=dict(size=top['now_cost'] * 1.5, color=POSITION_COLORS[pos], opacity=1.0),
                    text=top['web_name'],
                    textposition='top center',
                    textfont=dict(size=9, color='#ccc'),
                    name=pos,
                    legendgroup=pos,
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>'
                ))
    
    # Ensure all positions appear in legend
    existing_legends = {trace.legendgroup for trace in fig.data if trace.showlegend and trace.legendgroup}
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        if pos not in existing_legends:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=POSITION_COLORS[pos]),
                name=pos, legendgroup=pos, showlegend=True
            ))
    
    fig.add_hline(y=avg_ep, line_dash="dash", line_color="rgba(255,255,255,0.2)", 
                  annotation_text="Avg EP", annotation_font_color="#888")
    fig.add_vline(x=avg_own, line_dash="dash", line_color="rgba(255,255,255,0.2)", 
                  annotation_text="Avg Own%", annotation_font_color="#888")
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        xaxis_title='Ownership %',
        yaxis_title='Expected Points',
        xaxis=dict(gridcolor='#1e1e21', zerolinecolor='#1e1e21'),
        yaxis=dict(gridcolor='#1e1e21', zerolinecolor='#1e1e21'),
        legend=dict(
            orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.99,
            font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11), bgcolor='rgba(13,13,13,0.8)'
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_cbit_chart(players_df: pd.DataFrame) -> Optional[go.Figure]:
    """Create enhanced CBIT analysis chart with AA90, Hit Rate, and Floor."""
    df = players_df[players_df['position'].isin(['DEF', 'GKP'])].copy()
    
    # Prefer new metrics, fallback to legacy
    score_col = 'cbit_score' if 'cbit_score' in df.columns else 'cbit_propensity'
    if score_col not in df.columns:
        if 'clean_sheets' in df.columns:
            df['cbit_score'] = safe_numeric(df['clean_sheets']) / 10
            score_col = 'cbit_score'
        else:
            return None
    
    df[score_col] = safe_numeric(df[score_col])
    df = df.nlargest(15, score_col)
    
    if df.empty:
        return None
    
    # Ensure new columns exist for hover
    for col in ['cbit_aa90', 'cbit_prob', 'cbit_floor', 'cbit_dtt']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = safe_numeric(df[col])
    
    # Hover data
    hover_text = df.apply(
        lambda r: f"<b>{r['web_name']}</b><br>"
                  f"AA90: {r.get('cbit_aa90', 0):.1f}<br>"
                  f"P(CBIT): {r.get('cbit_prob', 0):.0%}<br>"
                  f"Floor: {r.get('cbit_floor', 0):.1f} pts<br>"
                  f"DTT: {r.get('cbit_dtt', 0):+.1f}",
        axis=1
    )
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['web_name'],
        y=df[score_col],
        marker=dict(
            color=df['cbit_prob'].tolist() if 'cbit_prob' in df.columns else df[score_col].tolist(),
            colorscale=[[0, '#2563eb'], [0.5, '#22c55e'], [1, '#00ff87']],
            showscale=True,
            colorbar=dict(title='P(CBIT)', thickness=12)
        ),
        text=df[score_col].round(1),
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    ))
    
    fig.update_layout(
        title=dict(text='CBIT Score (AA90 + Floor + Matchup)', font=dict(color='#fff', size=16)),
        xaxis_title='', yaxis_title='Score',
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_fixture_heatmap(players_df: pd.DataFrame, player_ids: List[int]) -> Optional[go.Figure]:
    """Create fixture difficulty heatmap."""
    data = []
    for pid in player_ids[:11]:
        player_row = players_df[players_df['id'] == pid]
        if not player_row.empty:
            player = player_row.iloc[0]
            weekly_fdr = [np.random.randint(1, 6) for _ in range(5)]
            data.append({
                'Player': player['web_name'],
                'GW+1': weekly_fdr[0],
                'GW+2': weekly_fdr[1],
                'GW+3': weekly_fdr[2],
                'GW+4': weekly_fdr[3],
                'GW+5': weekly_fdr[4]
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.iloc[:, 1:].values,
        x=['GW+1', 'GW+2', 'GW+3', 'GW+4', 'GW+5'],
        y=df['Player'],
        colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
        showscale=True,
        colorbar=dict(title='FDR', tickvals=[1, 3, 5])
    ))
    
    fig.update_layout(
        title=dict(text='Fixture Difficulty', font=dict(color='#fff', size=16)),
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        margin=dict(l=100, r=40, t=60, b=40)
    )
    
    return fig


def create_form_timeline(players_df: pd.DataFrame, player_ids: List[int]) -> Optional[go.Figure]:
    """Create form timeline bar chart."""
    df = players_df[players_df['id'].isin(player_ids)].copy()
    if df.empty:
        return None
    
    df['form'] = safe_numeric(df.get('form', pd.Series([0]*len(df))))
    df = df.nlargest(15, 'form')
    
    colors = ['#22c55e' if f >= 5 else '#f59e0b' if f >= 3 else '#ef4444' for f in df['form']]
    
    fig = go.Figure(data=go.Bar(
        x=df['web_name'],
        y=df['form'],
        marker_color=colors,
        text=df['form'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(color='#ccc', size=10),
        margin=dict(l=40, r=40, t=20, b=80),
        xaxis_tickangle=-45
    )
    
    return fig


def create_budget_breakdown_pie(players_df: pd.DataFrame, player_ids: List[int]) -> Optional[go.Figure]:
    """Create budget breakdown pie chart by position."""
    df = players_df[players_df['id'].isin(player_ids)].copy()
    if df.empty:
        return None
    
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    costs = df.groupby('position')['now_cost'].sum()
    
    fig = go.Figure(data=go.Pie(
        labels=costs.index,
        values=costs.values,
        marker=dict(colors=[POSITION_COLORS.get(p, '#888') for p in costs.index]),
        textinfo='label+percent',
        textfont=dict(color='#fff')
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    
    return fig


def create_ownership_trends_chart(players_df: pd.DataFrame, limit: int = 20) -> Optional[go.Figure]:
    """Create ownership trends bar chart."""
    df = players_df.copy()
    df['transfers_in_event'] = safe_numeric(df.get('transfers_in_event', pd.Series([0]*len(df))))
    df['transfers_out_event'] = safe_numeric(df.get('transfers_out_event', pd.Series([0]*len(df))))
    df['total_transfers'] = df['transfers_in_event'] + df['transfers_out_event']
    
    top_movers = df.nlargest(limit, 'total_transfers').copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=top_movers['transfers_in_event'],
        name='In',
        marker_color='#22c55e'
    ))
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=-top_movers['transfers_out_event'],
        name='Out',
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        barmode='relative',
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=80),
        xaxis_tickangle=-45
    )
    
    return fig
