"""
FPL Strategy Engine - Streamlit Application
A high-performance dashboard for Fantasy Premier League 2025/26 season.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict

# Import custom modules
from fpl_api import (
    FPLDataFetcher, FPLDataProcessor, OddsDataFetcher,
    RivalScout, FPLAPIError, create_data_pipeline, get_differential_picks,
    CBIT_BONUS_THRESHOLD, CBIT_BONUS_POINTS, MAX_FREE_TRANSFERS, CAPTAIN_MULTIPLIER
)
from optimizer import (
    FPLOptimizer, OptimizationMode, OptimizationConstraints,
    DefConEngine
)

# Page configuration
st.set_page_config(
    page_title="FPL Strategy Engine 2025/26",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS - Black with white text and red accents
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .stApp { background: #0d0d0d; }
    .main-header { font-size: 2rem; font-weight: 600; color: #fff; text-align: center; padding: 1rem 0; }
    .sub-header { font-size: 0.9rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .section-title { font-size: 1.2rem; font-weight: 500; color: #fff; margin: 1.5rem 0 1rem 0; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .rule-card { background: #1a1a1a; border: 1px solid #333; padding: 1rem; margin: 0.5rem 0; text-align: center; }
    .rule-value { font-size: 1.5rem; font-weight: 600; color: #ef4444; }
    .rule-label { color: #888; font-size: 0.85rem; }
    .status-bar { background: #1a1a1a; border: 1px solid #333; padding: 0.5rem 1rem; text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1a1a; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { background: #ef4444 !important; color: #fff !important; }
    .stButton > button { background: #ef4444; color: #fff; border: none; padding: 0.5rem 1.5rem; }
    .stButton > button:hover { background: #dc2626; }
    [data-testid="stMetricValue"] { color: #fff; }
    [data-testid="stMetricLabel"] { color: #888; }
    .stTextInput input, .stNumberInput input { background: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }
    .stSelectbox > div > div { background: #1a1a1a !important; color: #fff !important; }
    .stSlider label { color: #fff !important; }
    p, span, label, .stMarkdown { color: #ccc; }
    h1, h2, h3, h4, h5, h6 { color: #fff !important; }
    .stDataFrame { background: #1a1a1a; }
    [data-testid="stDataFrame"] { background: #1a1a1a; }
    .stAlert { background: #1a1a1a !important; border: 1px solid #333 !important; color: #ccc !important; }
    .stExpander { background: #1a1a1a; border: 1px solid #333; }
    [data-testid="stExpander"] summary { color: #fff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=300, show_spinner=False)
def load_fpl_data():
    """Load FPL data with caching."""
    try:
        fetcher, processor = create_data_pipeline()
        return fetcher, processor, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(ttl=300, show_spinner=False)
def get_featured_data(_processor, weeks_ahead: int = 5):
    """Get engineered features with caching."""
    try:
        return _processor.get_engineered_features_df(weeks_ahead), None
    except Exception as e:
        return None, str(e)


def safe_numeric(series, default=0):
    """Safely convert series to numeric."""
    return pd.to_numeric(series, errors='coerce').fillna(default)


def get_player_name(player_id: int, players_df: pd.DataFrame) -> str:
    """Get player name from ID."""
    player = players_df[players_df['id'] == player_id]
    if not player.empty:
        return player.iloc[0]['web_name']
    return f"Unknown ({player_id})"


def create_ep_ownership_scatter(players_df: pd.DataFrame, position_filter: str = "All", search_player: str = ""):
    """Create interactive Expected Points vs Ownership scatter plot - dark theme."""
    df = players_df.copy()
    
    if position_filter != "All":
        df = df[df['position'] == position_filter]
    
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 90]
    
    df['selected_by_percent'] = safe_numeric(df.get('selected_by_percent', pd.Series([0]*len(df))))
    
    # Use ep_next or expected_points, whichever exists
    if 'expected_points' in df.columns:
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
    
    color_map = {
        'GKP': '#3b82f6',
        'DEF': '#22c55e',
        'MID': '#f59e0b',
        'FWD': '#ef4444'
    }
    
    # When searching: grey out all non-matching, highlight matching
    if is_searching:
        # Non-matching players - grey and small
        non_match = df[~df['is_searched']]
        if not non_match.empty:
            fig.add_trace(go.Scatter(
                x=non_match['selected_by_percent'],
                y=non_match['ep'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='#444',
                    opacity=0.2
                ),
                name='Others',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Matching players - full color with labels
        match = df[df['is_searched']]
        if not match.empty:
            for pos in match['position'].unique():
                pos_match = match[match['position'] == pos]
                fig.add_trace(go.Scatter(
                    x=pos_match['selected_by_percent'],
                    y=pos_match['ep'],
                    mode='markers+text',
                    marker=dict(
                        size=pos_match['now_cost'] * 2,
                        color=color_map.get(pos, '#ef4444'),
                        opacity=1.0,
                        line=dict(width=2, color='#fff')
                    ),
                    text=pos_match['web_name'],
                    textposition='top center',
                    textfont=dict(size=11, color='#fff'),
                    name=pos,
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>'
                ))
    else:
        # Normal view - show top players per position with labels
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = df[df['position'] == pos]
            if pos_df.empty:
                continue
            
            # Top 3 get labels
            top_3_idx = pos_df.nlargest(3, 'ep').index
            pos_df = pos_df.copy()
            pos_df['is_top'] = pos_df.index.isin(top_3_idx)
            
            # Non-top players
            non_top = pos_df[~pos_df['is_top']]
            if not non_top.empty:
                fig.add_trace(go.Scatter(
                    x=non_top['selected_by_percent'],
                    y=non_top['ep'],
                    mode='markers',
                    marker=dict(
                        size=non_top['now_cost'] * 1.2,
                        color=color_map[pos],
                        opacity=0.5
                    ),
                    name=pos,
                    showlegend=False,
                    hovertemplate='<b>%{customdata}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>',
                    customdata=non_top['web_name']
                ))
            
            # Top players with labels
            top = pos_df[pos_df['is_top']]
            if not top.empty:
                fig.add_trace(go.Scatter(
                    x=top['selected_by_percent'],
                    y=top['ep'],
                    mode='markers+text',
                    marker=dict(
                        size=top['now_cost'] * 1.5,
                        color=color_map[pos],
                        opacity=1.0
                    ),
                    text=top['web_name'],
                    textposition='top center',
                    textfont=dict(size=9, color='#ccc'),
                    name=pos,
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Ownership: %{x:.1f}%<br>EP: %{y:.1f}<extra></extra>'
                ))
    
    fig.add_hline(y=avg_ep, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Avg EP", annotation_font_color="#888")
    fig.add_vline(x=avg_own, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Avg Own%", annotation_font_color="#888")
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ccc'),
        xaxis_title='Ownership %',
        yaxis_title='Expected Points',
        xaxis=dict(gridcolor='#333', zerolinecolor='#333'),
        yaxis=dict(gridcolor='#333', zerolinecolor='#333'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#ccc')
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_cbit_chart(players_df: pd.DataFrame):
    """Create CBIT analysis chart."""
    df = players_df[players_df['position'] == 'DEF'].copy()
    
    if 'cbit_propensity' not in df.columns:
        # Create a proxy score from available data
        if 'clean_sheets' in df.columns:
            df['cbit_propensity'] = safe_numeric(df['clean_sheets']) / 10
        else:
            return None
    
    df['cbit_propensity'] = safe_numeric(df['cbit_propensity'])
    df = df.nlargest(15, 'cbit_propensity')
    
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['web_name'],
        y=df['cbit_propensity'],
        marker=dict(
            color=df['cbit_propensity'].tolist(),
            colorscale=[[0, '#22c55e'], [1, '#00ff87']]
        ),
        text=df['cbit_propensity'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='CBIT Propensity Score', font=dict(color='#fff', size=16)),
        xaxis_title='',
        yaxis_title='Score',
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ccc'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_fixture_heatmap(players_df: pd.DataFrame, player_ids: List[int]):
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
        colorscale=[
            [0, '#22c55e'],
            [0.5, '#f59e0b'],
            [1, '#ef4444']
        ],
        showscale=True,
        colorbar=dict(title='FDR', tickvals=[1,3,5])
    ))
    
    fig.update_layout(
        title=dict(text='Fixture Difficulty', font=dict(color='#fff', size=16)),
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ccc'),
        margin=dict(l=100, r=40, t=60, b=40)
    )
    
    return fig


def render_strategy_tab(processor: FPLDataProcessor, players_df: pd.DataFrame):
    """Strategy tab - EP vs Ownership visualization with filters."""
    
    # Season rules at top
    st.markdown('<p class="section-title">2025/26 Season Rules</p>', unsafe_allow_html=True)
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="rule-card">
            <div class="rule-value">{MAX_FREE_TRANSFERS}</div>
            <div class="rule-label">Max Free Transfers</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="rule-card">
            <div class="rule-value">{CAPTAIN_MULTIPLIER}x</div>
            <div class="rule-label">Captain Multiplier</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="rule-card">
            <div class="rule-value">+{CBIT_BONUS_POINTS}</div>
            <div class="rule-label">CBIT Bonus ({CBIT_BONUS_THRESHOLD} actions)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Player Landscape</p>', unsafe_allow_html=True)
    
    # Filters row 1
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    
    with f1:
        pos_filter = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="strat_pos")
    with f2:
        max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="strat_price")
    with f3:
        horizon = st.slider("Horizon (GWs)", 3, 10, 5, key="strat_horizon")
    with f4:
        search_player = st.text_input("Search player", placeholder="Type name to highlight...", key="strat_search")
    
    # Get featured data based on horizon
    featured_df, error = get_featured_data(processor, horizon)
    if error or featured_df is None:
        df = players_df.copy()
    else:
        df = featured_df.copy()
    
    # Apply filters
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df = df[df['now_cost'] <= max_price]
    
    if pos_filter != 'All':
        df = df[df['position'] == pos_filter]
    
    # Create and display scatter plot
    if 'expected_points' in df.columns or 'ep_next' in df.columns:
        fig = create_ep_ownership_scatter(df, pos_filter, search_player=search_player)
        if fig:
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Data loading...")
    
    # Quick stats
    st.markdown('<p class="section-title">Quick Stats</p>', unsafe_allow_html=True)
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    
    # Ensure numeric types for stats
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    with stat1:
        top_owned = df.nlargest(1, 'selected_by_percent')
        if not top_owned.empty:
            st.metric("Most Owned", top_owned.iloc[0]['web_name'], f"{top_owned.iloc[0]['selected_by_percent']:.1f}%")
    
    with stat2:
        if 'ep_next' in df.columns:
            df['ep_next'] = safe_numeric(df['ep_next'])
            top_ep = df.nlargest(1, 'ep_next')
            if not top_ep.empty:
                st.metric("Top EP", top_ep.iloc[0]['web_name'], f"{top_ep.iloc[0]['ep_next']:.1f}")
    
    with stat3:
        total_players = len(df)
        st.metric("Players Shown", total_players)
    
    with stat4:
        avg_price = df['now_cost'].mean()
        st.metric("Avg Price", f"{avg_price:.1f}m")


def render_rival_tab(processor: FPLDataProcessor, players_df: pd.DataFrame):
    """Rival Scout tab content."""
    
    c1, c2 = st.columns(2)
    
    with c1:
        your_id = st.number_input("Your Team ID", min_value=1, max_value=10000000, value=1, key="your_id")
    with c2:
        rival_id = st.number_input("Rival Team ID", min_value=1, max_value=10000000, value=2, key="rival_id")
    
    if st.button("Compare Teams", type="primary", width="stretch"):
        if your_id == rival_id:
            st.warning("Enter different Team IDs")
            return
        
        with st.spinner("Analyzing..."):
            try:
                scout = RivalScout(processor.fetcher, processor)
                result = scout.compare_teams(your_id, rival_id)
                
                if result['valid']:
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric("Jaccard Similarity", f"{result['jaccard_similarity']*100:.1f}%")
                    with m2:
                        st.metric("Common Players", result['overlap_count'])
                    with m3:
                        if result['biggest_threat']:
                            st.metric("Biggest Threat", result['biggest_threat']['name'])
                        else:
                            st.metric("Biggest Threat", "None")
                    
                    st.markdown('<p class="section-title">Squad Comparison</p>', unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Your Unique', 'Shared', 'Rival Unique'],
                        y=[len(result['unique_to_you']), result['overlap_count'], len(result['unique_to_rival'])],
                        marker_color=['#22c55e', '#f59e0b', '#ef4444'],
                        text=[len(result['unique_to_you']), result['overlap_count'], len(result['unique_to_rival'])],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        height=300,
                        template='plotly_dark',
                        paper_bgcolor='#0d0d0d',
                        plot_bgcolor='#1a1a1a',
                        font=dict(color='#ccc'),
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Your Unique Players**")
                        for pid in result['unique_to_you']:
                            st.write(f"- {get_player_name(pid, players_df)}")
                    with c2:
                        st.markdown("**Rival's Unique Players**")
                        for pid in result['unique_to_rival']:
                            st.write(f"- {get_player_name(pid, players_df)}")
                else:
                    st.error(result.get('error', 'Comparison failed'))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter Team IDs and click Compare to analyze rival squads")


def render_factory_tab(processor: FPLDataProcessor, players_df: pd.DataFrame):
    """Feature Factory tab content."""
    
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        pos_filter = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'])
    with f2:
        max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5)
    with f3:
        min_mins = st.slider("Min Minutes", 0, 1000, 90, 90)
    with f4:
        sort_col = st.selectbox("Sort By", ['Expected Points', 'Differential', 'Value', 'CBIT', 'Price'])
    
    search = st.text_input("Search player...", placeholder="Enter name")
    
    featured_df, error = get_featured_data(processor, 5)
    
    if error:
        st.warning(f"Using base data: {error}")
        featured_df = players_df.copy()
        # Add basic columns if missing
        if 'expected_points' not in featured_df.columns:
            featured_df['expected_points'] = safe_numeric(featured_df.get('ep_next', pd.Series([2.0]*len(featured_df))))
        if 'differential_score' not in featured_df.columns:
            featured_df['differential_score'] = 0.0
        if 'cbit_propensity' not in featured_df.columns:
            featured_df['cbit_propensity'] = 0.0
        if 'xg_per_pound' not in featured_df.columns:
            featured_df['xg_per_pound'] = featured_df['expected_points'] / safe_numeric(featured_df['now_cost'], 5).clip(lower=4)
    
    if featured_df is not None:
        df = featured_df.copy()
        
        if pos_filter != 'All':
            df = df[df['position'] == pos_filter]
        
        df['now_cost'] = safe_numeric(df['now_cost'], 5)
        df['minutes'] = safe_numeric(df['minutes'])
        
        df = df[df['now_cost'] <= max_price]
        df = df[df['minutes'] >= min_mins]
        
        if search:
            df = df[df['web_name'].str.lower().str.contains(search.lower(), na=False)]
        
        sort_map = {
            'Expected Points': 'expected_points',
            'Differential': 'differential_score',
            'Value': 'xg_per_pound',
            'CBIT': 'cbit_propensity',
            'Price': 'now_cost'
        }
        
        sort_by = sort_map.get(sort_col, 'expected_points')
        if sort_by in df.columns:
            df[sort_by] = safe_numeric(df[sort_by])
            df = df.sort_values(sort_by, ascending=False)
        
        display_cols = ['web_name', 'team_name', 'position', 'now_cost', 'selected_by_percent',
                       'expected_points', 'differential_score', 'cbit_propensity', 'xg_per_pound']
        display_cols = [c for c in display_cols if c in df.columns]
        
        # Ensure numeric columns are properly formatted
        numeric_cols = ['now_cost', 'selected_by_percent', 'expected_points', 'differential_score', 'cbit_propensity', 'xg_per_pound']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = safe_numeric(df[col])
        
        rename = {
            'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
            'now_cost': 'Price', 'selected_by_percent': 'Own%',
            'expected_points': 'EP', 'differential_score': 'Diff',
            'cbit_propensity': 'CBIT', 'xg_per_pound': 'Value'
        }
        
        # Round numeric columns for display
        display_df = df[display_cols].head(50).copy()
        for col in ['now_cost', 'selected_by_percent', 'expected_points', 'differential_score', 'cbit_propensity', 'xg_per_pound']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.markdown(f'<p class="section-title">Players ({len(df)} found)</p>', unsafe_allow_html=True)
        
        st.dataframe(
            display_df.rename(columns=rename),
            width="stretch",
            hide_index=True
        )
        
        v1, v2 = st.columns(2)
        
        with v1:
            st.markdown('<p class="section-title">EP vs Ownership</p>', unsafe_allow_html=True)
            scatter_search_ff = st.text_input("Highlight player", placeholder="Search...", key="ff_scatter_search")
            fig = create_ep_ownership_scatter(df, pos_filter, search_player=scatter_search_ff)
            if fig:
                st.plotly_chart(fig, width="stretch")
        
        with v2:
            if pos_filter in ['All', 'DEF']:
                st.markdown('<p class="section-title">CBIT Analysis</p>', unsafe_allow_html=True)
                fig = create_cbit_chart(df)
                if fig:
                    st.plotly_chart(fig, width="stretch")
            else:
                st.markdown('<p class="section-title">Price vs EP</p>', unsafe_allow_html=True)
                if 'expected_points' in df.columns and 'now_cost' in df.columns:
                    plot_df = df.head(50).copy()
                    plot_df['expected_points'] = safe_numeric(plot_df['expected_points'])
                    plot_df['now_cost'] = safe_numeric(plot_df['now_cost'], 5)
                    
                    fig = px.scatter(
                        plot_df,
                        x='now_cost', y='expected_points',
                        color='position', hover_data=['web_name'],
                        labels={'now_cost': 'Price', 'expected_points': 'EP'}
                    )
                    fig.update_layout(
                        height=350,
                        template='plotly_dark',
                        paper_bgcolor='#0d0d0d',
                        plot_bgcolor='#1a1a1a',
                        font=dict(color='#ccc'),
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig, width="stretch")
        
        st.markdown('<p class="section-title">Differential Picks (under 5% ownership)</p>', unsafe_allow_html=True)
        try:
            diffs = get_differential_picks(processor, min_ep=3.0, max_ownership=5.0)
            if diffs is not None and not diffs.empty:
                st.dataframe(diffs.head(10), width="stretch", hide_index=True)
            else:
                st.info("No differentials matching criteria")
        except Exception as e:
            st.warning(f"Could not load differentials: {e}")


def render_optimization_tab(processor: FPLDataProcessor, players_df: pd.DataFrame, fetcher):
    """Optimization tab - Transfer recommendations based on team."""
    
    st.markdown('<p class="section-title">Transfer Optimizer</p>', unsafe_allow_html=True)
    
    # Configuration
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        team_id = st.number_input("Your Team ID", min_value=1, max_value=10000000, value=1, key="opt_team_id")
    with c2:
        weeks_ahead = st.slider("Planning Horizon", min_value=3, max_value=10, value=5, key="opt_horizon")
    with c3:
        free_transfers = st.slider("Free Transfers Available", min_value=1, max_value=MAX_FREE_TRANSFERS, value=1, key="opt_fts")
    with c4:
        strategy = st.selectbox("Strategy", ['Balanced', 'Maximum Points', 'Differential', 'Value'], key="opt_strategy")
    
    if st.button("Analyze Transfers", type="primary"):
        with st.spinner("Analyzing your squad..."):
            try:
                # Get featured data with engineered features
                featured_df, error = get_featured_data(processor, weeks_ahead)
                
                if error or featured_df is None:
                    st.error(f"Error loading data: {error}")
                    return
                
                # Try to get user's current team
                try:
                    gw = fetcher.get_current_gameweek()
                    picks_data = fetcher.get_team_picks(team_id, gw)
                    # Extract player IDs from picks
                    if isinstance(picks_data, dict) and 'picks' in picks_data:
                        your_squad = set(p['element'] for p in picks_data['picks'])
                    elif isinstance(picks_data, list):
                        your_squad = set(p.get('element', p) for p in picks_data)
                    else:
                        your_squad = set()
                    has_real_team = len(your_squad) > 0
                except Exception as e:
                    your_squad = set()
                    has_real_team = False
                    st.warning(f"Could not fetch team data: {e}. Showing general recommendations.")
                
                # Calculate transfer scores for all players
                featured_df['transfer_score'] = 0.0
                
                # Get EP column
                if 'expected_points' in featured_df.columns:
                    ep_col = 'expected_points'
                elif 'ep_next' in featured_df.columns:
                    ep_col = 'ep_next'
                else:
                    featured_df['ep_next'] = 2.0
                    ep_col = 'ep_next'
                
                featured_df[ep_col] = safe_numeric(featured_df[ep_col])
                featured_df['selected_by_percent'] = safe_numeric(featured_df['selected_by_percent'])
                featured_df['now_cost'] = safe_numeric(featured_df['now_cost'], 5)
                
                # Calculate transfer score based on strategy
                if strategy == 'Maximum Points':
                    featured_df['transfer_score'] = featured_df[ep_col]
                elif strategy == 'Differential':
                    # High EP, low ownership
                    featured_df['transfer_score'] = featured_df[ep_col] / (featured_df['selected_by_percent'].clip(lower=0.5) ** 0.5)
                elif strategy == 'Value':
                    # EP per million
                    featured_df['transfer_score'] = featured_df[ep_col] / featured_df['now_cost'].clip(lower=4)
                else:  # Balanced
                    # Combination of factors
                    featured_df['transfer_score'] = (
                        featured_df[ep_col] * 0.5 +
                        (featured_df[ep_col] / featured_df['now_cost'].clip(lower=4)) * 2 +
                        (100 - featured_df['selected_by_percent'].clip(upper=100)) * 0.01
                    )
                
                # Display results
                st.markdown('<p class="section-title">Transfer Recommendations</p>', unsafe_allow_html=True)
                
                if has_real_team and your_squad:
                    # Players to transfer OUT (lowest scores in current squad)
                    current_squad_df = featured_df[featured_df['id'].isin(your_squad)].copy()
                    
                    if not current_squad_df.empty:
                        st.markdown("**Recommended OUT** (lowest value in your squad)", unsafe_allow_html=True)
                        out_candidates = current_squad_df.nsmallest(5, 'transfer_score')
                        
                        out_display = out_candidates[['web_name', 'team_name', 'position', 'now_cost', ep_col, 'transfer_score']].copy()
                        out_display.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Score']
                        out_display['Score'] = out_display['Score'].round(2)
                        out_display['EP'] = out_display['EP'].round(2)
                        st.dataframe(out_display, width="stretch", hide_index=True)
                        
                        # Players to transfer IN (not in squad, highest scores)
                        st.markdown("**Recommended IN** (best available)", unsafe_allow_html=True)
                        available_df = featured_df[~featured_df['id'].isin(your_squad)].copy()
                else:
                    st.markdown("**Top Transfer Targets**", unsafe_allow_html=True)
                    available_df = featured_df.copy()
                
                # Show top recommendations by position
                for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                    pos_df = available_df[available_df['position'] == pos].nlargest(5, 'transfer_score')
                    
                    if not pos_df.empty:
                        with st.expander(f"{pos} Recommendations", expanded=True):
                            display_df = pos_df[['web_name', 'team_name', 'now_cost', ep_col, 'selected_by_percent', 'transfer_score']].copy()
                            display_df.columns = ['Player', 'Team', 'Price', 'EP', 'Owned%', 'Score']
                            display_df['Score'] = display_df['Score'].round(2)
                            display_df['EP'] = display_df['EP'].round(2)
                            display_df['Owned%'] = display_df['Owned%'].round(1)
                            st.dataframe(display_df, width="stretch", hide_index=True)
                
                # Top overall picks
                st.markdown('<p class="section-title">Top 10 Overall Picks</p>', unsafe_allow_html=True)
                top_10 = available_df.nlargest(10, 'transfer_score')
                top_display = top_10[['web_name', 'team_name', 'position', 'now_cost', ep_col, 'selected_by_percent', 'transfer_score']].copy()
                top_display.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Owned%', 'Score']
                top_display['Score'] = top_display['Score'].round(2)
                top_display['EP'] = top_display['EP'].round(2)
                top_display['Owned%'] = top_display['Owned%'].round(1)
                st.dataframe(top_display, width="stretch", hide_index=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter your Team ID and click 'Analyze Transfers' to get personalized recommendations based on your current squad and selected strategy.")


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">FPL Strategy Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">2025/26 SEASON</div>', unsafe_allow_html=True)
    
    # Auto-load data
    with st.spinner("Loading data..."):
        fetcher, processor, error = load_fpl_data()
    
    if error:
        st.error(f"Failed to load FPL data: {error}")
        st.info("Please check your internet connection and refresh the page.")
        if st.button("Retry"):
            st.cache_resource.clear()
            st.rerun()
        return
    
    if processor is None:
        st.error("Could not initialize data processor")
        return
    
    # Get players data
    try:
        players_df = processor.players_df.copy()
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return
    
    # Show current gameweek in status bar
    try:
        gw = fetcher.get_current_gameweek()
        st.markdown(f'<div class="status-bar">Gameweek {gw} | {len(players_df)} Players Loaded</div>', unsafe_allow_html=True)
    except:
        st.markdown(f'<div class="status-bar">{len(players_df)} Players Loaded</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Strategy", "Optimization", "Rival Scout", "Feature Factory"])
    
    with tab1:
        render_strategy_tab(processor, players_df)
    
    with tab2:
        render_optimization_tab(processor, players_df, fetcher)
    
    with tab3:
        render_rival_tab(processor, players_df)
    
    with tab4:
        render_factory_tab(processor, players_df)


if __name__ == "__main__":
    main()
