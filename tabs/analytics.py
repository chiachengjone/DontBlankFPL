"""Analytics tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd

from utils.helpers import safe_numeric, style_df_with_injuries
from fpl_api import get_differential_picks


def render_analytics_tab(processor, players_df: pd.DataFrame):
    """Analytics tab - player discovery and advanced metrics."""
    
    # Filters
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        pos_filter = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="analytics_pos")
    with f2:
        max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="analytics_price")
    with f3:
        min_mins = st.slider("Min Minutes", 0, 1000, 90, 90, key="analytics_mins")
    with f4:
        sort_col = st.selectbox("Sort By", ['Expected Points', 'Differential', 'Value', 'CBIT', 'Price'], key="analytics_sort")
    
    search = st.text_input("Search player...", placeholder="Enter name", key="analytics_search")
    
    # Prepare data
    df = players_df.copy()
    df['expected_points'] = safe_numeric(df.get('ep_next', pd.Series([2.0]*len(df))))
    df['differential_score'] = df['expected_points'] / safe_numeric(df['selected_by_percent'], 5).clip(lower=0.5)
    df['cbit_propensity'] = 0.0
    if 'clean_sheets' in df.columns:
        df.loc[df['position'] == 'DEF', 'cbit_propensity'] = safe_numeric(df['clean_sheets']) / 10
    df['xg_per_pound'] = df['expected_points'] / safe_numeric(df['now_cost'], 5).clip(lower=4)
    
    # Apply filters
    if pos_filter != 'All':
        df = df[df['position'] == pos_filter]
    
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['minutes'] = safe_numeric(df['minutes'])
    df = df[df['now_cost'] <= max_price]
    df = df[df['minutes'] >= min_mins]
    
    if search:
        df = df[df['web_name'].str.lower().str.contains(search.lower(), na=False)]
    
    # Sort
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
    
    # Display player table
    render_player_table(df)
    
    # Differential picks
    render_differential_picks(processor)
    
    # Set & Forget finder
    render_set_and_forget(players_df)
    
    # Expected vs Actual
    render_expected_vs_actual(players_df)


def render_player_table(df: pd.DataFrame):
    """Render the main player table."""
    display_cols = ['web_name', 'team_name', 'position', 'now_cost', 'selected_by_percent',
                   'expected_points', 'differential_score', 'cbit_propensity', 'xg_per_pound']
    display_cols = [c for c in display_cols if c in df.columns]
    
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
    
    display_df = df[display_cols].head(50).copy()
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    st.markdown(f'<p class="section-title">Players ({len(df)} found)</p>', unsafe_allow_html=True)
    st.dataframe(style_df_with_injuries(display_df.rename(columns=rename)), hide_index=True, use_container_width=True)


def render_differential_picks(processor):
    """Render differential picks section."""
    st.markdown('<p class="section-title">Differential Picks (under 5% ownership)</p>', unsafe_allow_html=True)
    try:
        diffs = get_differential_picks(processor, min_ep=3.0, max_ownership=5.0)
        if diffs is not None and not diffs.empty:
            st.dataframe(style_df_with_injuries(diffs.head(10)), hide_index=True, use_container_width=True)
        else:
            st.info("No differentials matching criteria")
    except Exception as e:
        st.warning(f"Could not load differentials: {e}")


def render_set_and_forget(players_df: pd.DataFrame):
    """Render Set & Forget finder."""
    st.markdown('<p class="section-title">Set & Forget Picks</p>', unsafe_allow_html=True)
    st.caption("Players with high EP, good fixture run, and consistent minutes - minimal rotation needed")
    
    df = players_df.copy()
    df['ep_next'] = safe_numeric(df.get('ep_next', pd.Series([0]*len(df))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df['form'] = safe_numeric(df.get('form', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    
    df['minutes_reliability'] = df['minutes'].clip(upper=2500) / 2500
    df['sf_score'] = df['ep_next'] * 0.5 + df['form'] * 0.2 + df['minutes_reliability'] * 10 * 0.3
    
    sf_cols = st.columns(4)
    
    for i, pos in enumerate(['GKP', 'DEF', 'MID', 'FWD']):
        with sf_cols[i]:
            st.markdown(f"**{pos}**")
            pos_df = df[df['position'] == pos].nlargest(3, 'sf_score')[['web_name', 'now_cost', 'sf_score']]
            pos_df.columns = ['Player', 'Price', 'S&F Score']
            pos_df['S&F Score'] = pos_df['S&F Score'].round(1)
            st.dataframe(style_df_with_injuries(pos_df), hide_index=True, use_container_width=True)


def render_expected_vs_actual(players_df: pd.DataFrame):
    """Render Expected vs Actual performance analysis."""
    st.markdown('<p class="section-title">Expected vs Actual (Over/Under Performers)</p>', unsafe_allow_html=True)
    st.caption("Comparing total points vs what was expected - find unlucky players worth targeting")
    
    df = players_df.copy()
    df['total_points'] = safe_numeric(df.get('total_points', pd.Series([0]*len(df))))
    df['ep_next'] = safe_numeric(df.get('ep_next', pd.Series([0]*len(df))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 500]
    
    df['games_played'] = df['minutes'] / 90
    df['expected_total'] = df['ep_next'] * df['games_played'] * 0.6
    df['diff'] = df['total_points'] - df['expected_total']
    
    xva1, xva2 = st.columns(2)
    
    with xva1:
        st.markdown("**Underperformers** (unlucky, may bounce back)")
        under = df.nsmallest(8, 'diff')[['web_name', 'team_name', 'total_points', 'expected_total', 'diff']]
        under.columns = ['Player', 'Team', 'Actual Pts', 'Expected', 'Diff']
        under['Expected'] = under['Expected'].round(0).astype(int)
        under['Diff'] = under['Diff'].round(0).astype(int)
        st.dataframe(style_df_with_injuries(under), hide_index=True, use_container_width=True)
    
    with xva2:
        st.markdown("**Overperformers** (may regress)")
        over = df.nlargest(8, 'diff')[['web_name', 'team_name', 'total_points', 'expected_total', 'diff']]
        over.columns = ['Player', 'Team', 'Actual Pts', 'Expected', 'Diff']
        over['Expected'] = over['Expected'].round(0).astype(int)
        over['Diff'] = over['Diff'].round(0).astype(int)
        st.dataframe(style_df_with_injuries(over), hide_index=True, use_container_width=True)
