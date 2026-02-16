"""ML Predictions Tab - Machine Learning player performance forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import safe_numeric, round_df, style_df_with_injuries, normalize_name


def render_ml_tab(processor, players_df: pd.DataFrame):
    """Machine Learning Predictions tab — intuitive UI for all users."""

    # Metrics explanation dropdown
    with st.expander("Understanding ML Predictions"):
        st.markdown("""
        **What is ML Prediction?**
        - Machine learning model trained on historical FPL data
        - Uses XGBoost ensemble with 5-fold cross-validation
        - Predicts xP for upcoming gameweeks
        
        **Key Metrics**
        - **ML xP**: Model's predicted points for the player
        - **FPL xP**: Official FPL expected points estimate
        - **vs FPL**: Difference between ML and FPL predictions
        - **Poisson xP**: Statistical model using xG/xA data
        - **Certainty %**: Model confidence (higher = more reliable)
        
        **Understanding Certainty**
        - 80%+: High confidence, trust the prediction
        - 60-80%: Moderate confidence, consider other factors
        - <60%: Low confidence, player is unpredictable
        
        **Model Features Used**
        - Form, ICT Index, minutes, fixture difficulty
        - Historical points per game, home/away splits
        - Opponent defensive strength, recent results
        
        **How to Use**
        - Compare ML vs FPL xP for value identification
        - High ML, Low FPL = potential differential
        - Low certainty = high variance player (risky captain)
        
        **Model Performance (shown after running)**
        - MAE: Average prediction error in points
        - R²: How well model explains point variance (higher = better)
        """)

    # ── Controls ──
    current_horizon = st.session_state.get('pref_weeks_ahead', 1)
    ml_label = f"ML xP x{current_horizon}" if current_horizon > 1 else "ML xP"
    fpl_label = f"FPL xP x{current_horizon}" if current_horizon > 1 else "FPL xP"
    poisson_label = f"Poisson xP ({current_horizon}GW)" if current_horizon > 1 else "Poisson xP"
    
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    
    with ctrl1:
        position_filter = st.selectbox(
            "Position",
            options=["All", "GKP", "DEF", "MID", "FWD"],
            key="ml_pos_filter"
        )
    
    with ctrl2:
        price_range = st.slider(
            "Price Range (£m)",
            4.0, 15.0, (4.0, 15.0), 0.1,
            key="ml_price_range"
        )

    with ctrl3:
        min_minutes = st.slider(
            "Min Minutes Played",
            0, 3500, 0, 90,
            key="ml_min_minutes"
        )
    
    ctrl4, ctrl5, ctrl6 = st.columns([1, 1, 1])
    
    with ctrl4:
        all_teams = sorted(players_df['team_name'].dropna().unique().tolist()) if 'team_name' in players_df.columns else []
        ml_team_filter = st.selectbox("Team", ['All'] + all_teams, key="ml_team_filter")
        
    with ctrl5:
        ml_sort_col = st.selectbox(
            "Sort By",
            [ml_label, fpl_label, poisson_label, "vs FPL (+)", "Certainty"],
            key="ml_sort"
        )
    
    with ctrl6:
        n_gameweeks = st.slider(
            "xP Horizon (GWs)", 1, 5,
            value=current_horizon,
            help="Sum xP over the next N gameweeks using full engine",
            key="ml_horizon_slider"
        )
        if n_gameweeks != current_horizon:
            st.session_state.pref_weeks_ahead = n_gameweeks
            # Clear cached players_df to force recalculation in app.py
            if 'players_df' in st.session_state:
                del st.session_state.players_df
            st.rerun()
        
        horizon = current_horizon # Use consistent state for current run
    
    search_player_input = st.text_input(
        "Search player",
        placeholder="Type player name...",
        key="ml_player_search"
    )
    

    # ── Results section ──
    if "ml_predictions" in st.session_state:
        predictions = st.session_state["ml_predictions"]
        gws = st.session_state.get("ml_gws", 1)
        cv_scores = st.session_state.get("ml_cv_scores", {})
        
        # Build results dataframe
        pred_data = []
        
        # Calculate global stats for relative uncertainty
        all_preds = [p.predicted_points for p in predictions.values()]
        all_stds = [p.prediction_std for p in predictions.values()]
        mean_pred = np.mean(all_preds) if all_preds else 2.0
        std_of_preds = np.std(all_preds) if all_preds else 1.0
        
        for pid, pred in predictions.items():
            match = players_df[players_df["id"] == pid]
            if match.empty:
                continue
            player = match.iloc[0]
            
            # Get raw metrics for scaling
            fpl_ep_raw = safe_numeric(pd.Series([player.get('ep_next_num', player.get('ep_next', 0))])).iloc[0]
            ml_pred = pred.predicted_points
            
            # Certainty calculation
            ci_low, ci_high = pred.confidence_interval
            ci_width = ci_high - ci_low
            if ml_pred > 0.5:
                relative_uncertainty = (ci_width / ml_pred) * 100
            else:
                relative_uncertainty = ci_width * 50
            certainty = max(0, min(100, 100 - relative_uncertainty))

            # Scale metrics by horizon - match Analytics tab simple scaling for comparison
            horizon = n_gameweeks
            
            # Use pre-calculated Poisson EP from engine (already totaled for horizon)
            poisson_ep = safe_numeric(player.get('expected_points_poisson', 0.0))
            
            # FPL EP scaling (simple multiplier to match Analytics tab display)
            fpl_ep = fpl_ep_raw * horizon
            
            # ML Pred scaling (simple multiplier to match Analytics tab comparison)
            ml_pred_total = ml_pred * horizon
            
            # Certainty scaling (reduces with distance)
            certainty_base = certainty # From previous calculation
            certainty_horizon = certainty_base * (0.95 ** (horizon - 1))
            
            # Scale Range (uncertainty) based on horizon
            # Sum range linearly as it represents the bounds of total points
            # Scaling factors for decay still apply
            scaled_ci_low = 0.0
            scaled_ci_high = 0.0
            for i in range(horizon):
                decay = 0.92 ** i
                scaled_ci_low += ci_low * decay
                scaled_ci_high += ci_high * decay

            pred_data.append({
                "id": pid,
                "Player": player.get("web_name", f"ID:{pid}"),
                "Pos": player.get("position", "?"),
                "Team": player.get("team_name", "?"),
                "Price": round(float(player.get("now_cost", 0)), 1),
                "Minutes": player.get("minutes", 0),
                "ML xP": round(ml_pred_total, 2),
                "Avg ML": round(ml_pred_total / horizon, 2) if horizon > 1 else round(ml_pred_total, 2),
                "Range": f"{scaled_ci_low:.2f}-{scaled_ci_high:.2f}",
                "FPL xP": round(fpl_ep, 2),
                "Poisson xP": round(poisson_ep, 2),
                "vs FPL": round(ml_pred_total - fpl_ep, 2),
                "vs Poisson": round(ml_pred_total - poisson_ep, 2),
                "Certainty": round(certainty_horizon, 0),
                "EO%": player.get("selected_by_percent", 0),
                "_ci_low": scaled_ci_low,
                "_ci_high": scaled_ci_high,
                "_uncertainty": pred.prediction_std * sum([0.92 ** i for i in range(horizon)]),
                # Raw sorting keys matching dynamic labels
                f"ML xP x{horizon}" if horizon > 1 else "ML xP": ml_pred_total,
                f"FPL xP x{horizon}" if horizon > 1 else "FPL xP": fpl_ep,
                "Poisson xP": poisson_ep, # Match Analytics tab internal ID
                poisson_label: poisson_ep,
            })

        pred_df = pd.DataFrame(pred_data)
        
        # ── Player-specific view ──
        if search_player_input and search_player_input.strip():
            q = normalize_name(search_player_input.lower().strip())
            
            # Match if query starts with beginning of either name on the original players_df
            # then find the corresponding ID in pred_df
            if 'first_normalized' not in players_df.columns:
                players_df['first_normalized'] = players_df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
                players_df['second_normalized'] = players_df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
            
            matched_ids = players_df[
                (players_df['first_normalized'].str.startswith(q, na=False)) |
                (players_df['second_normalized'].str.startswith(q, na=False))
            ]["id"].tolist()
            
            player_row = pred_df[pred_df["id"].isin(matched_ids)]
            
            if not player_row.empty:
                player_row = player_row.head(1)
                st.markdown(f"### ML Analysis: {player_row.iloc[0]['web_name']}")
                
                p = player_row.iloc[0]
                pid = p["id"]
                pred_obj = predictions.get(pid)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ML xP", f"{p['ML xP']:.2f} pts")
                with col2:
                    delta_color = "normal" if p["vs FPL"] >= 0 else "inverse"
                    st.metric("vs FPL", f"{p['FPL xP']:.2f} pts", 
                              f"{p['vs FPL']:+.2f}", delta_color=delta_color)
                with col3:
                    delta_color = "normal" if p["vs Poisson"] >= 0 else "inverse"
                    st.metric("vs Poisson", f"{p['Poisson xP']:.2f} pts", 
                              f"{p['vs Poisson']:+.2f}", delta_color=delta_color)
                with col4:
                    st.metric("Prediction Range", p["Range"])
                
                # Feature breakdown for this player
                if pred_obj and pred_obj.feature_importance:
                    st.markdown("**Key factors for this player:**")
                    
                    # Get top 5 features
                    importance = pred_obj.feature_importance
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    feature_names = {
                        'fpl_ep': 'FPL Expected Pts',
                        'form_numeric': 'Form Score',
                        'ppg': 'Points/Game',
                        'form_momentum': 'Form Momentum',
                        'ict_index': 'ICT Index',
                        'influence': 'Influence',
                        'creativity': 'Creativity',
                        'threat': 'Threat',
                        'ict_per_90': 'ICT/90',
                        'threat_per_90': 'Threat/90',
                        'minutes_per_start': 'Minutes/Start',
                        'nailed_score': 'Nailed Score',
                        'price': 'Price',
                        'ownership': 'Ownership',
                        'value_score': 'Value',
                        'team_strength': 'Team Quality',
                    }
                    
                    factor_data = []
                    for feat, imp in sorted_imp:
                        display_name = feature_names.get(feat, feat.replace('_', ' ').title())
                        factor_data.append({'Feature': display_name, 'Importance': imp})
                    
                    factor_df = pd.DataFrame(factor_data).sort_values('Importance', ascending=True)
                    
                    fig = go.Figure(go.Bar(
                        x=factor_df['Importance'],
                        y=factor_df['Feature'],
                        orientation='h',
                        marker_color='#3b82f6',
                        text=factor_df['Importance'].apply(lambda x: f'{x:.1%}'),
                        textposition='outside',
                        textfont=dict(size=11, color='#1d1d1f'),
                    ))
                    fig.update_layout(
                        height=max(180, len(factor_data) * 36),
                        template='plotly_white',
                        paper_bgcolor='#ffffff',
                        plot_bgcolor='#ffffff',
                        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
                        xaxis=dict(
                            tickformat='.0%',
                            gridcolor='#e5e5ea',
                            range=[0, max(factor_df['Importance']) * 1.3],
                        ),
                        yaxis=dict(gridcolor='#e5e5ea'),
                        margin=dict(l=10, r=40, t=10, b=20),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig, use_container_width=True, key='ml_player_factors')
                
                st.markdown("---")
        
        # ── Filter Results ──
        if ml_team_filter != 'All':
            pred_df = pred_df[pred_df["Team"] == ml_team_filter]
            
        # Price and Minutes Sliders
        pred_df = pred_df[
            (pred_df['Price'] >= price_range[0]) & 
            (pred_df['Price'] <= price_range[1]) &
            (pred_df['Minutes'] >= min_minutes)
        ]

        # Search logic (starts with first or last name)
        if search_player_input.strip():
            q = normalize_name(search_player_input.lower().strip())
            if 'first_normalized' not in players_df.columns:
                players_df['first_normalized'] = players_df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
                players_df['second_normalized'] = players_df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
                
            matched_ids = players_df[
                (players_df['first_normalized'].str.startswith(q, na=False)) |
                (players_df['second_normalized'].str.startswith(q, na=False))
            ]["id"].tolist()
            pred_df = pred_df[pred_df["id"].isin(matched_ids)]

        sort_col = ml_sort_col
        pred_df = pred_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        # ── Quick Insights ──
        st.markdown("### Quick Picks")
        
        qp1, qp2, qp3, qp4 = st.columns(4)
        
        top_pick = pred_df.head(1)
        if not top_pick.empty:
            with qp1:
                st.metric(
                    "Top Pick",
                    top_pick.iloc[0]["Player"],
                    f"{top_pick.iloc[0]['ML xP']:.2f} pts",
                    help="Highest ML predicted points"
                )
        
        # Best value (ML xP / price)
        pred_df["_value"] = pred_df["ML xP"] / pred_df["Price"].clip(lower=4)
        best_value = pred_df.nlargest(1, "_value")
        if not best_value.empty:
            with qp2:
                st.metric(
                    "Best Value",
                    best_value.iloc[0]["Player"],
                    f"£{best_value.iloc[0]['Price']}m",
                    help="Best ML prediction per £"
                )
        
        # Biggest upside vs FPL
        biggest_upside = pred_df.nlargest(1, "vs FPL")
        if not biggest_upside.empty:
            with qp3:
                st.metric(
                    "Underpriced",
                    biggest_upside.iloc[0]["Player"],
                    f"{biggest_upside.iloc[0]['vs FPL']:+.2f} vs FPL",
                    help="ML thinks this player is underrated"
                )
        
        # Most certain high prediction
        high_certainty = pred_df[pred_df["Certainty"] >= 50].nlargest(1, "ML xP")
        if not high_certainty.empty:
            with qp4:
                st.metric(
                    "Safest Pick",
                    high_certainty.iloc[0]["Player"],
                    f"{high_certainty.iloc[0]['Certainty']:.0f}% certain",
                    help="High prediction with narrow range"
                )
        
        # ── Full results table ──
        st.markdown(f"### ML Predictions (next {gws} GW{'s' if gws > 1 else ''}) — {len(pred_df)} found")
        
        # Prepare display dataframe
        poisson_label = f"Poisson xP ({horizon}GW)" if horizon > 1 else "Poisson xP"
        
        display_cols = ["Player", "Pos", "Team", "Price", "Minutes", ml_label]
        if horizon > 1:
            display_cols.append("Avg ML")
        display_cols.extend(["Range", fpl_label, "vs FPL", poisson_label, "vs Poisson", "Certainty"])
        
        full_df = pred_df[display_cols].copy()
        
        # Use column_config for formatting
        col_config = {
            "Price": st.column_config.NumberColumn("Price", format="£%.1fm"),
            "Minutes": st.column_config.NumberColumn("Mins", format="%d"),
            ml_label: st.column_config.NumberColumn(ml_label, format="%.2f"),
            fpl_label: st.column_config.NumberColumn(fpl_label, format="%.2f"),
            "vs FPL": st.column_config.NumberColumn("vs FPL", format="%+.2f"),
            poisson_label: st.column_config.NumberColumn(poisson_label, format="%.2f"),
            "vs Poisson": st.column_config.NumberColumn("vs Poisson", format="%+.2f"),
            "Certainty": st.column_config.NumberColumn("Certainty", format="%.0f%%"),
        }
        if "Avg ML" in full_df.columns:
            col_config["Avg ML"] = st.column_config.NumberColumn("Avg ML", format="%.2f")

        st.dataframe(
            style_df_with_injuries(full_df, players_df),
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config=col_config
        )
        
        # ── Model accuracy info (collapsed) ──
        with st.expander("Model Accuracy & Feature Importance"):
            acc1, acc2 = st.columns(2)
            
            with acc1:
                st.markdown("**Model Performance (5-Fold CV)**")
                if cv_scores:
                    test_mae = cv_scores.get('mean_mae', cv_scores.get('test_mae_mean', 0))
                    test_r2 = cv_scores.get('mean_r2', cv_scores.get('test_r2_mean', 0))
                    train_mae = cv_scores.get('train_mae_mean', test_mae)
                    train_r2 = cv_scores.get('train_r2_mean', test_r2)
                    overfit_ratio = cv_scores.get('overfit_ratio', 1.0)
                    
                    # Interpret for users
                    if test_mae < 0.8:
                        accuracy_text = "Excellent"
                        accuracy_color = "green"
                    elif test_mae < 1.2:
                        accuracy_text = "Good"
                        accuracy_color = "orange"
                    else:
                        accuracy_text = "Fair"
                        accuracy_color = "red"
                    
                    # Overfit check
                    if overfit_ratio > 2.0:
                        overfit_status = ":red[Overfitting detected]"
                    elif overfit_ratio > 1.5:
                        overfit_status = ":orange[Slight overfit]"
                    else:
                        overfit_status = ":green[Good generalization]"
                    
                    st.markdown(f"""
                    **Test Set (unseen data):**
                    - MAE: {test_mae:.2f} pts • R²: {test_r2:.0%}
                    
                    **Train Set:**
                    - MAE: {train_mae:.2f} pts • R²: {train_r2:.0%}
                    
                    **Status:** {overfit_status}
                    """)
                else:
                    st.info("Run predictions to see accuracy metrics")
            
            with acc2:
                st.markdown("**What drives predictions?**")
                if "ml_predictor" in st.session_state:
                    predictor = st.session_state["ml_predictor"]
                    importance_df = predictor.get_feature_importance(top_n=8)
                    
                    if not importance_df.empty:
                        # Updated feature names for pre-match features
                        feature_names = {
                            'fpl_ep': 'FPL Expected Pts',
                            'form_numeric': 'Form Score',
                            'ppg': 'Points Per Game',
                            'form_momentum': 'Form Momentum',
                            'ict_index': 'ICT Index',
                            'influence': 'Influence',
                            'creativity': 'Creativity',
                            'threat': 'Threat',
                            'ict_per_90': 'ICT per 90 mins',
                            'threat_per_90': 'Threat per 90',
                            'creativity_per_90': 'Creativity per 90',
                            'minutes_per_start': 'Minutes/Start',
                            'nailed_score': 'Nailed Score',
                            'chance_of_playing': 'Availability',
                            'price': 'Price',
                            'ownership': 'Ownership %',
                            'value_score': 'Value Score',
                            'price_tier': 'Price Tier',
                            'ownership_tier': 'Ownership Tier',
                            'team_strength': 'Team Quality',
                        }
                        importance_df['feature'] = importance_df['feature'].map(
                            lambda x: feature_names.get(x, x.replace('_', ' ').title())
                        )
                        
                        fig = px.bar(
                            importance_df,
                            x="importance",
                            y="feature",
                            orientation="h",
                            color="importance",
                            color_continuous_scale=["#3b82f6", "#22c55e"],
                        )
                        fig.update_layout(
                            height=300,
                            showlegend=False,
                            coloraxis_showscale=False,
                            template="plotly_white",
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            font=dict(family="Inter, sans-serif", color="#86868b", size=11),
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Importance",
                            yaxis_title="",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Run predictions to see feature importance")
        
        # ── Position breakdown ──
        with st.expander("View by Position"):
            pos_tabs = st.tabs(["GKP", "DEF", "MID", "FWD"])
            
            for i, pos in enumerate(["GKP", "DEF", "MID", "FWD"]):
                with pos_tabs[i]:
                    pos_df = pred_df[pred_df["Pos"] == pos].head(10)
                    if pos_df.empty:
                        st.info(f"No {pos} predictions available")
                    else:
                        pos_display = pos_df[["Player", "Team", "Price", "ML xP", "vs FPL", "vs Poisson", "Certainty", poisson_label]].copy()
                        st.dataframe(
                            pos_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price", format="£%.1fm"),
                                "ML Pred": st.column_config.NumberColumn("ML Pred", format="%.2f"),
                                "vs FPL": st.column_config.NumberColumn("vs FPL", format="%+.2f"),
                                "vs Poisson": st.column_config.NumberColumn("vs Poisson", format="%+.2f"),
                                "Certainty": st.column_config.NumberColumn("Certainty", format="%.0f%%"),
                                poisson_label: st.column_config.NumberColumn(poisson_label, format="%.2f"),
                            }
                        )
    
    else:
        # ── Empty state ──
        st.markdown("---")
        st.info("""
        **Ready to analyze?** Click **Run ML Analysis** above to:
        1. Train machine learning models on this season's data
        2. Get points predictions for every player
        3. See which players are undervalued
        
        *Takes about 10 seconds. No data leaves your browser.*
        """)

