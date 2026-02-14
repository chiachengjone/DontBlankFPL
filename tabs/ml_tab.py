"""ML Predictions Tab - Machine Learning player performance forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import safe_numeric, round_df, style_df_with_injuries


def render_ml_tab(processor, players_df: pd.DataFrame):
    """Machine Learning Predictions tab — intuitive UI for all users."""

    st.markdown('<p class="section-title">ML Predictions</p>', unsafe_allow_html=True)
    
    # ── Controls ──
    st.markdown("---")
    
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1.5, 1.5])
    
    with ctrl1:
        n_gameweeks = st.selectbox(
            "Predict for",
            options=[1, 2, 3, 5],
            format_func=lambda x: f"Next {x} GW{'s' if x > 1 else ''}",
            index=0,
            key="ml_horizon",
            help="Gameweeks ahead to predict. 1 GW is most accurate."
        )
    
    with ctrl2:
        position_filter = st.selectbox(
            "Position",
            options=["All", "GKP", "DEF", "MID", "FWD"],
            index=0,
            key="ml_pos_filter",
            help="Filter results by position"
        )
    
    with ctrl3:
        # Player-specific search
        player_names = ["-- All Players --"] + sorted(players_df["web_name"].dropna().unique().tolist())
        selected_player = st.selectbox(
            "Focus on player",
            options=player_names,
            index=0,
            key="ml_player_focus",
            help="Get detailed ML breakdown for a specific player"
        )
    
    with ctrl4:
        run_ml = st.button(
            "Run ML Analysis",
            type="primary",
            use_container_width=True,
            help="Takes ~10 seconds. Trains models on your data."
        )
    
    # ── Run ML ──
    if run_ml:
        with st.spinner("Training ML models (this takes ~10 seconds)..."):
            try:
                from ml_predictor import create_ml_pipeline

                predictor = create_ml_pipeline(players_df)
                predictions = predictor.predict_gameweek_points(
                    n_gameweeks=n_gameweeks,
                    use_ensemble=True,
                )

                st.session_state["ml_predictions"] = predictions
                st.session_state["ml_predictor"] = predictor
                st.session_state["ml_gws"] = n_gameweeks
                
                # Cross-validation for accuracy metrics
                cv_scores = predictor.cross_validate_predictions(n_splits=3)
                st.session_state["ml_cv_scores"] = cv_scores
                
                st.success(f"Analyzed {len(predictions)} players.")

            except ImportError:
                st.error("ML libraries not installed. Run: `pip install xgboost scikit-learn`")
            except Exception as e:
                st.error(f"Error: {e}")
    
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
            
            # Get raw FPL EP (ep_next from API), NOT the Poisson-modified expected_points
            # ep_next_num is the raw value preserved before Poisson blending
            raw_fpl_ep = player.get('ep_next_num', player.get('ep_next', 0))
            fpl_ep = safe_numeric(pd.Series([raw_fpl_ep])).iloc[0]
            
            # Get Poisson EP (expected_points which is the blended value when Poisson is on)
            poisson_ep_raw = player.get('expected_points', fpl_ep)
            poisson_ep = safe_numeric(pd.Series([poisson_ep_raw])).iloc[0]
            
            ml_pred = pred.predicted_points
            
            # Better uncertainty: use CI width relative to prediction magnitude
            ci_low, ci_high = pred.confidence_interval
            ci_width = ci_high - ci_low
            
            # Relative uncertainty: how wide is the CI compared to the prediction?
            # Lower is better. A CI width of 0.5 on a prediction of 5 = 10% uncertainty
            if ml_pred > 0.5:
                relative_uncertainty = (ci_width / ml_pred) * 100
            else:
                relative_uncertainty = ci_width * 50  # Scale for low predictions
            
            # Cap and invert to get "certainty" score (0-100)
            # 0% uncertainty = 100 certainty, 100%+ uncertainty = 0 certainty
            certainty = max(0, min(100, 100 - relative_uncertainty))
            
            pred_data.append({
                "id": pid,
                "Player": player.get("web_name", f"ID:{pid}"),
                "Pos": player.get("position", "?"),
                "Team": player.get("team_name", "?"),
                "Price": round(float(player.get("now_cost", 0)), 1),
                "ML Pred": round(ml_pred, 1),
                "Range": f"{ci_low:.1f}-{ci_high:.1f}",
                "FPL EP": round(fpl_ep, 1),
                "Poisson EP": round(poisson_ep, 1),
                "vs FPL": round(ml_pred - fpl_ep, 1),
                "vs Poisson": round(ml_pred - poisson_ep, 1),
                "Certainty": round(certainty, 0),
                "_ci_low": ci_low,
                "_ci_high": ci_high,
                "_uncertainty": pred.prediction_std,
            })

        pred_df = pd.DataFrame(pred_data)
        
        # ── Player-specific view ──
        if selected_player != "-- All Players --":
            player_row = pred_df[pred_df["Player"] == selected_player]
            if not player_row.empty:
                st.markdown(f"### ML Analysis: {selected_player}")
                
                p = player_row.iloc[0]
                pid = p["id"]
                pred_obj = predictions.get(pid)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ML Prediction", f"{p['ML Pred']:.1f} pts")
                with col2:
                    delta_color = "normal" if p["vs FPL"] >= 0 else "inverse"
                    st.metric("vs FPL", f"{p['FPL EP']:.1f} pts", 
                              f"{p['vs FPL']:+.1f}", delta_color=delta_color)
                with col3:
                    delta_color = "normal" if p["vs Poisson"] >= 0 else "inverse"
                    st.metric("vs Poisson", f"{p['Poisson EP']:.1f} pts", 
                              f"{p['vs Poisson']:+.1f}", delta_color=delta_color)
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
                    
                    for feat, imp in sorted_imp:
                        display_name = feature_names.get(feat, feat.replace('_', ' ').title())
                        bar_width = int(imp * 100)
                        st.markdown(f"- {display_name}: {'█' * min(bar_width, 20)} ({imp:.1%})")
                
                st.markdown("---")
        
        # Apply position filter
        if position_filter != "All":
            pred_df = pred_df[pred_df["Pos"] == position_filter]
        
        pred_df = pred_df.sort_values("ML Pred", ascending=False).reset_index(drop=True)
        
        # ── Quick Insights ──
        st.markdown("### Quick Picks")
        
        qp1, qp2, qp3, qp4 = st.columns(4)
        
        top_pick = pred_df.head(1)
        if not top_pick.empty:
            with qp1:
                st.metric(
                    "Top Pick",
                    top_pick.iloc[0]["Player"],
                    f"{top_pick.iloc[0]['ML Pred']:.1f} pts",
                    help="Highest ML predicted points"
                )
        
        # Best value (ML pred / price)
        pred_df["_value"] = pred_df["ML Pred"] / pred_df["Price"].clip(lower=4)
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
                    f"+{biggest_upside.iloc[0]['vs FPL']:.1f} vs FPL",
                    help="ML thinks this player is underrated"
                )
        
        # Most certain high prediction
        high_certainty = pred_df[pred_df["Certainty"] >= 50].nlargest(1, "ML Pred")
        if not high_certainty.empty:
            with qp4:
                st.metric(
                    "Safest Pick",
                    high_certainty.iloc[0]["Player"],
                    f"{high_certainty.iloc[0]['Certainty']:.0f}% certain",
                    help="High prediction with narrow range"
                )
        
        # ── Full results table ──
        st.markdown(f"### All Players (Next {gws} GW{'s' if gws > 1 else ''}) — {len(pred_df)} players")
        
        # Prepare display dataframe - keep numeric values for proper sorting/filtering
        display_cols = ["Player", "Pos", "Team", "Price", "ML Pred", "Range", "FPL EP", "vs FPL", "Poisson EP", "vs Poisson", "Certainty"]
        full_df = pred_df[display_cols].copy()
        
        # Use column_config for formatting without converting to strings
        st.dataframe(
            style_df_with_injuries(full_df, players_df),
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="£%.1fm"),
                "ML Pred": st.column_config.NumberColumn("ML Pred", format="%.1f"),
                "FPL EP": st.column_config.NumberColumn("FPL EP", format="%.1f"),
                "vs FPL": st.column_config.NumberColumn("vs FPL", format="%+.1f"),
                "Poisson EP": st.column_config.NumberColumn("Poisson EP", format="%.1f"),
                "vs Poisson": st.column_config.NumberColumn("vs Poisson", format="%+.1f"),
                "Certainty": st.column_config.NumberColumn("Certainty", format="%.0f%%"),
            }
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
                            template="plotly_dark",
                            paper_bgcolor="#0a0a0b",
                            plot_bgcolor="#111113",
                            font=dict(family="Inter, sans-serif", color="#6b6b6b", size=11),
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
                        pos_display = pos_df[["Player", "Team", "Price", "ML Pred", "vs FPL", "vs Poisson", "Certainty"]].copy()
                        st.dataframe(
                            pos_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price", format="£%.1fm"),
                                "ML Pred": st.column_config.NumberColumn("ML Pred", format="%.1f"),
                                "vs FPL": st.column_config.NumberColumn("vs FPL", format="%+.1f"),
                                "vs Poisson": st.column_config.NumberColumn("vs Poisson", format="%+.1f"),
                                "Certainty": st.column_config.NumberColumn("Certainty", format="%.0f%%"),
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

