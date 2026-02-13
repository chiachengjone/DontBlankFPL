"""ML Predictions Tab - Machine Learning player performance forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import safe_numeric, round_df


def render_ml_tab(processor, players_df: pd.DataFrame):
    """Machine Learning Predictions tab — its own top-level tab."""

    st.markdown('<p class="section-title">ML Predictions</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Model Configuration**")
        n_gameweeks = st.slider("Prediction Horizon (GWs)", 1, 5, 1, key="ml_horizon")
        use_ensemble = st.checkbox("Use Ensemble", value=True, key="ml_ensemble")
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)

        if st.button("Generate ML Predictions", type="primary", use_container_width=True):
            with st.spinner("Training ML models…"):
                try:
                    from ml_predictor import create_ml_pipeline

                    predictor = create_ml_pipeline(players_df)
                    predictions = predictor.predict_gameweek_points(
                        n_gameweeks=n_gameweeks,
                        use_ensemble=use_ensemble,
                    )

                    st.session_state["ml_predictions"] = predictions
                    st.session_state["ml_predictor"] = predictor
                    st.success(f"Predicted {len(predictions)} players!")

                    cv_scores = predictor.cross_validate_predictions(n_splits=3)

                    st.markdown("**Model Performance**")
                    st.metric("Mean Absolute Error", f"{cv_scores['mean_mae']:.2f}")
                    st.metric("R2 Score", f"{cv_scores['mean_r2']:.2f}")
                    st.metric("RMSE", f"{cv_scores['mean_rmse']:.2f}")

                except ImportError:
                    st.error("ML module not available. Install: pip install xgboost scikit-learn")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "ml_predictions" in st.session_state:
            predictions = st.session_state["ml_predictions"]

            pred_data = []
            for pid, pred in predictions.items():
                match = players_df[players_df["id"] == pid]
                player = match.iloc[0] if not match.empty else {}
                pred_data.append(
                    {
                        "Player": player.get("web_name", f"ID:{pid}"),
                        "Position": player.get("position", "?"),
                        "Team": player.get("team_name", "?"),
                        "Cost": round(float(player.get("now_cost", 0)), 1),
                        "ML Predicted": round(pred.predicted_points, 2),
                        "CI Lower": round(pred.confidence_interval[0], 2),
                        "CI Upper": round(pred.confidence_interval[1], 2),
                        "Uncertainty": round(pred.prediction_std, 2),
                    }
                )

            pred_df = pd.DataFrame(pred_data).sort_values("ML Predicted", ascending=False)

            st.markdown("**Top ML Predictions**")
            st.dataframe(round_df(pred_df.head(20), 2), use_container_width=True, hide_index=True)

            # Feature importance chart
            if show_feature_importance and "ml_predictor" in st.session_state:
                st.markdown("**Feature Importance**")
                predictor = st.session_state["ml_predictor"]
                importance_df = predictor.get_feature_importance(top_n=10)

                if not importance_df.empty:
                    importance_df["importance"] = importance_df["importance"].round(2)
                    fig = px.bar(
                        importance_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 10 Features Driving Predictions",
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        template="plotly_dark",
                        paper_bgcolor="#0a0a0b",
                        plot_bgcolor="#111113",
                        font=dict(family="Inter, sans-serif", color="#6b6b6b", size=11),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click **Generate ML Predictions** to train the model and see results here.")
