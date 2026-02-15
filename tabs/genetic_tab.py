"""Genetic Optimizer Tab - Evolutionary squad optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric, round_df, style_df_with_injuries


def render_genetic_tab(processor, players_df: pd.DataFrame):
    """Genetic Algorithm Optimizer tab — its own top-level tab."""

    st.markdown('<p class="section-title">Genetic Optimizer</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Algorithm Configuration**")

        pop_size = st.select_slider(
            "Population Size",
            options=[50, 100, 200, 500],
            value=100,
            key="ga_pop",
        )

        generations = st.slider(
            "Generations", min_value=10, max_value=100, value=50, step=10, key="ga_gen"
        )

        mutation_rate = st.slider(
            "Mutation Rate", min_value=0.05, max_value=0.30, value=0.15, step=0.05, key="ga_mut"
        )

        st.markdown("**Evolution Strategy**")
        st.caption(f"• Population: {pop_size} squads")
        st.caption(f"• Generations: {generations} iterations")
        st.caption("• Selection: Tournament (k=5)")
        st.caption("• Crossover: Multi-point")
        st.caption(f"• Mutation: {mutation_rate * 100:.0f} % rate")

        if st.button("Evolve Squad", type="primary", use_container_width=True):
            with st.spinner(f"Evolving for {generations} generations…"):
                try:
                    from genetic_optimizer import create_genetic_optimizer

                    optimizer = create_genetic_optimizer(
                        players_df,
                        population_size=pop_size,
                        n_generations=generations,
                    )
                    best = optimizer.evolve()

                    st.session_state["ga_result"] = best
                    st.session_state["ga_history"] = optimizer.get_optimization_history()
                    st.session_state["ga_optimizer"] = optimizer
                    st.success(f"Evolution complete! Fitness: {best.fitness:.2f}")
                except ImportError:
                    st.error("Genetic optimizer not available")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "ga_result" in st.session_state:
            best = st.session_state["ga_result"]
            history = st.session_state["ga_history"]

            st.markdown("**Optimized Squad**")

            squad_data = []
            for pid in best.squad:
                match = players_df[players_df["id"] == pid]
                player = match.iloc[0] if not match.empty else {}
                is_starting = pid in best.starting_xi
                is_captain = pid == best.captain

                squad_data.append(
                    {
                        "Player": player.get("web_name", f"ID:{pid}"),
                        "Position": player.get("position", "?"),
                        "Cost": round(float(player.get("now_cost", 0)), 1),
                        "EP": round(float(safe_numeric(pd.Series([player.get("expected_points", player.get("ep_next", 0))])).iloc[0]), 2),
                        "Status": "(C)" if is_captain else "XI" if is_starting else "Bench",
                    }
                )

            squad_df = pd.DataFrame(squad_data)

            starting_df = squad_df[squad_df["Status"].isin(["(C)", "XI"])].copy()
            bench_df = squad_df[squad_df["Status"] == "Bench"].copy()

            st.markdown("**Starting XI**")
            st.dataframe(style_df_with_injuries(starting_df, players_df), use_container_width=True, hide_index=True)

            st.markdown("**Bench**")
            st.dataframe(style_df_with_injuries(bench_df, players_df), use_container_width=True, hide_index=True)

            # Convergence chart
            st.markdown("**Evolution Convergence**")

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Fitness Evolution", "Population Validity"),
            )

            fig.add_trace(
                go.Scatter(
                    x=history["generation"],
                    y=history["best_fitness"],
                    name="Best",
                    line=dict(color="#22c55e", width=2),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=history["generation"],
                    y=history["avg_fitness"],
                    name="Average",
                    line=dict(color="#3b82f6", width=2, dash="dash"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=history["generation"],
                    y=history["valid_count"],
                    name="Valid Squads",
                    fill="tozeroy",
                    line=dict(color="#8b5cf6"),
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text="Generation", row=1, col=1)
            fig.update_xaxes(title_text="Generation", row=1, col=2)
            fig.update_yaxes(title_text="Fitness", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            fig.update_layout(
                height=400,
                showlegend=True,
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#86868b", size=11),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure the algorithm on the left and click **Evolve Squad** to see results here.")
