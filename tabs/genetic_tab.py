"""Genetic Optimizer Tab - Evolutionary squad optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric, round_df, style_df_with_injuries, get_consensus_label, calculate_consensus_ep, normalize_name
from utils.visualizations import render_pitch_view


def render_genetic_tab(processor, players_df: pd.DataFrame, fetcher=None):
    """Genetic Algorithm Optimizer tab — its own top-level tab."""
    # Ensure consensus_ep is available
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    players_df = calculate_consensus_ep(players_df, active_models)

    st.markdown('<p class="section-title">Genetic Optimizer</p>', unsafe_allow_html=True)
    
    # Metrics explanation dropdown
    with st.expander("Understanding Genetic Optimizer"):
        st.markdown("""
        **What is Genetic Optimization?**
        - Evolutionary algorithm inspired by natural selection
        - "Evolves" thousands of possible squads to find the optimal one
        - Better than greedy algorithms for complex optimization
        
        **Algorithm Settings**
        - **Population Size**: Number of squad candidates per generation (higher = more exploration)
        - **Generations**: Number of evolution cycles (higher = better convergence)
        - **Mutation Rate**: Chance of random changes (prevents local optima)
        
        **Evolution Process**
        1. **Selection**: Best squads are chosen to "reproduce"
        2. **Crossover**: Combines players from two good squads
        3. **Mutation**: Random player swaps for diversity
        4. **Fitness**: Squads scored on total xP, value, and constraints
        
        **Fitness Score**
        - Combined metric of squad's total xP and budget efficiency
        - Higher fitness = better squad
        
        **When to Use**
        - Wildcard planning: Find optimal 15-man squad
        - Free Hit: Best squad for a single GW
        - Exploring non-obvious squad structures
        """)

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
        
        st.markdown("**Constraints**")
        
        # Prepare player list for dropdowns
        # Prepare player list for dropdowns
        def _build_display_name(row):
            name = row['web_name']
            norm = normalize_name(name)
            base = f"{name} ({row['team_name']}, {row['position']}) £{row['now_cost']}m"
            # Append normalized name if different to allow search by "joao" for "João"
            if norm and norm.lower() != name.lower():
                return f"{base} [{norm}]"
            return base

        players_df['display_name'] = players_df.apply(_build_display_name, axis=1)
        player_options = players_df[['id', 'display_name']].set_index('display_name')['id'].to_dict()
        
        locked_players = st.multiselect(
            "Type to Search & Lock Players (Must Include)",
            options=player_options.keys(),
            default=[],
            key="ga_lock",
            placeholder="Type player name..."
        )
        locked_ids = [player_options[name] for name in locked_players]
        
        excluded_players = st.multiselect(
            "Type to Search & Exclude Players",
            options=player_options.keys(),
            default=[],
            key="ga_exclude",
            placeholder="Type player name..."
        )
        excluded_ids = [player_options[name] for name in excluded_players]

        st.markdown("**Evolution Strategy**")
        st.caption(f"• Population: {pop_size} squads")
        st.caption(f"• Generations: {generations} iterations")
        st.caption("• Selection: Tournament (k=5)")
        st.caption("• Crossover: Multi-point")
        st.caption(f"• Mutation: {mutation_rate * 100:.0f} % rate")

        if st.button("Evolve Squad", type="primary", width="stretch"):
            with st.spinner(f"Evolving for {generations} generations…"):
                try:
                    from genetic_optimizer import create_genetic_optimizer

                    optimizer = create_genetic_optimizer(
                        players_df,
                        population_size=pop_size,
                        n_generations=generations,
                        locked_ids=locked_ids,
                        excluded_ids=excluded_ids
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

            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']))
            squad_data = []
            
            # Helper to get team name
            def _get_team(pid):
                return players_df[players_df['id'] == pid]['team_name'].iloc[0] if not players_df[players_df['id'] == pid].empty else "UNK"

            for pid in best.squad:
                match = players_df[players_df["id"] == pid]
                player = match.iloc[0] if not match.empty else {}
                is_starting = pid in best.starting_xi
                is_captain = pid == best.captain
                is_vice = pid == best.vice_captain # Added to ensure vice works
                
                status_label = "Bench"
                if is_starting: status_label = "XI"
                if is_captain: status_label = "(C)"
                if is_vice: status_label = "(V)"

                squad_data.append(
                    {
                        "Player": player.get("web_name", f"ID:{pid}"),
                        "Pos": player.get("position", "?"),
                        "Team": player.get("team_short_name", _get_team(pid)),
                        "Cost": round(float(player.get("now_cost", 0)), 1),
                        "xP": round(float(safe_numeric(pd.Series([player.get("consensus_ep", 0)])).iloc[0]), 2),
                        "Status": status_label,
                    }
                )

            squad_df = pd.DataFrame(squad_data)
            
            # --- Comparison Section ---
            team_id = st.session_state.get('fpl_team_id', 0)
            if team_id > 0 and fetcher:
                try:
                    current_gw = fetcher.get_current_gameweek()
                    picks = fetcher.get_team_picks(team_id, current_gw)
                    
                    if picks and 'picks' in picks:
                        current_ids = [p['element'] for p in picks['picks']]
                        current_xp = players_df[players_df['id'].isin(current_ids)]['consensus_ep'].sum()
                        current_cost = players_df[players_df['id'].isin(current_ids)]['now_cost'].sum()
                        
                        evolved_xp = squad_df[squad_df['Status'].isin(['XI', '(C)', '(V)'])]['xP'].sum()
                        evolved_cost = squad_df['Cost'].sum()
                        
                        st.markdown('<p class="section-title">Squad Comparison</p>', unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total xP (XI)", f"{evolved_xp:.1f}", delta=f"{evolved_xp - current_xp:.1f}")
                        with m2:
                            st.metric("Squad Value", f"£{evolved_cost:.1f}m", delta=f"{evolved_cost - current_cost:.1f}m")
                        with m3:
                            overlap = len(set(best.squad).intersection(set(current_ids)))
                            st.metric("Overlap", f"{overlap}/15")
                        with m4:
                            st.metric("Fitness", f"{best.fitness:.2f}")
                            
                except Exception as e:
                    # st.warning(f"Could not load current team for comparison: {e}")
                    pass
            
            # --- Table View (Restored) ---
            starting_df = squad_df[squad_df["Status"].isin(["(C)", "XI", "(V)"])].copy()
            bench_df = squad_df[squad_df["Status"] == "Bench"].copy()

            st.markdown("**Starting XI**")
            st.dataframe(style_df_with_injuries(starting_df, players_df), width="stretch", hide_index=True)

            st.markdown("**Bench**")
            st.dataframe(style_df_with_injuries(bench_df, players_df), width="stretch", hide_index=True)

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
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Configure the algorithm on the left and click **Evolve Squad** to see results here.")
