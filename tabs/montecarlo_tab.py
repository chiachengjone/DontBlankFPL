"""Monte Carlo Simulation Tab - Uncertainty quantification for FPL decisions."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import safe_numeric, round_df, style_df_with_injuries, normalize_name, get_consensus_label


def render_monte_carlo_tab(processor, players_df: pd.DataFrame):
    """Monte Carlo Simulation tab — its own top-level tab."""

    st.markdown('<p class="section-title">Monte Carlo Simulation</p>', unsafe_allow_html=True)
    
    # Metrics explanation dropdown
    with st.expander("Understanding Monte Carlo Simulation"):
        st.markdown("""
        **What is Monte Carlo Simulation?**
        - Runs thousands of random scenarios based on statistical distributions
        - Shows the range of possible outcomes, not just averages
        - Essential for understanding risk and variance in FPL
        
        **Simulation Settings**
        - **Simulations**: More = more accurate (10k+ recommended)
        - **Gameweeks**: How far ahead to simulate
        - **Distribution**: Statistical model for point variation
        
        **Distribution Types**
        - **Mixed**: Combines multiple distributions (most realistic)
        - **Gamma**: Right-skewed, good for attacking returns
        - **Normal**: Bell curve, good for consistent performers
        - **Poisson**: Count-based, good for goals/assists
        
        **Output Metrics**
        - **Mean**: Average {get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']))} across all simulations
        - **Std Dev**: Volatility/variance (higher = more unpredictable)
        - **5th/95th Percentile**: Range of likely outcomes (90% confidence)
        - **Upside**: Chance of significantly exceeding expectation
        
        **Portfolio Analysis**
        - Simulates your entire squad together
        - Shows combined floor/ceiling for GW points
        - Identifies if your squad is too risky or too safe
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Simulation Settings**")

        n_sims = st.select_slider(
            "Simulations",
            options=[1000, 5000, 10000, 25000, 50000],
            value=10000,
            key="mc_sims",
        )

        n_gameweeks = st.slider("Gameweeks", 1, 10, 5, key="mc_gw")

        method = st.selectbox(
            "Distribution",
            ["Mixed", "Gamma", "Normal", "Poisson"],
            key="mc_method",
        )

        # Player selection
        st.markdown("**Individual Player Analysis**")
        player_search = st.text_input("Search player", key="mc_player_search", placeholder="First or last name...")

        if player_search:
            q = normalize_name(player_search.lower().strip())
            # Match if query starts with beginning of either name
            if 'first_normalized' not in players_df.columns:
                players_df['first_normalized'] = players_df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
                players_df['second_normalized'] = players_df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
            
            matched = players_df[
                (players_df['first_normalized'].str.startswith(q, na=False)) |
                (players_df['second_normalized'].str.startswith(q, na=False))
            ].sort_values('full_name').head(20)

            if not matched.empty:
                selected_player = st.selectbox(
                    "Select Player",
                    options=matched["id"].tolist(),
                    format_func=lambda x: matched[matched["id"] == x]["full_name"].iloc[0] if not matched[matched["id"] == x].empty else f"ID:{x}",
                    key="mc_player_select",
                )

                if st.button("Simulate Player", type="primary", width="stretch"):
                    # Clear previous results so they don't stack
                    st.session_state.pop("mc_player_result", None)
                    st.session_state.pop("mc_player_id", None)
                    st.session_state.pop("mc_portfolio", None)
                    st.session_state.pop("mc_portfolio_ids", None)
                    st.session_state.pop("mc_captain_id", None)
                    with st.spinner("Running Monte Carlo..."):
                        try:
                            from monte_carlo import create_monte_carlo_engine

                            engine = create_monte_carlo_engine(players_df, n_sims)
                            result = engine.simulate_player(
                                selected_player,
                                n_gameweeks=n_gameweeks,
                                method=method.lower(),
                            )
                            st.session_state["mc_player_result"] = result
                            st.session_state["mc_player_id"] = selected_player
                        except ImportError:
                            st.error("Monte Carlo module not available. Install: pip install scipy")
                        except Exception as e:
                            st.error(f"Error: {e}")

        # Squad simulation — uses global Team ID
        st.markdown("**Squad Simulation**")
        team_id = st.session_state.get("fpl_team_id", 0)
        if team_id and team_id > 0:
            st.caption(f"Team ID: {team_id}")
            if st.button("Simulate My Squad", width="stretch"):
                # Clear previous results so they don't stack
                st.session_state.pop("mc_player_result", None)
                st.session_state.pop("mc_player_id", None)
                st.session_state.pop("mc_portfolio", None)
                st.session_state.pop("mc_portfolio_ids", None)
                st.session_state.pop("mc_captain_id", None)
                with st.spinner("Fetching squad and running simulations..."):
                    try:
                        from monte_carlo import create_monte_carlo_engine

                        gw = processor.fetcher.get_current_gameweek()
                        picks_data = processor.fetcher.get_team_picks(team_id, gw)

                        if not picks_data or not isinstance(picks_data, dict):
                            st.error(f"Failed to fetch team data for ID {team_id}. Please verify your Team ID.")
                        elif "picks" not in picks_data:
                            st.error(f"No squad data found for Team ID {team_id}. Team may be private or ID is invalid.")
                        else:
                            picks = picks_data["picks"]
                            starting_ids = [p["element"] for p in picks if p.get("position", p.get("multiplier", 0)) <= 11 or p.get("multiplier", 0) > 0][:11]
                            captain_id = next((p["element"] for p in picks if p.get("is_captain", False)), starting_ids[0] if starting_ids else 0)
                            bench_ids = [p["element"] for p in picks if p["element"] not in starting_ids]

                            engine = create_monte_carlo_engine(players_df, n_sims)
                            portfolio = engine.simulate_portfolio(
                                squad_ids=starting_ids,
                                captain_id=captain_id,
                                bench_ids=bench_ids,
                                n_gameweeks=n_gameweeks,
                                method=method.lower(),
                            )
                            st.session_state["mc_portfolio"] = portfolio
                            st.session_state["mc_portfolio_ids"] = starting_ids
                            st.session_state["mc_captain_id"] = captain_id
                    except ImportError:
                        st.error("Monte Carlo module not available. Install: pip install scipy")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Enter your FPL Team ID in the header bar above to enable squad simulation.")

    with col2:
        # Show squad simulation results if available
        if "mc_portfolio" in st.session_state:
            portfolio = st.session_state["mc_portfolio"]
            starting_ids = st.session_state.get("mc_portfolio_ids", [])
            captain_id = st.session_state.get("mc_captain_id", 0)

            st.markdown("**Squad Simulation Results**")

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Mean Total", f"{portfolio.mean_total_points:.2f}")
            with s2:
                st.metric("Median Total", f"{portfolio.median_total_points:.2f}")
            with s3:
                st.metric("Best Case (95th)", f"{portfolio.best_case_points:.2f}")
            with s4:
                st.metric("Worst Case (5th)", f"{portfolio.worst_case_points:.2f}")

            s5, s6, s7 = st.columns(3)
            with s5:
                st.metric("Std Dev", f"{portfolio.std_total_points:.2f}")
            with s6:
                st.metric("Top 10k Prob", f"{portfolio.probability_top_10k * 100:.1f}%")
            with s7:
                st.metric("Top 100k Prob", f"{portfolio.probability_top_100k * 100:.1f}%")

            # Player contributions table
            if portfolio.player_contributions:
                st.markdown("**Player Contributions**")
                contrib_data = []
                for pid, res in portfolio.player_contributions.items():
                    match = players_df[players_df["id"] == pid]
                    name = match.iloc[0]["web_name"] if not match.empty else f"ID:{pid}"
                    is_cap = "(C)" if pid == captain_id else ""
                    contrib_data.append({
                        "Player": f"{name} {is_cap}".strip(),
                        "Mean Pts": round(res.mean_points * (1.25 if pid == captain_id else 1.0), 2),
                        "Std Dev": round(res.std_points, 2),
                        "Sharpe": round(res.sharpe_ratio, 2),
                    })
                contrib_df = pd.DataFrame(contrib_data).sort_values("Mean Pts", ascending=False)
                st.dataframe(style_df_with_injuries(contrib_df, players_df), width="stretch", hide_index=True)

            st.markdown("---")

        # Show individual player results
        if "mc_player_result" in st.session_state:
            result = st.session_state["mc_player_result"]
            player_id = st.session_state["mc_player_id"]
            player_name = players_df[players_df["id"] == player_id]["web_name"].iloc[0]

            st.markdown(f"**Simulation Results: {player_name}**")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Mean Points", f"{result.mean_points:.2f}")
            with m2:
                st.metric("Median Points", f"{result.median_points:.2f}")
            with m3:
                st.metric("Std Dev", f"{result.std_points:.2f}")
            with m4:
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

            # Percentile chart
            st.markdown("**Confidence Intervals**")
            ci_data = pd.DataFrame(
                {
                    "Percentile": ["5th", "25th", "50th", "75th", "95th"],
                    "Points": [
                        round(result.percentile_5, 2),
                        round(result.percentile_25, 2),
                        round(result.median_points, 2),
                        round(result.percentile_75, 2),
                        round(result.percentile_95, 2),
                    ],
                }
            )

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=ci_data["Percentile"],
                    y=ci_data["Points"],
                    marker_color=["#ef4444", "#f59e0b", "#22c55e", "#f59e0b", "#ef4444"],
                )
            )
            fig.update_layout(
                title=f"Points Distribution ({n_gameweeks} GW)",
                xaxis_title="Percentile",
                yaxis_title="Points",
                height=350,
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#86868b", size=11),
            )
            st.plotly_chart(fig, width="stretch")

            # Risk metrics
            st.markdown("**Risk Analysis**")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Value at Risk (95%)", f"{result.value_at_risk_95:.2f}")
                st.caption("Worst case in 95 % of outcomes")
            with r2:
                st.metric("Expected Shortfall", f"{result.expected_shortfall_95:.2f}")
                st.caption("Average of worst 5 % (CVaR)")
            with r3:
                st.metric("Prob > Expected", f"{result.probability_exceeds_threshold * 100:.1f}%")
                st.caption(f"Chance of exceeding {get_consensus_label(st.session_state.active_models)}")
        if "mc_player_result" not in st.session_state and "mc_portfolio" not in st.session_state:
            st.info("Search for a player or simulate your squad to see results here.")
