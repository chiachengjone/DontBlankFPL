"""Advanced Analysis Tab - ML, Monte Carlo, Backtesting, and Genetic Algorithms."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric, normalize_name


def render_advanced_tab(processor, players_df: pd.DataFrame):
    """Advanced analysis tab with cutting-edge features."""
    
    st.markdown('<p class="section-title">Advanced Analytics Laboratory</p>', unsafe_allow_html=True)
    st.caption("ML predictions, Monte Carlo simulations, genetic algorithms, and backtesting")
    
    # Sub-tabs for different advanced features
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "ML Predictions",
        "Monte Carlo Sim",
        "Genetic Optimizer",
        "Backtesting"
    ])
    
    with subtab1:
        render_ml_predictions(processor, players_df)
    
    with subtab2:
        render_monte_carlo(processor, players_df)
    
    with subtab3:
        render_genetic_optimizer(processor, players_df)
    
    with subtab4:
        render_backtesting(processor, players_df)


def render_ml_predictions(processor, players_df: pd.DataFrame):
    """Machine Learning predictions section."""
    st.markdown("### Machine Learning Predictions")
    st.markdown("Ensemble model using XGBoost, Random Forest, and Gradient Boosting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Model Configuration**")
        n_gameweeks = st.slider("Prediction Horizon", 1, 5, 1, key="ml_horizon")
        use_ensemble = st.checkbox("Use Ensemble", value=True, key="ml_ensemble")
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        
        if st.button("Generate ML Predictions", type="primary", use_container_width=True):
            with st.spinner("Training ML models..."):
                try:
                    from ml_predictor import create_ml_pipeline
                    
                    # Create and run ML pipeline
                    predictor = create_ml_pipeline(players_df)
                    predictions = predictor.predict_gameweek_points(
                        n_gameweeks=n_gameweeks,
                        use_ensemble=use_ensemble
                    )
                    
                    # Store in session state
                    st.session_state['ml_predictions'] = predictions
                    st.session_state['ml_predictor'] = predictor
                    
                    st.success(f"Predicted {len(predictions)} players!")
                    
                    # Cross-validation scores
                    cv_scores = predictor.cross_validate_predictions(n_splits=3)
                    
                    st.markdown("**Model Performance**")
                    st.metric("Mean Absolute Error", f"{cv_scores['mean_mae']:.2f}")
                    st.metric("R² Score", f"{cv_scores['mean_r2']:.3f}")
                    st.metric("RMSE", f"{cv_scores['mean_rmse']:.2f}")
                    
                except ImportError:
                    st.error("ML module not available. Install: `pip install xgboost scikit-learn`")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if 'ml_predictions' in st.session_state:
            predictions = st.session_state['ml_predictions']
            
            # Build DataFrame
            pred_data = []
            for pid, pred in predictions.items():
                player = players_df[players_df['id'] == pid].iloc[0] if pid in players_df['id'].values else {}
                
                pred_data.append({
                    'Player': player.get('web_name', f'ID:{pid}'),
                    'Position': player.get('position', '?'),
                    'Team': player.get('team_name', '?'),
                    'Cost': player.get('now_cost', 0),
                    'ML Predicted': round(pred.predicted_points, 2),
                    'CI Lower': round(pred.confidence_interval[0], 2),
                    'CI Upper': round(pred.confidence_interval[1], 2),
                    'Uncertainty': round(pred.prediction_std, 2),
                })
            
            pred_df = pd.DataFrame(pred_data).sort_values('ML Predicted', ascending=False)
            
            st.markdown("**Top ML Predictions**")
            st.dataframe(
                pred_df.head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Uncertainty visualization
            if show_feature_importance and 'ml_predictor' in st.session_state:
                st.markdown("**Feature Importance**")
                predictor = st.session_state['ml_predictor']
                importance_df = predictor.get_feature_importance(top_n=10)
                
                if not importance_df.empty:
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Features Driving Predictions'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)


def render_monte_carlo(processor, players_df: pd.DataFrame):
    """Monte Carlo simulation section."""
    st.markdown("### Monte Carlo Simulation")
    st.markdown("Probabilistic modeling with 10,000+ simulations for uncertainty quantification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Simulation Settings**")
        
        n_sims = st.select_slider(
            "Simulations",
            options=[1000, 5000, 10000, 25000, 50000],
            value=10000,
            key="mc_sims"
        )
        
        n_gameweeks = st.slider("Gameweeks", 1, 10, 5, key="mc_gw")
        
        method = st.selectbox(
            "Distribution",
            ["Mixed", "Gamma", "Normal", "Poisson"],
            key="mc_method"
        )
        
        # Player selection for individual simulation
        st.markdown("**Individual Player Analysis**")
        player_search = st.text_input("Search player", key="mc_player_search")
        
        if player_search:
            search_norm = normalize_name(player_search.lower().strip())
            players_df['_name_norm'] = players_df['web_name'].apply(lambda x: normalize_name(str(x).lower()))
            matched = players_df[
                players_df['_name_norm'].str.contains(search_norm, na=False)
            ]
            
            if not matched.empty:
                selected_player = st.selectbox(
                    "Select Player",
                    options=matched['id'].tolist(),
                    format_func=lambda x: matched[matched['id']==x]['web_name'].iloc[0],
                    key="mc_player_select"
                )
                
                if st.button("Simulate Player", type="primary", use_container_width=True):
                    with st.spinner("Running Monte Carlo..."):
                        try:
                            from monte_carlo import create_monte_carlo_engine
                            
                            engine = create_monte_carlo_engine(players_df, n_sims)
                            result = engine.simulate_player(
                                selected_player,
                                n_gameweeks=n_gameweeks,
                                method=method.lower()
                            )
                            
                            st.session_state['mc_player_result'] = result
                            st.session_state['mc_player_id'] = selected_player
                            
                        except ImportError:
                            st.error("Monte Carlo module not available")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # Squad simulation
        st.markdown("**Squad Simulation**")
        if st.button("Simulate Current Squad", use_container_width=True):
            st.info("Feature coming soon: connect your team ID for squad simulation")
    
    with col2:
        if 'mc_player_result' in st.session_state:
            result = st.session_state['mc_player_result']
            player_id = st.session_state['mc_player_id']
            player_name = players_df[players_df['id']==player_id]['web_name'].iloc[0]
            
            st.markdown(f"**Simulation Results: {player_name}**")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Mean Points", f"{result.mean_points:.2f}")
            with m2:
                st.metric("Median Points", f"{result.median_points:.2f}")
            with m3:
                st.metric("Std Dev", f"{result.std_points:.2f}")
            with m4:
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            
            # Percentiles
            st.markdown("**Confidence Intervals**")
            ci_data = pd.DataFrame({
                'Percentile': ['5th', '25th', '50th', '75th', '95th'],
                'Points': [
                    result.percentile_5,
                    result.percentile_25,
                    result.median_points,
                    result.percentile_75,
                    result.percentile_95
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ci_data['Percentile'],
                y=ci_data['Points'],
                marker_color=['#ef4444', '#f59e0b', '#22c55e', '#f59e0b', '#ef4444']
            ))
            fig.update_layout(
                title=f"Points Distribution ({n_gameweeks} GW)",
                xaxis_title="Percentile",
                yaxis_title="Points",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.markdown("**Risk Analysis**")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Value at Risk (95%)", f"{result.value_at_risk_95:.2f}")
                st.caption("Worst case in 95% of outcomes")
            with r2:
                st.metric("Expected Shortfall", f"{result.expected_shortfall_95:.2f}")
                st.caption("Average of worst 5%")
            with r3:
                st.metric("Prob > Expected", f"{result.probability_exceeds_threshold*100:.1f}%")
                st.caption("Chance of exceeding EP")


def render_genetic_optimizer(processor, players_df: pd.DataFrame):
    """Genetic algorithm optimization section."""
    st.markdown("### Genetic Algorithm Optimizer")
    st.markdown("Evolutionary squad optimization using nature-inspired algorithms")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Algorithm Configuration**")
        
        pop_size = st.select_slider(
            "Population Size",
            options=[50, 100, 200, 500],
            value=100,
            key="ga_pop"
        )
        
        generations = st.slider(
            "Generations",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="ga_gen"
        )
        
        mutation_rate = st.slider(
            "Mutation Rate",
            min_value=0.05,
            max_value=0.30,
            value=0.15,
            step=0.05,
            key="ga_mut"
        )
        
        st.markdown("**Evolution Strategy**")
        st.caption(f"• Population: {pop_size} squads")
        st.caption(f"• Generations: {generations} iterations")
        st.caption(f"• Selection: Tournament (k=5)")
        st.caption(f"• Crossover: Multi-point")
        st.caption(f"• Mutation: {mutation_rate*100:.0f}% rate")
        
        if st.button("Evolve Squad", type="primary", use_container_width=True):
            with st.spinner(f"Evolving for {generations} generations..."):
                try:
                    from genetic_optimizer import create_genetic_optimizer
                    
                    optimizer = create_genetic_optimizer(
                        players_df,
                        population_size=pop_size,
                        n_generations=generations
                    )
                    
                    best = optimizer.evolve()
                    
                    st.session_state['ga_result'] = best
                    st.session_state['ga_history'] = optimizer.get_optimization_history()
                    st.session_state['ga_optimizer'] = optimizer
                    
                    st.success(f"Evolution complete! Fitness: {best.fitness:.2f}")
                    
                except ImportError:
                    st.error("Genetic optimizer not available")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if 'ga_result' in st.session_state:
            best = st.session_state['ga_result']
            history = st.session_state['ga_history']
            
            st.markdown("**Optimized Squad**")
            
            # Squad display
            squad_data = []
            for i, pid in enumerate(best.squad):
                player = players_df[players_df['id'] == pid].iloc[0] if pid in players_df['id'].values else {}
                is_starting = pid in best.starting_xi
                is_captain = pid == best.captain
                
                squad_data.append({
                    'Player': player.get('web_name', f'ID:{pid}'),
                    'Position': player.get('position', '?'),
                    'Cost': player.get('now_cost', 0),
                    'EP': player.get('ep_next', 0),
                    'Status': '(C)' if is_captain else 'Starting' if is_starting else 'Bench'
                })
            
            squad_df = pd.DataFrame(squad_data)
            
            # Split into starting XI and bench
            starting_df = squad_df[squad_df['Status'].isin(['(C)', 'Starting'])].copy()
            bench_df = squad_df[squad_df['Status'] == 'Bench'].copy()
            
            st.markdown("**Starting XI**")
            st.dataframe(starting_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Bench**")
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
            
            # Evolution convergence chart
            st.markdown("**Evolution Convergence**")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Fitness Evolution", "Population Validity")
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history['generation'],
                    y=history['best_fitness'],
                    name='Best',
                    line=dict(color='#22c55e', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history['generation'],
                    y=history['avg_fitness'],
                    name='Average',
                    line=dict(color='#3b82f6', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history['generation'],
                    y=history['valid_count'],
                    name='Valid Squads',
                    fill='tozeroy',
                    line=dict(color='#8b5cf6')
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Generation", row=1, col=1)
            fig.update_xaxes(title_text="Generation", row=1, col=2)
            fig.update_yaxes(title_text="Fitness", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)


def render_backtesting(processor, players_df: pd.DataFrame):
    """Backtesting section."""
    st.markdown("### Strategy Backtesting")
    st.markdown("Historical validation of optimization strategies")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Backtest Configuration**")
        
        season = st.selectbox(
            "Season",
            ["2024-25", "2023-24", "2022-23"],
            key="bt_season"
        )
        
        strategy = st.selectbox(
            "Strategy",
            ["Greedy (Max Points)", "Balanced", "Value Focus", "Differential"],
            key="bt_strategy"
        )
        
        start_gw = st.number_input("Start GW", 1, 38, 1, key="bt_start")
        end_gw = st.number_input("End GW", 1, 38, 38, key="bt_end")
        
        st.markdown("**Backtest Features**")
        st.caption("- Historical FPL data")
        st.caption("- Transfer cost modeling")
        st.caption("- Chip usage simulation")
        st.caption("- Rank percentile estimation")
        
        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Backtesting strategy..."):
                try:
                    from backtesting import HistoricalDataLoader, create_backtest_engine
                    from backtesting import greedy_strategy, balanced_strategy
                    
                    # Load historical data (synthetic for demo)
                    loader = HistoricalDataLoader(season)
                    historical_data = loader.load_season_data()
                    
                    # Create engine
                    engine = create_backtest_engine(historical_data)
                    
                    # Select strategy
                    if "Greedy" in strategy:
                        strategy_func = greedy_strategy
                    else:
                        strategy_func = balanced_strategy
                    
                    # Run backtest
                    result = engine.run_backtest(
                        strategy_func,
                        strategy_name=strategy,
                        start_gw=start_gw,
                        end_gw=end_gw
                    )
                    
                    st.session_state['bt_result'] = result
                    
                    st.success("Backtest complete!")
                    
                except ImportError:
                    st.error("Backtesting module not available")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if 'bt_result' in st.session_state:
            result = st.session_state['bt_result']
            m = result.metrics
            
            st.markdown(f"**Results: {result.strategy_name}**")
            
            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Points", f"{m.total_points:,}")
            with c2:
                st.metric("Final Rank", f"{m.final_rank:,}")
            with c3:
                st.metric("Percentile", f"{m.percentile_finish:.1f}%")
            with c4:
                st.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
            
            # Performance details
            st.markdown("**Performance Breakdown**")
            perf_data = pd.DataFrame({
                'Metric': [
                    'Average GW Points',
                    'Best GW',
                    'Worst GW',
                    'Std Dev',
                    'Total Transfers',
                    'Transfer Cost',
                    'Net Points',
                    'Bench Waste'
                ],
                'Value': [
                    f"{m.average_gw_points:.1f}",
                    f"{m.best_gw_points}",
                    f"{m.worst_gw_points}",
                    f"{m.std_gw_points:.1f}",
                    f"{m.total_transfers}",
                    f"{m.transfer_cost}",
                    f"{m.net_points:,}",
                    f"{m.bench_points_wasted}"
                ]
            })
            st.dataframe(perf_data, use_container_width=True, hide_index=True)
            
            # Gameweek performance chart
            st.markdown("**Gameweek Performance**")
            gw_df = result.gameweek_history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gw_df['gameweek'],
                y=gw_df['points'],
                mode='lines+markers',
                name='Points',
                line=dict(color='#22c55e', width=2)
            ))
            
            # Add average line
            fig.add_hline(
                y=m.average_gw_points,
                line_dash="dash",
                line_color="#888",
                annotation_text=f"Avg: {m.average_gw_points:.1f}"
            )
            
            fig.update_layout(
                xaxis_title="Gameweek",
                yaxis_title="Points",
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            with st.expander("Detailed Summary"):
                st.code(result.summary(), language="text")
