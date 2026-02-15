"""
ML Diagnostics - Check for target leakage, feature correlation, and overfitting.

Run this script to identify issues with your ML pipeline:
    python ml_diagnostics.py

This will:
1. Load player data from FPL API
2. Compute feature correlations with target
3. Flag potential target leakage (correlation > 0.9)
4. Run K-Fold CV to check for overfitting
5. Output a report
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load player data from FPL API."""
    from fpl_api import FPLDataFetcher, FPLDataProcessor
    fetcher = FPLDataFetcher()
    processor = FPLDataProcessor(fetcher)
    return processor.players_df


def compute_feature_correlations(df: pd.DataFrame, target_col: str = 'ep_next') -> pd.DataFrame:
    """
    Compute correlation between all numeric features and the target.
    High correlation (>0.9) may indicate target leakage.
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        # Try to create target from available data
        if 'ep_next' in df.columns:
            target_col = 'ep_next'
        elif 'expected_points' in df.columns:
            target_col = 'expected_points'
        else:
            print(f"Warning: Target column '{target_col}' not found. Using 'total_points'.")
            target_col = 'total_points'
    
    correlations = []
    target = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    
    for col in numeric_cols:
        if col == target_col:
            continue
        
        feature = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Skip constant columns
        if feature.std() == 0:
            continue
        
        corr = np.corrcoef(feature, target)[0, 1]
        
        # Classify risk level
        abs_corr = abs(corr)
        if abs_corr >= 0.95:
            risk = "CRITICAL - Almost certainly leakage"
        elif abs_corr >= 0.85:
            risk = "HIGH - Likely leakage"
        elif abs_corr >= 0.70:
            risk = "MEDIUM - Possible leakage"
        elif abs_corr >= 0.50:
            risk = "LOW - Monitor"
        else:
            risk = "OK"
        
        correlations.append({
            'feature': col,
            'correlation': round(corr, 4),
            'abs_correlation': round(abs_corr, 4),
            'risk': risk
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def identify_leakage_features(corr_df: pd.DataFrame, threshold: float = 0.85) -> List[str]:
    """Get list of features with high correlation (potential leakage)."""
    leaky = corr_df[corr_df['abs_correlation'] >= threshold]['feature'].tolist()
    return leaky


def classify_features() -> Dict[str, List[str]]:
    """
    Classify FPL features into pre-match and post-match categories.
    
    Pre-match: Known before the match starts
    Post-match: Only known after results (potential leakage if used to predict future)
    """
    return {
        'pre_match_safe': [
            # Player static info
            'now_cost', 'selected_by_percent', 'position', 'team',
            
            # Pre-match estimates from FPL
            'ep_next', 'ep_this', 'chance_of_playing_next_round', 'chance_of_playing_this_round',
            
            # ICT Index components (calculated by FPL from historical data)
            'ict_index', 'influence', 'creativity', 'threat',
            
            # Form (FPL's rolling average - historical, not future)
            'form', 'points_per_game',
            
            # Fixture difficulty
            'fixture_difficulty', 'next_fixture_difficulty',
            
            # Historical rolling stats (from PAST games, not including current)
            'minutes',  # Only safe if it's cumulative BEFORE current GW
            'starts',
        ],
        'post_match_leakage': [
            # Season totals that include current/recent results
            'total_points',  # This IS the target!
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'saves', 'penalties_saved', 'penalties_missed',
            'yellow_cards', 'red_cards', 'own_goals',
            
            # Derived features that use season totals
            'points_per_90',  # Derived from total_points
            'goal_involvement',  # Derived from goals + assists
            'bonus_per_appearance',  # Derived from bonus
            'cs_rate',  # Derived from clean_sheets
        ],
        'needs_careful_handling': [
            # These can be safe if calculated from PAST GWs only
            'xG', 'xA', 'xGI', 'xGC',  # Understat data - safe if historical
            'team_strength',  # Safe if based on pre-season or long-term average
        ]
    }


def run_kfold_cv(df: pd.DataFrame, feature_cols: List[str], target_col: str, n_splits: int = 5) -> Dict:
    """
    Run K-Fold Cross-Validation to detect overfitting.
    
    If train scores >> test scores, the model is overfitting.
    If both are very high, there's likely target leakage.
    """
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Prepare data
    X = df[feature_cols].fillna(0).values
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_maes = []
    test_maes = []
    train_r2s = []
    test_r2s = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_maes.append(mean_absolute_error(y_train, train_pred))
        test_maes.append(mean_absolute_error(y_test, test_pred))
        train_r2s.append(r2_score(y_train, train_pred))
        test_r2s.append(r2_score(y_test, test_pred))
    
    results = {
        'n_splits': n_splits,
        'train_mae_mean': np.mean(train_maes),
        'train_mae_std': np.std(train_maes),
        'test_mae_mean': np.mean(test_maes),
        'test_mae_std': np.std(test_maes),
        'train_r2_mean': np.mean(train_r2s),
        'train_r2_std': np.std(train_r2s),
        'test_r2_mean': np.mean(test_r2s),
        'test_r2_std': np.std(test_r2s),
        'overfit_ratio': np.mean(test_maes) / max(np.mean(train_maes), 0.001),
    }
    
    # Diagnosis
    if results['train_r2_mean'] > 0.95 and results['test_r2_mean'] > 0.90:
        results['diagnosis'] = "LIKELY TARGET LEAKAGE - Both train and test R² are suspiciously high"
    elif results['overfit_ratio'] > 2.0:
        results['diagnosis'] = "OVERFITTING - Test error is 2x+ train error"
    elif results['overfit_ratio'] > 1.5:
        results['diagnosis'] = "MILD OVERFITTING - Consider regularization"
    else:
        results['diagnosis'] = "OK - Model generalizes reasonably"
    
    return results


def generate_report(df: pd.DataFrame) -> str:
    """Generate a full diagnostic report."""
    report = []
    report.append("=" * 70)
    report.append("ML DIAGNOSTICS REPORT - Target Leakage & Overfitting Analysis")
    report.append("=" * 70)
    report.append("")
    
    # 1. Feature Classification
    report.append("1. FEATURE CLASSIFICATION")
    report.append("-" * 40)
    categories = classify_features()
    report.append(f"   Safe pre-match features: {len(categories['pre_match_safe'])}")
    report.append(f"   Post-match (leakage risk): {len(categories['post_match_leakage'])}")
    report.append("")
    
    # 2. Correlation Analysis
    report.append("2. FEATURE CORRELATION WITH TARGET (ep_next)")
    report.append("-" * 40)
    
    target_col = 'ep_next' if 'ep_next' in df.columns else 'total_points'
    corr_df = compute_feature_correlations(df, target_col)
    
    # Show top 15 most correlated
    report.append("   Top 15 most correlated features:")
    for _, row in corr_df.head(15).iterrows():
        report.append(f"   {row['feature']:30s} r={row['correlation']:+.3f}  [{row['risk']}]")
    
    leaky = identify_leakage_features(corr_df, threshold=0.85)
    if leaky:
        report.append("")
        report.append(f"   HIGH LEAKAGE RISK FEATURES ({len(leaky)}):")
        for feat in leaky:
            report.append(f"      - {feat}")
    
    report.append("")
    
    # 3. K-Fold CV Analysis
    report.append("3. K-FOLD CROSS-VALIDATION (5 folds)")
    report.append("-" * 40)
    
    # Use current feature set from ml_predictor
    current_features = [
        'form', 'points_per_game', 'now_cost', 'selected_by_percent',
        'ict_index', 'influence', 'creativity', 'threat',
        'minutes', 'starts', 'total_points', 'goals_scored', 'assists',
        'clean_sheets', 'bonus'
    ]
    available_features = [f for f in current_features if f in df.columns]
    
    if len(available_features) >= 5:
        cv_results = run_kfold_cv(df, available_features, target_col)
        
        report.append(f"   Train MAE: {cv_results['train_mae_mean']:.3f} ± {cv_results['train_mae_std']:.3f}")
        report.append(f"   Test MAE:  {cv_results['test_mae_mean']:.3f} ± {cv_results['test_mae_std']:.3f}")
        report.append(f"   Train R²:  {cv_results['train_r2_mean']:.3f} ± {cv_results['train_r2_std']:.3f}")
        report.append(f"   Test R²:   {cv_results['test_r2_mean']:.3f} ± {cv_results['test_r2_std']:.3f}")
        report.append(f"   Overfit Ratio: {cv_results['overfit_ratio']:.2f}x")
        report.append("")
        report.append(f"   DIAGNOSIS: {cv_results['diagnosis']}")
    else:
        report.append("   Insufficient features for CV analysis")
    
    report.append("")
    
    # 4. Recommendations
    report.append("4. RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("   To fix target leakage:")
    report.append("   1. Remove 'total_points' from features (it IS the target)")
    report.append("   2. Remove derived features: points_per_90, bonus_per_appearance, etc.")
    report.append("   3. Use only PRE-MATCH features for prediction")
    report.append("   4. Safe features include: form, ep_next, ICT components, fixture data")
    report.append("")
    report.append("   Pre-match features to use:")
    for feat in categories['pre_match_safe'][:10]:
        report.append(f"      - {feat}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Loading FPL data...")
    try:
        df = load_data()
        print(f"Loaded {len(df)} players")
        print("")
        
        report = generate_report(df)
        print(report)
        
        # Also save to file
        with open("ml_diagnostics_report.txt", "w") as f:
            f.write(report)
        print("\nReport saved to: ml_diagnostics_report.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo run diagnostics, make sure you have:")
        print("1. FPL API access (fpl_api.py)")
        print("2. Required packages: pandas, numpy, scikit-learn")
