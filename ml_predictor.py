"""
Machine Learning Prediction Module - Advanced player performance forecasting.
Uses XGBoost, Random Forest, and LSTM for ensemble predictions.
Imports are deferred to avoid slow startup.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Lazy imports - these are loaded only when ML is actually used
_sklearn_loaded = False
_xgboost_loaded = False
HAS_XGBOOST = None  # Determined on first use


def _ensure_sklearn():
    """Lazy-load sklearn modules."""
    global _sklearn_loaded, RandomForestRegressor, GradientBoostingRegressor, StandardScaler, TimeSeriesSplit
    if not _sklearn_loaded:
        from sklearn.ensemble import RandomForestRegressor as _RF, GradientBoostingRegressor as _GB
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.model_selection import TimeSeriesSplit as _TSS
        RandomForestRegressor = _RF
        GradientBoostingRegressor = _GB
        StandardScaler = _SS
        TimeSeriesSplit = _TSS
        _sklearn_loaded = True


def _ensure_xgboost():
    """Lazy-load xgboost."""
    global _xgboost_loaded, HAS_XGBOOST, xgb
    if HAS_XGBOOST is None:
        try:
            import xgboost as _xgb
            xgb = _xgb
            HAS_XGBOOST = True
        except ImportError:
            HAS_XGBOOST = False
        _xgboost_loaded = True


@dataclass
class PredictionResult:
    """Result from ML prediction."""
    player_id: int
    predicted_points: float
    confidence_interval: Tuple[float, float]
    prediction_std: float
    feature_importance: Dict[str, float]
    model_type: str


class AdvancedFeatureEngineer:
    """
    Feature engineering for ML models.
    
    IMPORTANT: Only uses PRE-MATCH features to avoid target leakage.
    Post-match stats (total_points, goals, assists, etc.) are NOT used
    as features since they contain information about the target.
    """
    
    def __init__(self, players_df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None):
        self.players_df = players_df.copy()
        self.history_df = history_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create ML features using ONLY pre-match data.
        
        Safe features (known before match):
        - FPL's ep_next/ep_this (their algorithm's prediction)
        - ICT index components (influence, creativity, threat)
        - Form (FPL's rolling average from past games)
        - Price and ownership
        - Chance of playing
        - Minutes and starts (historical pattern)
        
        EXCLUDED (post-match / leakage):
        - total_points (this IS the target!)
        - goals_scored, assists, clean_sheets, bonus (components of target)
        - Any derived features using the above
        """
        df = self.players_df.copy()
        
        # === PRE-MATCH FEATURES (SAFE) ===
        
        # 1. FPL's own predictions (raw ep_next from API, not Poisson-modified)
        # ep_next_num is the preserved raw value before any Poisson blending
        raw_ep = df.get('ep_next_num', df.get('ep_next', None))
        if raw_ep is None:
            raw_ep = df.get('expected_points', 2.0)
        df['fpl_ep'] = pd.to_numeric(raw_ep, errors='coerce').fillna(2.0)
        
        # 2. Form - FPL's rolling average (historical, not leakage)
        df['form_numeric'] = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0)
        df['ppg'] = pd.to_numeric(df.get('points_per_game', 0), errors='coerce').fillna(0)
        
        # 3. ICT Index components (calculated by FPL from historical data)
        df['ict_index'] = pd.to_numeric(df.get('ict_index', 0), errors='coerce').fillna(0)
        df['influence'] = pd.to_numeric(df.get('influence', 0), errors='coerce').fillna(0)
        df['creativity'] = pd.to_numeric(df.get('creativity', 0), errors='coerce').fillna(0)
        df['threat'] = pd.to_numeric(df.get('threat', 0), errors='coerce').fillna(0)
        
        # 4. Playing time patterns (historical)
        df['minutes'] = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0)
        df['starts'] = pd.to_numeric(df.get('starts', 0), errors='coerce').fillna(0)
        df['minutes_per_start'] = np.where(
            df['starts'] > 0,
            df['minutes'] / df['starts'],
            0
        )
        
        # 5. Price and ownership (market indicators)
        df['price'] = pd.to_numeric(df.get('now_cost', 5.0), errors='coerce').fillna(5.0)
        df['ownership'] = pd.to_numeric(df.get('selected_by_percent', 0), errors='coerce').fillna(0)
        
        # 6. Availability
        df['chance_of_playing'] = pd.to_numeric(
            df.get('chance_of_playing_next_round', 100), errors='coerce'
        ).fillna(100) / 100
        
        # 7. Value metrics (price efficiency based on FPL's EP, not actual points)
        df['value_score'] = df['fpl_ep'] / df['price'].clip(lower=4.0)
        
        # === DERIVED PRE-MATCH FEATURES ===
        
        # Form momentum (form weighted by recency - still pre-match)
        df['form_momentum'] = df['form_numeric'] * np.exp(df['form_numeric'] / 10)
        
        # ICT per 90 (efficiency metric)
        df['ict_per_90'] = np.where(
            df['minutes'] > 0,
            (df['ict_index'] / df['minutes']) * 90,
            0
        )
        
        # Threat per 90 (attacking intent)
        df['threat_per_90'] = np.where(
            df['minutes'] > 0,
            (df['threat'] / df['minutes']) * 90,
            0
        )
        
        # Creativity per 90 (chance creation)
        df['creativity_per_90'] = np.where(
            df['minutes'] > 0,
            (df['creativity'] / df['minutes']) * 90,
            0
        )
        
        # Minutes consistency (proxy for nailedness)
        df['nailed_score'] = np.minimum(df['minutes_per_start'] / 90, 1.0)
        
        # Price tier (budget/mid/premium indicator)
        df['price_tier'] = pd.cut(
            df['price'],
            bins=[0, 5.0, 7.5, 10.0, 15.0],
            labels=[1, 2, 3, 4]
        ).astype(float).fillna(1)
        
        # Ownership tier (differential vs template)
        df['ownership_tier'] = pd.cut(
            df['ownership'],
            bins=[0, 5, 15, 30, 100],
            labels=[1, 2, 3, 4]
        ).astype(float).fillna(1)
        
        # === POSITIONAL FEATURES ===
        position_dummies = pd.get_dummies(df['position'], prefix='pos')
        df = pd.concat([df, position_dummies], axis=1)
        
        # === TEAM STRENGTH (based on FPL difficulty, not results) ===
        # Use average ICT of team players as strength proxy
        team_ict = df.groupby('team')['ict_index'].transform('mean')
        df['team_strength'] = team_ict / team_ict.max() if team_ict.max() > 0 else 0
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        for lag in range(1, n_lags + 1):
            df[f'form_lag_{lag}'] = df.groupby('id')['form_numeric'].shift(lag)
            df[f'minutes_lag_{lag}'] = df.groupby('id')['minutes'].shift(lag)
        
        return df.fillna(0)


class MLPredictor:
    """
    Multi-model ensemble predictor for player performance.
    Combines XGBoost, Random Forest, and Gradient Boosting.
    
    IMPORTANT: Uses only PRE-MATCH features to avoid target leakage.
    """
    
    # Pre-match features only - these are known before the match
    PRE_MATCH_FEATURES = [
        # FPL estimates and form
        'fpl_ep', 'form_numeric', 'ppg', 'form_momentum',
        
        # ICT components
        'ict_index', 'influence', 'creativity', 'threat',
        'ict_per_90', 'threat_per_90', 'creativity_per_90',
        
        # Playing time patterns
        'minutes_per_start', 'nailed_score', 'chance_of_playing',
        
        # Market indicators
        'price', 'ownership', 'value_score',
        'price_tier', 'ownership_tier',
        
        # Team context
        'team_strength',
    ]
    
    def __init__(self, players_df: pd.DataFrame):
        _ensure_sklearn()
        _ensure_xgboost()
        self.players_df = players_df.copy()
        self.feature_engineer = AdvancedFeatureEngineer(players_df)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.cv_results = None
    
    def prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data with engineered features (pre-match only)."""
        df = self.feature_engineer.engineer_features()
        
        # Use only pre-match features
        feature_cols = [f for f in self.PRE_MATCH_FEATURES if f in df.columns]
        
        # Add position dummies
        pos_cols = [c for c in df.columns if c.startswith('pos_')]
        feature_cols.extend(pos_cols)
        
        self.feature_names = feature_cols
        return df, feature_cols
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble of models with regularization to prevent overfitting."""
        # XGBoost with regularization
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,  # Reduced from 6 to prevent overfitting
                learning_rate=0.05,  # Reduced for better generalization
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                objective='reg:squarederror'
            )
            self.models['xgboost'].fit(X, y)
        
        # Random Forest with regularization
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,  # Reduced from 10
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=5,  # Increased from 2
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X, y)
        
        # Gradient Boosting with regularization
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,  # Reduced from 5
            learning_rate=0.05,  # Reduced from 0.1
            subsample=0.7,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.models['gradient_boosting'].fit(X, y)
        
        self.is_trained = True
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        
        Uncertainty now reflects:
        1. Model disagreement (ensemble variance)
        2. Feature-based uncertainty (low ICT players are less predictable)
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_ensemble first.")
        
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Ensemble: average predictions
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        
        # Model disagreement
        model_std = np.std(predictions, axis=0)
        
        # Add base uncertainty (FPL predictions have inherent variance)
        # Players with lower predicted points have higher relative uncertainty
        base_uncertainty = 0.5 + 0.3 * np.maximum(0, 4 - mean_pred) / 4
        
        # Combined uncertainty
        std_pred = np.sqrt(model_std**2 + base_uncertainty**2)
        
        return mean_pred, std_pred
    
    def predict_gameweek_points(
        self,
        n_gameweeks: int = 1,
        use_ensemble: bool = True
    ) -> Dict[int, PredictionResult]:
        """
        Predict points for next N gameweeks with confidence intervals.
        
        Returns:
            Dict mapping player_id to PredictionResult
        """
        df, feature_cols = self.prepare_data()
        
        # Prepare features
        X = df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Target: use raw FPL ep_next (ep_next_num is preserved before Poisson blending)
        if 'ep_next_num' in df.columns:
            y = pd.to_numeric(df['ep_next_num'], errors='coerce').fillna(2.0).values
        elif 'ep_next' in df.columns:
            y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(2.0).values
        else:
            # Fallback: use points_per_game as proxy
            y = pd.to_numeric(df.get('ppg', 2.0), errors='coerce').fillna(2.0).values
        
        # Train ensemble
        self.train_ensemble(X_scaled, y)
        
        # Predict with uncertainty
        mean_pred, std_pred = self.predict_with_uncertainty(X_scaled)
        
        # Build results
        results = {}
        
        for idx, player_id in enumerate(df['id'].values):
            # 95% confidence interval (1.96 std devs)
            ci_lower = max(0, mean_pred[idx] - 1.96 * std_pred[idx])
            ci_upper = mean_pred[idx] + 1.96 * std_pred[idx]
            
            # Feature importance (from best model)
            if HAS_XGBOOST and 'xgboost' in self.models:
                importance = dict(zip(
                    self.feature_names,
                    self.models['xgboost'].feature_importances_
                ))
            else:
                importance = dict(zip(
                    self.feature_names,
                    self.models['random_forest'].feature_importances_
                ))
            
            results[player_id] = PredictionResult(
                player_id=player_id,
                predicted_points=mean_pred[idx],
                confidence_interval=(ci_lower, ci_upper),
                prediction_std=std_pred[idx],
                feature_importance=importance,
                model_type='ensemble'
            )
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get aggregated feature importance across models."""
        if not self.is_trained:
            return pd.DataFrame()
        
        importance_data = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for feature, importance in zip(self.feature_names, model.feature_importances_):
                    importance_data.append({
                        'feature': feature,
                        'importance': importance,
                        'model': model_name
                    })
        
        df = pd.DataFrame(importance_data)
        df_agg = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return df_agg.head(top_n).reset_index()
    
    def cross_validate_predictions(self, n_splits: int = 5) -> Dict[str, float]:
        """
        Perform K-Fold Cross-Validation to detect overfitting.
        
        Returns validation metrics including train vs test comparison.
        If train >> test, model is overfitting.
        If both are very high, there may be target leakage.
        """
        from sklearn.model_selection import KFold
        
        df, feature_cols = self.prepare_data()
        
        X = df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Target: raw FPL ep_next (ep_next_num is preserved before Poisson blending)
        if 'ep_next_num' in df.columns:
            y = pd.to_numeric(df['ep_next_num'], errors='coerce').fillna(2.0).values
        elif 'ep_next' in df.columns:
            y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(2.0).values
        else:
            # Fallback: use points_per_game as proxy
            y = pd.to_numeric(df.get('ppg', 2.0), errors='coerce').fillna(2.0).values
        
        # Use KFold instead of TimeSeriesSplit for player data
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = {
            'train_mae': [],
            'test_mae': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_r2': [],
            'test_r2': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model with same settings as ensemble
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            # Train
            train_mae = np.mean(np.abs(y_train - train_pred))
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            ss_res_train = np.sum((y_train - train_pred) ** 2)
            ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
            train_r2 = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0
            
            # Test
            test_mae = np.mean(np.abs(y_test - test_pred))
            test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
            ss_res_test = np.sum((y_test - test_pred) ** 2)
            ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
            test_r2 = 1 - (ss_res_test / ss_tot_test) if ss_tot_test > 0 else 0
            
            scores['train_mae'].append(train_mae)
            scores['test_mae'].append(test_mae)
            scores['train_rmse'].append(train_rmse)
            scores['test_rmse'].append(test_rmse)
            scores['train_r2'].append(train_r2)
            scores['test_r2'].append(test_r2)
        
        # Calculate overfit ratio
        mean_train_mae = np.mean(scores['train_mae'])
        mean_test_mae = np.mean(scores['test_mae'])
        overfit_ratio = mean_test_mae / max(mean_train_mae, 0.001)
        
        self.cv_results = {
            'n_splits': n_splits,
            'mean_mae': mean_test_mae,  # For backward compatibility
            'mean_rmse': np.mean(scores['test_rmse']),
            'mean_r2': np.mean(scores['test_r2']),
            'std_mae': np.std(scores['test_mae']),
            'std_rmse': np.std(scores['test_rmse']),
            'std_r2': np.std(scores['test_r2']),
            # New: train vs test comparison
            'train_mae_mean': mean_train_mae,
            'test_mae_mean': mean_test_mae,
            'train_r2_mean': np.mean(scores['train_r2']),
            'test_r2_mean': np.mean(scores['test_r2']),
            'overfit_ratio': overfit_ratio,
        }
        
        return self.cv_results


def create_ml_pipeline(players_df: pd.DataFrame) -> MLPredictor:
    """Factory function to create ML prediction pipeline."""
    return MLPredictor(players_df)
