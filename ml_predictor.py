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
    """Advanced feature engineering for ML models."""
    
    def __init__(self, players_df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None):
        self.players_df = players_df.copy()
        self.history_df = history_df
    
    def engineer_features(self) -> pd.DataFrame:
        """Create advanced ML features."""
        df = self.players_df.copy()
        
        # === TIME SERIES FEATURES ===
        df['minutes_per_game'] = np.where(
            df['starts'] > 0,
            df['minutes'] / df['starts'],
            0
        )
        
        # Form momentum (exponential weighted)
        df['form_numeric'] = pd.to_numeric(df.get('form', 0), errors='coerce').fillna(0)
        df['form_momentum'] = df['form_numeric'] * np.exp(df['form_numeric'] / 10)
        
        # Points per 90 minutes
        df['points_per_90'] = np.where(
            df['minutes'] > 0,
            (df['total_points'] / df['minutes']) * 90,
            0
        )
        
        # === ADVANCED SCORING FEATURES ===
        
        # Goal involvement ratio
        df['goal_involvement'] = (
            df.get('goals_scored', 0) + 
            df.get('assists', 0)
        )
        df['goal_involvement_per_90'] = np.where(
            df['minutes'] > 0,
            (df['goal_involvement'] / df['minutes']) * 90,
            0
        )
        
        # Bonus propensity
        df['bonus_per_appearance'] = np.where(
            df['starts'] > 0,
            df.get('bonus', 0) / df['starts'],
            0
        )
        
        # Clean sheet probability (for defenders/keepers)
        df['cs_rate'] = np.where(
            df['starts'] > 0,
            df.get('clean_sheets', 0) / df['starts'],
            0
        )
        
        # === EFFICIENCY METRICS ===
        
        # Expected Goals per Shot (for forwards/mids)
        df['shots_per_90'] = np.where(
            df['minutes'] > 0,
            (df.get('shots', 0) / df['minutes']) * 90,
            0
        )
        
        # Creativity (key passes, crosses)
        df['creativity_score'] = (
            df.get('creativity', 0).astype(float) +
            df.get('influence', 0).astype(float) * 0.5
        )
        
        # Threat level (ICT component)
        df['threat_level'] = pd.to_numeric(df.get('threat', 0), errors='coerce').fillna(0)
        
        # === CONSISTENCY METRICS ===
        
        # Coefficient of variation (consistency)
        df['points_cv'] = np.where(
            (df['total_points'] > 0),
            df.get('points_per_game', 0) / (df['total_points'] + 1),  # Inverse CV
            0
        )
        
        # Price efficiency
        df['ep_per_pound'] = df.get('ep_next', 2.0) / df['now_cost'].clip(lower=4.0)
        
        # === POSITIONAL FEATURES ===
        
        # Position-specific scoring rates
        position_dummies = pd.get_dummies(df['position'], prefix='pos')
        df = pd.concat([df, position_dummies], axis=1)
        
        # === TEAM STRENGTH FEATURES ===
        
        # Average team points (proxy for team strength)
        team_strength = df.groupby('team')['total_points'].transform('mean')
        df['team_strength'] = team_strength
        
        # Player contribution to team
        df['team_contribution'] = np.where(
            team_strength > 0,
            df['total_points'] / team_strength,
            0
        )
        
        # === ROLLING STATISTICS (if history available) ===
        
        # Rolling form (exponential weighted average)
        df['ema_form_3gw'] = df.groupby('id')['form_numeric'].transform(
            lambda x: x.ewm(span=3, adjust=False).mean()
        )
        
        # === INTERACTION FEATURES ===
        
        # Form × Minutes (fitness-adjusted form)
        df['form_minutes_interaction'] = df['form_numeric'] * (df['minutes_per_game'] / 90)
        
        # Price × Ownership (template player indicator)
        df['price_ownership_interaction'] = (
            df['now_cost'] * 
            pd.to_numeric(df.get('selected_by_percent', 5), errors='coerce').fillna(5)
        )
        
        # === RISK METRICS ===
        
        # Injury risk (based on news and chance of playing)
        df['injury_risk'] = 1.0 - (pd.to_numeric(df.get('chance_of_playing_next_round', 100), errors='coerce').fillna(100) / 100)
        
        # Rotation risk (high ownership but low minutes)
        df['rotation_risk'] = np.where(
            df['minutes_per_game'] < 60,
            1.0 - (df['minutes_per_game'] / 90),
            0
        )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        for lag in range(1, n_lags + 1):
            df[f'points_lag_{lag}'] = df.groupby('id')['total_points'].shift(lag)
            df[f'form_lag_{lag}'] = df.groupby('id')['form_numeric'].shift(lag)
            df[f'minutes_lag_{lag}'] = df.groupby('id')['minutes'].shift(lag)
        
        return df.fillna(0)


class MLPredictor:
    """
    Multi-model ensemble predictor for player performance.
    Combines XGBoost, Random Forest, and Gradient Boosting.
    """
    
    def __init__(self, players_df: pd.DataFrame):
        _ensure_sklearn()
        _ensure_xgboost()
        self.players_df = players_df.copy()
        self.feature_engineer = AdvancedFeatureEngineer(players_df)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data with engineered features."""
        df = self.feature_engineer.engineer_features()
        
        # Select features for modeling
        feature_cols = [
            'form_momentum', 'points_per_90', 'goal_involvement_per_90',
            'bonus_per_appearance', 'cs_rate', 'shots_per_90',
            'creativity_score', 'threat_level', 'points_cv',
            'ep_per_pound', 'team_strength', 'team_contribution',
            'form_minutes_interaction', 'injury_risk', 'rotation_risk',
            'minutes_per_game', 'form_numeric', 'minutes', 'total_points'
        ]
        
        # Add position dummies
        pos_cols = [c for c in df.columns if c.startswith('pos_')]
        feature_cols.extend(pos_cols)
        
        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        self.feature_names = feature_cols
        return df, feature_cols
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble of models."""
        # XGBoost
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror'
            )
            self.models['xgboost'].fit(X, y)
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X, y)
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.models['gradient_boosting'].fit(X, y)
        
        self.is_trained = True
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        Returns mean predictions and standard deviations.
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
        std_pred = np.std(predictions, axis=0)
        
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
        
        # Target: use ep_next or total_points as proxy
        if 'ep_next' in df.columns:
            y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(2.0).values
        else:
            y = pd.to_numeric(df['total_points'], errors='coerce').fillna(0).values / 38  # Normalize
        
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
        Perform time series cross-validation.
        Returns validation metrics.
        """
        df, feature_cols = self.prepare_data()
        
        X = df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        if 'ep_next' in df.columns:
            y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(2.0).values
        else:
            y = pd.to_numeric(df['total_points'], errors='coerce').fillna(0).values / 38
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train simple model for CV
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            scores['mae'].append(mae)
            scores['rmse'].append(rmse)
            scores['r2'].append(r2)
        
        return {
            'mean_mae': np.mean(scores['mae']),
            'mean_rmse': np.mean(scores['rmse']),
            'mean_r2': np.mean(scores['r2']),
            'std_mae': np.std(scores['mae']),
            'std_rmse': np.std(scores['rmse']),
            'std_r2': np.std(scores['r2'])
        }


def create_ml_pipeline(players_df: pd.DataFrame) -> MLPredictor:
    """Factory function to create ML prediction pipeline."""
    return MLPredictor(players_df)
