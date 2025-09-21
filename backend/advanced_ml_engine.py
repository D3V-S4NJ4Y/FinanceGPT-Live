import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    symbol: str
    predicted_price: float
    confidence: float
    direction: str  # 'up', 'down', 'neutral'
    probability: float
    target_price: float
    stop_loss: float
    time_horizon: str
    risk_score: float
    model_used: str
    features_importance: Dict[str, float]

@dataclass
class TechnicalIndicators:
    rsi: float
    macd: float
    bollinger_upper: float
    bollinger_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    atr: float
    volume_sma: float

class AdvancedMLTradingEngine:
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_cache = {}
        self.last_training = {}
        
        # Model hyperparameters
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': 42
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': 42
            }
        }
        
        logger.info(" Advanced ML Trading Engine initialized")
    
    async def get_enhanced_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get comprehensive market data with technical indicators"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            data = self._calculate_technical_indicators(data)
            
            # Add additional features
            data = self._add_advanced_features(data)
            
            logger.info(f"✅ Enhanced market data prepared for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['Bollinger_Upper'] = data['SMA_20'] + (rolling_std * 2)
            data['Bollinger_Lower'] = data['SMA_20'] - (rolling_std * 2)
            data['Bollinger_Width'] = data['Bollinger_Upper'] - data['Bollinger_Lower']
            data['Bollinger_Position'] = (data['Close'] - data['Bollinger_Lower']) / data['Bollinger_Width']
            
            # Moving Averages
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # Stochastic Oscillator
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['Stochastic_K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
            data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()
            
            # Williams %R
            data['Williams_R'] = -100 * (high_max - data['Close']) / (high_max - low_min)
            
            # Average True Range (ATR)
            data['TR'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            data['ATR'] = data['TR'].rolling(window=14).mean()
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price momentum
            data['Price_Momentum'] = data['Close'] / data['Close'].shift(10) - 1
            data['Price_Velocity'] = data['Close'].pct_change(5)
            data['Price_Acceleration'] = data['Price_Velocity'].diff()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return data
    
    def _add_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced engineered features"""
        try:
            # Price patterns
            data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
            data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
            data['Doji'] = (abs(data['Close'] - data['Open']) < (data['High'] - data['Low']) * 0.1).astype(int)
            
            # Volatility measures
            data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
            data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
            
            # Trend strength
            data['Trend_Strength'] = abs(data['Close'] - data['SMA_20']) / data['SMA_20']
            data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            
            # Market structure
            data['Support_Level'] = data['Low'].rolling(window=20).min()
            data['Resistance_Level'] = data['High'].rolling(window=20).max()
            data['Distance_To_Support'] = (data['Close'] - data['Support_Level']) / data['Close']
            data['Distance_To_Resistance'] = (data['Resistance_Level'] - data['Close']) / data['Close']
            
            # Time-based features
            data['Day_Of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            data['Quarter'] = data.index.quarter
            
            # Target variable (next day return)
            data['Target'] = data['Close'].shift(-1) / data['Close'] - 1
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding advanced features: {e}")
            return data
    
    async def train_models(self, symbol: str, retrain: bool = False) -> Dict[str, Any]:
        """Train multiple ML models for a symbol"""
        try:
            # Check if models need retraining
            if symbol in self.last_training and not retrain:
                time_since_training = datetime.now() - self.last_training[symbol]
                if time_since_training < timedelta(hours=6):  # Retrain every 6 hours
                    logger.info(f"⏰ Models for {symbol} are recent, skipping training")
                    return {"status": "models_recent"}
            
            logger.info(f"🏋️ Training ML models for {symbol}...")
            
            # Get enhanced data
            data = await self.get_enhanced_market_data(symbol, period="2y")
            if data.empty:
                return {"error": "No data available"}
            
            # Prepare features
            feature_columns = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'Bollinger_Position', 'Bollinger_Width',
                'Stochastic_K', 'Stochastic_D', 'Williams_R',
                'ATR', 'Volume_Ratio', 'Price_Momentum',
                'Price_Velocity', 'Trend_Strength', 'Price_Position',
                'Distance_To_Support', 'Distance_To_Resistance',
                'Volatility', 'Price_Range'
            ]
            
            # Remove NaN values and prepare data
            clean_data = data[feature_columns + ['Target']].dropna()
            if len(clean_data) < 100:
                logger.warning(f"Insufficient data for training {symbol}: {len(clean_data)} samples")
                return {"error": "Insufficient data"}
            
            X = clean_data[feature_columns]
            y = clean_data['Target']
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[symbol] = scaler
            
            # Train models
            models_performance = {}
            
            # Random Forest
            rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_score = r2_score(y_test, rf_pred)
            rf_mse = mean_squared_error(y_test, rf_pred)
            
            models_performance['random_forest'] = {
                'r2_score': rf_score,
                'mse': rf_mse,
                'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
            }
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            gb_score = r2_score(y_test, gb_pred)
            gb_mse = mean_squared_error(y_test, gb_pred)
            
            models_performance['gradient_boosting'] = {
                'r2_score': gb_score,
                'mse': gb_mse,
                'feature_importance': dict(zip(feature_columns, gb_model.feature_importances_))
            }
            
            # Ridge Regression
            ridge_model = Ridge(**self.model_configs['ridge'])
            ridge_model.fit(X_train_scaled, y_train)
            ridge_pred = ridge_model.predict(X_test_scaled)
            ridge_score = r2_score(y_test, ridge_pred)
            ridge_mse = mean_squared_error(y_test, ridge_pred)
            
            models_performance['ridge'] = {
                'r2_score': ridge_score,
                'mse': ridge_mse,
                'feature_importance': dict(zip(feature_columns, abs(ridge_model.coef_)))
            }
            
            # Store models
            self.models[symbol] = {
                'random_forest': rf_model,
                'gradient_boosting': gb_model,
                'ridge': ridge_model
            }
            
            # Store feature importance
            self.feature_importance[symbol] = models_performance
            
            # Update last training time
            self.last_training[symbol] = datetime.now()
            
            # Select best model
            best_model = max(models_performance.keys(), 
            key=lambda x: models_performance[x]['r2_score'])
            
            logger.info(f"✅ Models trained for {symbol}. Best model: {best_model} (R²: {models_performance[best_model]['r2_score']:.4f})")
            
            return {
                "status": "success",
                "best_model": best_model,
                "performance": models_performance,
                "data_points": len(clean_data),
                "features": feature_columns
            }
            
        except Exception as e:
            logger.error(f"❌ Error training models for {symbol}: {e}")
            return {"error": str(e)}
    
    async def generate_prediction(self, symbol: str, use_ensemble: bool = True) -> Optional[PredictionResult]:
        """Generate advanced ML prediction for a symbol"""
        try:
            logger.info(f" Generating prediction for {symbol}...")
            
            # Ensure models are trained
            if symbol not in self.models:
                await self.train_models(symbol)
                if symbol not in self.models:
                    logger.error(f"Failed to train models for {symbol}")
                    return None
            
            # Get latest data
            data = await self.get_enhanced_market_data(symbol, period="6mo")
            if data.empty or len(data) < 50:
                logger.error(f"Insufficient data for prediction: {symbol}")
                return None
            
            # Prepare features
            feature_columns = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'Bollinger_Position', 'Bollinger_Width',
                'Stochastic_K', 'Stochastic_D', 'Williams_R',
                'ATR', 'Volume_Ratio', 'Price_Momentum',
                'Price_Velocity', 'Trend_Strength', 'Price_Position',
                'Distance_To_Support', 'Distance_To_Resistance',
                'Volatility', 'Price_Range'
            ]
            
            # Get latest features
            latest_features = data[feature_columns].iloc[-1:].dropna()
            if latest_features.empty:
                logger.error(f"No valid features for prediction: {symbol}")
                return None
            
            # Scale features
            scaler = self.scalers[symbol]
            features_scaled = scaler.transform(latest_features)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Generate predictions from all models
            predictions = {}
            for model_name, model in self.models[symbol].items():
                pred_return = model.predict(features_scaled)[0]
                predictions[model_name] = pred_return
            
            # Ensemble prediction (weighted average based on performance)
            if use_ensemble and symbol in self.feature_importance:
                weights = {}
                total_r2 = 0
                for model_name in predictions.keys():
                    r2_score = self.feature_importance[symbol][model_name]['r2_score']
                    weights[model_name] = max(0, r2_score)  # Ensure non-negative weights
                    total_r2 += weights[model_name]
                
                if total_r2 > 0:
                    # Normalize weights
                    for model_name in weights:
                        weights[model_name] /= total_r2
                    
                    # Calculate ensemble prediction
                    ensemble_return = sum(predictions[model] * weights[model] 
                                        for model in predictions.keys())
                else:
                    # Fallback to simple average
                    ensemble_return = np.mean(list(predictions.values()))
                
                best_model = "ensemble"
            else:
                # Use best performing model
                best_model = max(self.feature_importance[symbol].keys(),
                key=lambda x: self.feature_importance[symbol][x]['r2_score'])
                ensemble_return = predictions[best_model]
            
            # Calculate prediction details
            predicted_price = current_price * (1 + ensemble_return)
            
            # Determine direction and confidence
            if ensemble_return > 0.02:  # > 2%
                direction = "bullish"
                probability = min(0.95, 0.5 + abs(ensemble_return) * 10)
            elif ensemble_return < -0.02:  # < -2%
                direction = "bearish"
                probability = min(0.95, 0.5 + abs(ensemble_return) * 10)
            else:
                direction = "neutral"
                probability = 0.6
            
            # Calculate risk metrics
            volatility = data['Volatility'].iloc[-1] if not pd.isna(data['Volatility'].iloc[-1]) else 0.02
            risk_score = min(1.0, volatility * 50)  # Scale volatility to 0-1 risk score
            
            # Set target and stop loss
            if direction == "bullish":
                target_price = current_price * (1 + abs(ensemble_return) * 1.5)
                stop_loss = current_price * (1 - volatility * 2)
            elif direction == "bearish":
                target_price = current_price * (1 + ensemble_return * 1.5)
                stop_loss = current_price * (1 + volatility * 2)
            else:
                target_price = predicted_price
                stop_loss = current_price * (1 - volatility * 1.5)
            
            # Get feature importance
            if symbol in self.feature_importance and best_model in self.feature_importance[symbol]:
                features_importance = self.feature_importance[symbol][best_model]['feature_importance']
            else:
                features_importance = {}
            
            result = PredictionResult(
                symbol=symbol,
                predicted_price=round(predicted_price, 2),
                confidence=round(probability, 3),
                direction=direction,
                probability=round(probability, 3),
                target_price=round(target_price, 2),
                stop_loss=round(stop_loss, 2),
                time_horizon="1-5 days",
                risk_score=round(risk_score, 3),
                model_used=best_model,
                features_importance=features_importance
            )
            
            # Cache result
            self.prediction_cache[symbol] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Prediction generated for {symbol}: {direction} (${predicted_price:.2f}, confidence: {probability:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating prediction for {symbol}: {e}")
            return None
    
    async def get_portfolio_optimization(self, symbols: List[str], 
        risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """Generate optimal portfolio allocation using ML predictions"""
        try:
            logger.info(f" Generating portfolio optimization for {len(symbols)} symbols...")
            
            # Get predictions for all symbols
            predictions = {}
            for symbol in symbols:
                pred = await self.generate_prediction(symbol)
                if pred:
                    predictions[symbol] = pred
            
            if not predictions:
                return {"error": "No valid predictions available"}
            
            # Calculate expected returns and risks
            expected_returns = {}
            risk_scores = {}
            
            for symbol, pred in predictions.items():
                # Convert prediction to expected return
                expected_return = (pred.predicted_price / pred.target_price - 1) if pred.target_price > 0 else 0
                expected_returns[symbol] = expected_return * pred.confidence
                risk_scores[symbol] = pred.risk_score
            
            # Simple portfolio optimization (risk-adjusted returns)
            total_score = 0
            scores = {}
            
            for symbol in expected_returns:
                # Risk-adjusted score
                risk_adjustment = 1 - (risk_scores[symbol] * (1 - risk_tolerance))
                score = expected_returns[symbol] * risk_adjustment
                scores[symbol] = max(0, score)  # Ensure non-negative
                total_score += scores[symbol]
            
            # Calculate optimal weights
            if total_score > 0:
                weights = {symbol: score / total_score for symbol, score in scores.items()}
            else:
                # Equal weights if no positive scores
                weights = {symbol: 1/len(symbols) for symbol in symbols}
            
            # Ensure weights sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
            
            # Calculate portfolio metrics
            portfolio_expected_return = sum(expected_returns[symbol] * weights[symbol] 
            for symbol in weights)
            portfolio_risk = sum(risk_scores[symbol] * weights[symbol] 
            for symbol in weights)
            
            # Generate rebalancing suggestions
            suggestions = []
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                pred = predictions[symbol]
                suggestions.append({
                    "symbol": symbol,
                    "recommended_weight": round(weight * 100, 1),
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                    "expected_return": round(expected_returns[symbol] * 100, 2),
                    "risk_score": pred.risk_score,
                    "predicted_price": pred.predicted_price
                })
            
            return {
                "status": "success",
                "portfolio_metrics": {
                    "expected_return": round(portfolio_expected_return * 100, 2),
                    "risk_score": round(portfolio_risk, 3),
                    "sharpe_estimate": round(portfolio_expected_return / max(portfolio_risk, 0.01), 2),
                    "diversification": len(weights)
                },
                "optimal_weights": weights,
                "rebalancing_suggestions": suggestions,
                "risk_tolerance": risk_tolerance,
                "total_symbols_analyzed": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio optimization: {e}")
            return {"error": str(e)}
    
    async def get_market_regime_analysis(self) -> Dict[str, Any]:
        """Analyze current market regime using ML"""
        try:
            logger.info(" Analyzing market regime...")
            
            # Analyze major indices
            indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
            regime_data = {}
            
            for index in indices:
                data = await self.get_enhanced_market_data(index, period="6mo")
                if not data.empty:
                    # Calculate regime indicators
                    latest = data.iloc[-1]
                    regime_data[index] = {
                        'trend_strength': latest.get('Trend_Strength', 0),
                        'volatility': latest.get('Volatility', 0),
                        'rsi': latest.get('RSI', 50),
                        'bollinger_position': latest.get('Bollinger_Position', 0.5)
                    }
            
            # Determine market regime
            if 'SPY' in regime_data:
                spy_data = regime_data['SPY']
                
                # Market regime classification
                if spy_data['volatility'] > 0.02 and spy_data['rsi'] < 30:
                    regime = "bear_market"
                    confidence = 0.8
                elif spy_data['volatility'] < 0.015 and spy_data['rsi'] > 70:
                    regime = "bull_market"
                    confidence = 0.85
                elif spy_data['volatility'] > 0.025:
                    regime = "high_volatility"
                    confidence = 0.75
                else:
                    regime = "neutral"
                    confidence = 0.6
            else:
                regime = "unknown"
                confidence = 0.0
            
            return {
                "regime": regime,
                "confidence": confidence,
                "indicators": regime_data,
                "analysis_time": datetime.now().isoformat(),
                "recommendations": self._get_regime_recommendations(regime, confidence)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in market regime analysis: {e}")
            return {"error": str(e)}
    
    def _get_regime_recommendations(self, regime: str, confidence: float) -> List[str]:
        """Get recommendations based on market regime"""
        recommendations = []
        
        if regime == "bull_market":
            recommendations = [
                "Consider increasing equity exposure",
                "Focus on growth stocks and momentum strategies",
                "Reduce cash positions gradually",
                "Monitor for signs of market overheating"
            ]
        elif regime == "bear_market":
            recommendations = [
                "Increase defensive positions",
                "Consider value stocks and dividend strategies",
                "Maintain higher cash reserves",
                "Look for oversold opportunities"
            ]
        elif regime == "high_volatility":
            recommendations = [
                "Reduce position sizes",
                "Focus on risk management",
                "Consider volatility-based strategies",
                "Avoid momentum trades"
            ]
        else:
            recommendations = [
                "Maintain balanced portfolio",
                "Focus on quality stocks",
                "Regular rebalancing",
                "Monitor market developments closely"
            ]
        
        return recommendations

# Global instance
ml_engine = AdvancedMLTradingEngine()

async def get_ml_prediction(symbol: str) -> Optional[Dict[str, Any]]:
    """Get ML prediction for a symbol"""
    try:
        prediction = await ml_engine.generate_prediction(symbol)
        if prediction:
            return asdict(prediction)
        return None
    except Exception as e:
        logger.error(f"Error getting ML prediction for {symbol}: {e}")
        return None

async def get_portfolio_optimization(symbols: List[str], 
    risk_tolerance: float = 0.5) -> Dict[str, Any]:
    """Get portfolio optimization recommendations"""
    try:
        return await ml_engine.get_portfolio_optimization(symbols, risk_tolerance)
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return {"error": str(e)}

async def get_market_regime() -> Dict[str, Any]:
    """Get current market regime analysis"""
    try:
        return await ml_engine.get_market_regime_analysis()
    except Exception as e:
        logger.error(f"Error in market regime analysis: {e}")
        return {"error": str(e)}
