"""
🤖 Enhanced ML Engine for Real-Time Financial Predictions
=========================================================
Real machine learning models trained on live market data
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import ta

logger = logging.getLogger(__name__)

class RealTimeMLPredictor:
    """
    Real-time ML predictor using actual market data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cache = {}
        self.last_training = {}
        self.min_data_points = 100  # Minimum data points needed for training
        
        # Initialize popular symbols for quick access
        self.popular_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "AMD", "INTC", "JPM", "BAC", "JNJ", "UNH", "PG", "KO", "PFE", "V", "MA"
        ]
        
        logger.info("✅ RealTimeMLPredictor initialized")
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators as ML features
        """
        try:
            # Ensure we have enough data
            if len(df) < 20:
                return pd.DataFrame()
            
            features_df = pd.DataFrame(index=df.index)
            
            # Price-based features
            features_df['close'] = df['Close']
            features_df['volume'] = df['Volume']
            features_df['high'] = df['High']
            features_df['low'] = df['Low']
            
            # Returns
            features_df['returns_1d'] = df['Close'].pct_change()
            features_df['returns_5d'] = df['Close'].pct_change(5)
            features_df['returns_10d'] = df['Close'].pct_change(10)
            
            # Moving averages
            features_df['sma_5'] = df['Close'].rolling(5).mean()
            features_df['sma_10'] = df['Close'].rolling(10).mean()
            features_df['sma_20'] = df['Close'].rolling(20).mean()
            
            # Technical indicators using ta library
            if len(df) >= 14:  # RSI needs at least 14 periods
                features_df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            if len(df) >= 26:  # MACD needs at least 26 periods
                macd = ta.trend.MACD(df['Close'])
                features_df['macd'] = macd.macd()
                features_df['macd_signal'] = macd.macd_signal()
            
            if len(df) >= 20:  # Bollinger bands need 20 periods
                bb = ta.volatility.BollingerBands(df['Close'])
                features_df['bb_upper'] = bb.bollinger_hband()
                features_df['bb_lower'] = bb.bollinger_lband()
                features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
            
            # Volatility
            features_df['volatility_10d'] = df['Close'].rolling(10).std()
            features_df['volatility_20d'] = df['Close'].rolling(20).std()
            
            # Volume indicators
            features_df['volume_sma_10'] = df['Volume'].rolling(10).mean()
            features_df['volume_ratio'] = df['Volume'] / features_df['volume_sma_10']
            
            # Price position relative to ranges
            features_df['price_position_20d'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            return pd.DataFrame()
    
    async def train_model(self, symbol: str) -> bool:
        """
        Train ML model for a specific symbol using real market data
        """
        try:
            # Get historical data (6 months for training)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo", interval="1d")
            
            if hist.empty or len(hist) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(hist)} points")
                return False
            
            # Calculate technical features
            features_df = self.calculate_technical_features(hist)
            
            if features_df.empty or len(features_df) < 50:
                logger.warning(f"Insufficient features for {symbol}: {len(features_df)} points")
                return False
            
            # Prepare target variables
            # Predict next day's return direction and magnitude
            features_df['target_return'] = features_df['close'].shift(-1) / features_df['close'] - 1
            features_df['target_direction'] = (features_df['target_return'] > 0).astype(int)
            
            # Remove last row (no target)
            features_df = features_df[:-1]
            
            # Prepare features and targets
            feature_columns = [col for col in features_df.columns if col not in ['target_return', 'target_direction']]
            X = features_df[feature_columns]
            y_return = features_df['target_return']
            y_direction = features_df['target_direction']
            
            # Handle any remaining NaN values using new pandas methods
            X = X.ffill().bfill()
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_return_train, y_return_test, y_direction_train, y_direction_test = train_test_split(
                X_scaled, y_return, y_direction, test_size=0.2, random_state=42
            )
            
            # Train return magnitude model (regression)
            return_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            return_model.fit(X_train, y_return_train)
            
            # Train direction model (classification)
            direction_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
            direction_model.fit(X_train, y_direction_train)
            
            # Evaluate models
            return_score = return_model.score(X_test, y_return_test)
            direction_score = direction_model.score(X_test, y_direction_test)
            
            # Store models and scaler
            self.models[symbol] = {
                'return_model': return_model,
                'direction_model': direction_model,
                'return_score': return_score,
                'direction_score': direction_score,
                'feature_columns': feature_columns,
                'trained_at': datetime.utcnow()
            }
            self.scalers[symbol] = scaler
            
            logger.info(f"✅ Model trained for {symbol} - Return R²: {return_score:.3f}, Direction Acc: {direction_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training model for {symbol}: {e}")
            return False
    
    async def predict(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Make real ML prediction for a symbol
        """
        try:
            # Check if model exists and is recent (retrain if older than 1 day)
            if symbol not in self.models or \
               (datetime.utcnow() - self.models[symbol]['trained_at']).days > 1:
                
                logger.info(f"Training/retraining model for {symbol}...")
                success = await self.train_model(symbol)
                if not success:
                    return None
            
            # Get recent data for prediction
            ticker = yf.Ticker(symbol)
            recent_data = ticker.history(period="2mo", interval="1d")
            
            if recent_data.empty:
                return None
            
            # Calculate features
            features_df = self.calculate_technical_features(recent_data)
            
            if features_df.empty:
                return None
            
            # Get the most recent features
            latest_features = features_df.iloc[-1]
            
            model_info = self.models[symbol]
            feature_columns = model_info['feature_columns']
            
            # Prepare feature vector
            X = latest_features[feature_columns].values.reshape(1, -1)
            X = np.nan_to_num(X)  # Handle any NaN values
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(X)
            
            # Make predictions
            predicted_return = model_info['return_model'].predict(X_scaled)[0]
            predicted_direction_prob = model_info['direction_model'].predict_proba(X_scaled)[0]
            predicted_direction = model_info['direction_model'].predict(X_scaled)[0]
            
            current_price = float(recent_data['Close'].iloc[-1])
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence based on model performance and prediction certainty
            direction_confidence = max(predicted_direction_prob) * 100
            return_confidence = min(100, max(50, model_info['return_score'] * 100))
            overall_confidence = (direction_confidence + return_confidence) / 2
            
            # Determine risk score based on volatility and prediction magnitude
            recent_volatility = recent_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
            risk_score = min(100, max(10, recent_volatility * 100))
            
            # Calculate support and resistance levels
            recent_high = recent_data['High'].tail(20).max()
            recent_low = recent_data['Low'].tail(20).min()
            
            prediction = {
                "symbol": symbol,
                "predicted_price": float(predicted_price),
                "confidence": float(overall_confidence),
                "direction": "up" if predicted_direction == 1 else "down",
                "probability": float(max(predicted_direction_prob)),
                "target_price": float(predicted_price),
                "stop_loss": float(current_price * (0.95 if predicted_direction == 1 else 1.05)),
                "time_horizon": "1d",
                "risk_score": float(risk_score),
                "model_used": f"RF+GBM (R²={model_info['return_score']:.2f})",
                "technical_score": float(direction_confidence),
                "momentum_score": float(abs(predicted_return) * 100),
                "volatility": float(recent_volatility),
                "last_updated": datetime.utcnow().isoformat(),
                "current_price": float(current_price),
                "support_level": float(recent_low),
                "resistance_level": float(recent_high),
                "model_performance": {
                    "return_accuracy": model_info['return_score'],
                    "direction_accuracy": model_info['direction_score']
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error for {symbol}: {e}")
            return None
    
    async def get_batch_predictions(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get predictions for multiple symbols
        """
        predictions = {}
        
        # Process symbols concurrently but limit to avoid rate limiting
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def predict_with_semaphore(symbol):
            async with semaphore:
                return await self.predict(symbol)
        
        tasks = [predict_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error predicting {symbol}: {result}")
                continue
            
            if result is not None:
                predictions[symbol] = result
        
        return predictions
    
    async def get_market_predictions(self) -> Dict[str, Any]:
        """
        Get predictions for popular market symbols
        """
        return await self.get_batch_predictions(self.popular_symbols[:10])  # Top 10 to avoid rate limits

# Global instance
ml_predictor = RealTimeMLPredictor()
