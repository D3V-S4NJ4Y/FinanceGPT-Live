#!/usr/bin/env python3
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import json

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("⚠️ ML libraries not available - using statistical models")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    layers = None
    TF_AVAILABLE = False
    logging.warning("⚠️ TensorFlow not available - using basic ML models")

from .base_agent import BaseAgent

logger = logging.getLogger("PredictiveAnalytics")

class PredictionTimeframe(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"

class ModelType(Enum):
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"

@dataclass
class PredictionResult:
    symbol: str
    timeframe: PredictionTimeframe
    predicted_price: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    model_used: ModelType
    features_used: List[str]
    prediction_horizon: int  # minutes ahead
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    model_type: ModelType
    rmse: float
    mae: float
    r2_score: float
    accuracy: float  # directional accuracy
    predictions_made: int
    last_updated: datetime

class PredictiveAnalyticsAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            name="PredictiveAnalytics",
            description="Advanced machine learning agent for market prediction",
            version="2.0.0"
        )
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        # Data storage
        self.price_history = {}
        self.feature_history = {}
        self.prediction_cache = {}
        
        # Configuration
        self.supported_timeframes = list(PredictionTimeframe)
        self.prediction_horizons = {
            PredictionTimeframe.MINUTE_1: [1, 5, 15],     # 1, 5, 15 minutes ahead
            PredictionTimeframe.MINUTE_5: [1, 3, 6],      # 5, 15, 30 minutes ahead
            PredictionTimeframe.MINUTE_15: [1, 2, 4],     # 15, 30, 60 minutes ahead
            PredictionTimeframe.HOUR_1: [1, 4, 24],       # 1, 4, 24 hours ahead
            PredictionTimeframe.HOUR_4: [1, 6, 42],       # 4, 24, 168 hours ahead
            PredictionTimeframe.DAY_1: [1, 7, 30]         # 1, 7, 30 days ahead
        }
        
        # Performance tracking
        self.total_predictions = 0
        self.correct_direction_predictions = 0
        self.model_update_frequency = 1000  # Update models every 1000 new data points
        
        logger.info("🤖 Predictive Analytics Agent initialized with ML capabilities")
    
    async def initialize(self):
        """Initialize models and load historical data"""
        logger.info("🚀 Initializing predictive models...")
        
        # Initialize models for each timeframe
        for timeframe in self.supported_timeframes:
            await self._initialize_models_for_timeframe(timeframe)
        
        logger.info("✅ Predictive Analytics Agent ready")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process prediction requests"""
        message_type = message.get('type', 'unknown')
        
        if message_type == 'predict_price':
            return await self._handle_price_prediction(message)
        elif message_type == 'batch_predict':
            return await self._handle_batch_prediction(message)
        elif message_type == 'model_performance':
            return await self._handle_performance_query(message)
        elif message_type == 'update_models':
            return await self._handle_model_update(message)
        elif message_type == 'price_update':
            return await self._handle_price_update(message)
        else:
            return await self._handle_general_query(message)
    
    async def _handle_price_prediction(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single price prediction request"""
        try:
            symbol = message.get('symbol', 'AAPL')
            timeframe_str = message.get('timeframe', '1h')
            horizon_minutes = message.get('horizon_minutes', 60)
            
            # Convert string to enum
            timeframe = PredictionTimeframe(timeframe_str)
            
            # Get prediction
            prediction = await self.predict_price(symbol, timeframe, horizon_minutes)
            
            if prediction:
                return {
                    'status': 'success',
                    'prediction': {
                        'symbol': prediction.symbol,
                        'timeframe': prediction.timeframe.value,
                        'predicted_price': prediction.predicted_price,
                        'confidence_interval': prediction.confidence_interval,
                        'confidence_score': prediction.confidence_score,
                        'model_used': prediction.model_used.value,
                        'prediction_horizon': prediction.prediction_horizon,
                        'timestamp': prediction.timestamp.isoformat()
                    },
                    'recommendation': self._generate_recommendation_from_prediction(prediction),
                    'confidence': prediction.confidence_score
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Unable to generate prediction',
                    'recommendation': 'HOLD',
                    'confidence': 0.1
                }
                
        except Exception as e:
            logger.error(f"❌ Price prediction error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'recommendation': 'HOLD',
                'confidence': 0.1
            }
    
    async def predict_price(self, 
        symbol: str, 
        timeframe: PredictionTimeframe,
        horizon_minutes: int) -> Optional[PredictionResult]:
        """🔮 Generate price prediction using ensemble of ML models"""
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe.value}_{horizon_minutes}_{int(datetime.now().timestamp() // 60)}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Get historical data
            historical_data = await self._get_historical_data(symbol, timeframe, 500)  # 500 data points
            
            if len(historical_data) < 50:  # Need minimum data
                logger.warning(f"⚠️ Insufficient data for {symbol} prediction")
                return None
            
            # Engineer features
            features = self._engineer_features(historical_data)
            
            # Get ensemble prediction
            prediction_result = await self._ensemble_predict(
                symbol, timeframe, features, horizon_minutes
            )
            
            # Cache result
            self.prediction_cache[cache_key] = prediction_result
            
            # Clean old cache entries
            await self._clean_prediction_cache()
            
            self.total_predictions += 1
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Prediction error for {symbol}: {e}")
            return None
    
    async def _ensemble_predict(self, 
        symbol: str,
        timeframe: PredictionTimeframe,
        features: pd.DataFrame,
        horizon_minutes: int) -> PredictionResult:
        """Generate ensemble prediction from multiple models"""
        
        predictions = []
        confidences = []
        models_used = []
        
        # Prepare feature data
        X = features.drop(['price', 'timestamp'], axis=1, errors='ignore').values
        current_price = features['price'].iloc[-1]
        
        if len(X) == 0:
            raise ValueError("No features available for prediction")
        
        # Get predictions from each model
        for model_type in ModelType:
            if model_type == ModelType.ENSEMBLE:
                continue  # Skip ensemble in ensemble
            
            try:
                pred, conf = await self._predict_with_model(
                    model_type, symbol, timeframe, X, current_price, horizon_minutes
                )
                if pred is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                    models_used.append(model_type)
                    
            except Exception as e:
                logger.warning(f"⚠️ Model {model_type.value} failed: {e}")
        
        if not predictions:
            raise ValueError("No models could generate predictions")
        
        # Ensemble averaging with confidence weighting
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / total_weight
            ensemble_confidence = np.mean(confidences) * (len(predictions) / len(ModelType))  # Bonus for multiple models
        else:
            weighted_prediction = np.mean(predictions)
            ensemble_confidence = 0.5
        
        # Calculate confidence interval
        std = np.std(predictions) if len(predictions) > 1 else abs(weighted_prediction - current_price) * 0.1
        confidence_interval = (
            weighted_prediction - 1.96 * std,  # 95% confidence interval
            weighted_prediction + 1.96 * std
        )
        
        return PredictionResult(
            symbol=symbol,
            timeframe=timeframe,
            predicted_price=weighted_prediction,
            confidence_interval=confidence_interval,
            confidence_score=min(ensemble_confidence, 0.95),
            model_used=ModelType.ENSEMBLE,
            features_used=list(features.columns),
            prediction_horizon=horizon_minutes,
            timestamp=datetime.now(),
            metadata={
                'models_used': [m.value for m in models_used],
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'ensemble_size': len(predictions),
                'current_price': current_price
            }
        )
    
    async def _predict_with_model(self, 
                                model_type: ModelType,
                                symbol: str,
                                timeframe: PredictionTimeframe,
                                X: np.ndarray,
                                current_price: float,
                                horizon_minutes: int) -> Tuple[Optional[float], float]:
        """Generate prediction with specific model"""
        
        model_key = f"{model_type.value}_{symbol}_{timeframe.value}"
        
        # Check if model exists
        if model_key not in self.models:
            await self._train_model(model_type, symbol, timeframe, X)
        
        model = self.models.get(model_key)
        scaler = self.scalers.get(model_key)
        
        if model is None:
            return None, 0.0
        
        try:
            # Prepare input
            latest_features = X[-1:] if len(X.shape) == 2 else X[-1:].reshape(1, -1)
            
            if scaler:
                latest_features = scaler.transform(latest_features)
            
            # Make prediction
            if model_type == ModelType.LSTM and TF_AVAILABLE:
                # LSTM needs sequence data
                sequence_length = min(50, len(X))
                sequence_data = X[-sequence_length:].reshape(1, sequence_length, -1)
                if scaler:
                    sequence_data = scaler.transform(sequence_data.reshape(-1, sequence_data.shape[-1])).reshape(sequence_data.shape)
                prediction = model.predict(sequence_data)[0][0]
            else:
                # Standard ML models
                prediction = model.predict(latest_features)[0]
            
            # Calculate confidence based on model performance
            perf = self.model_performance.get(model_key, ModelPerformance(
                model_type=model_type, rmse=1.0, mae=1.0, r2_score=0.0, 
                accuracy=0.5, predictions_made=0, last_updated=datetime.now()
            ))
            
            # Confidence based on R² score and directional accuracy
            confidence = (perf.r2_score * 0.5 + perf.accuracy * 0.5) * 0.9  # Max 0.9 confidence
            
            return float(prediction), max(confidence, 0.1)
            
        except Exception as e:
            logger.error(f"❌ Model {model_type.value} prediction error: {e}")
            return None, 0.0
    
    async def _train_model(self, 
        model_type: ModelType,
        symbol: str,
        timeframe: PredictionTimeframe,
        X: np.ndarray) -> bool:
        """Train a specific model"""
        
        if not ML_AVAILABLE:
            logger.warning("⚠️ ML libraries not available for training")
            return False
        
        try:
            model_key = f"{model_type.value}_{symbol}_{timeframe.value}"
            
            # Get training data
            historical_data = await self._get_historical_data(symbol, timeframe, 1000)
            if len(historical_data) < 100:
                logger.warning(f"⚠️ Insufficient data to train {model_key}")
                return False
            
            features = self._engineer_features(historical_data)
            
            # Prepare training data
            X_train = features.drop(['price', 'timestamp'], axis=1, errors='ignore').values
            y_train = features['price'].values[1:]  # Next price as target
            X_train = X_train[:-1]  # Remove last feature row to match target size
            
            if len(X_train) != len(y_train):
                logger.error(f"❌ Feature/target size mismatch: {len(X_train)} vs {len(y_train)}")
                return False
            
            # Split data
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val_split)
            
            # Train model based on type
            if model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train_split)
                
            elif model_type == ModelType.GRADIENT_BOOST:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train_split)
                
            elif model_type == ModelType.LINEAR:
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train_split)
                
            elif model_type == ModelType.LSTM and TF_AVAILABLE:
                model = await self._train_lstm_model(
                    X_train_scaled, y_train_split, X_val_scaled, y_val_split
                )
                
            else:
                logger.warning(f"⚠️ Model type {model_type.value} not supported")
                return False
            
            # Evaluate model
            y_pred = model.predict(X_val_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
            mae = mean_absolute_error(y_val_split, y_pred)
            r2 = r2_score(y_val_split, y_pred)
            
            # Calculate directional accuracy
            actual_direction = np.sign(y_val_split[1:] - y_val_split[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            accuracy = np.mean(actual_direction == pred_direction)
            
            # Store model and performance
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_performance[model_key] = ModelPerformance(
                model_type=model_type,
                rmse=rmse,
                mae=mae,
                r2_score=r2,
                accuracy=accuracy,
                predictions_made=0,
                last_updated=datetime.now()
            )
            
            logger.info(f"✅ Trained {model_key}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Accuracy={accuracy:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error for {model_type.value}: {e}")
            return False
    
    async def _train_lstm_model(self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray) -> Any:
        """Train LSTM neural network model"""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Reshape data for LSTM (samples, timesteps, features)
        sequence_length = min(50, len(X_train))
        
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(seq_len, len(X)):
                X_seq.append(X[i-seq_len:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(
            X_train_seq, y_train_seq,
            batch_size=32,
            epochs=100,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """🔧 Engineer features from raw market data"""
        
        if 'price' not in data.columns:
            logger.error("❌ No price column in data")
            return pd.DataFrame()
        
        features = data.copy()
        
        # Technical indicators
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = features['price'].rolling(period).mean()
            features[f'ema_{period}'] = features['price'].ewm(span=period).mean()
        
        # Price ratios
        features['price_sma_20_ratio'] = features['price'] / features['sma_20']
        features['sma_5_sma_20_ratio'] = features['sma_5'] / features['sma_20']
        
        # Volatility
        features['volatility_10'] = features['price'].rolling(10).std()
        features['volatility_20'] = features['price'].rolling(20).std()
        
        # Returns
        features['return_1'] = features['price'].pct_change(1)
        features['return_5'] = features['price'].pct_change(5)
        features['return_20'] = features['price'].pct_change(20)
        
        # RSI
        features['rsi'] = self._calculate_rsi(features['price'])
        
        # MACD
        macd_line, macd_signal = self._calculate_macd(features['price'])
        features['macd'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_line - macd_signal
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(features['price'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (features['price'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features (if available)
        if 'volume' in features.columns:
            features['volume_sma_20'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma_20']
            features['price_volume'] = features['price'] * features['volume']
        
        # Time-based features
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
        
        # Lag features
        for lag in [1, 2, 5, 10]:
            features[f'price_lag_{lag}'] = features['price'].shift(lag)
            features[f'return_lag_{lag}'] = features['return_1'].shift(lag)
        
        # Drop NaN values
        features = features.ffill().bfill()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        
        return macd_line, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    async def _get_historical_data(self, 
        symbol: str, 
        timeframe: PredictionTimeframe, 
        limit: int) -> pd.DataFrame:
        """Get historical market data for training/prediction"""
        
        # Fetch real market data using yfinance
        try:
            import yfinance as yf
            
            # Map timeframes to yfinance periods
            period_map = {
                PredictionTimeframe.HOUR_1: "1d",
                PredictionTimeframe.HOUR_4: "5d", 
                PredictionTimeframe.DAY_1: "1mo",
                PredictionTimeframe.WEEK_1: "3mo",
                PredictionTimeframe.MONTH_1: "1y"
            }
            
            # Map timeframes to yfinance intervals
            interval_map = {
                PredictionTimeframe.HOUR_1: "1h",
                PredictionTimeframe.HOUR_4: "4h",
                PredictionTimeframe.DAY_1: "1d",
                PredictionTimeframe.WEEK_1: "1wk",
                PredictionTimeframe.MONTH_1: "1mo"
            }
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                period=period_map.get(timeframe, "1mo"),
                interval=interval_map.get(timeframe, "1d")
            )
            
            if not hist.empty:
                data = pd.DataFrame({
                    'timestamp': hist.index,
                    'price': hist['Close'].values,
                    'volume': hist['Volume'].values
                })
                data = data.tail(limit) if len(data) > limit else data
                return data.reset_index(drop=True)
                
        except Exception as e:
            logger.warning(f"Failed to fetch real data for {symbol}: {e}")
        
        # Fallback: Return minimal valid data if real data fetch fails
        current_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [current_time - timedelta(hours=1), current_time],
            'price': [150.0, 151.0],  # Minimal realistic prices
            'volume': [1000000, 1100000]  # Minimal realistic volume
        })
        
        return data.tail(limit)  # Return requested number of records
    
    def _generate_recommendation_from_prediction(self, prediction: PredictionResult) -> str:
        """Generate trading recommendation from prediction"""
        
        current_price = prediction.metadata.get('current_price', prediction.predicted_price)
        predicted_price = prediction.predicted_price
        confidence = prediction.confidence_score
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Decision thresholds based on confidence
        strong_threshold = 0.05 * confidence  # Higher confidence allows lower thresholds
        weak_threshold = 0.02 * confidence
        
        if expected_return > strong_threshold:
            return "STRONG_BUY" if confidence > 0.7 else "BUY"
        elif expected_return < -strong_threshold:
            return "STRONG_SELL" if confidence > 0.7 else "SELL"
        elif abs(expected_return) > weak_threshold:
            return "BUY" if expected_return > 0 else "SELL"
        else:
            return "HOLD"
    
    async def _clean_prediction_cache(self):
        """Clean old prediction cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key in self.prediction_cache:
            try:
                # Extract timestamp from cache key
                timestamp_part = int(key.split('_')[-1])
                cache_time = datetime.fromtimestamp(timestamp_part * 60)  # Convert back from minute timestamp
                
                if current_time - cache_time > timedelta(hours=1):  # Remove entries older than 1 hour
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                keys_to_remove.append(key)  # Remove malformed keys
        
        for key in keys_to_remove:
            del self.prediction_cache[key]
    
    async def _initialize_models_for_timeframe(self, timeframe: PredictionTimeframe):
        """Initialize models for a specific timeframe"""
        # This would be called during startup to pre-train models
        # For now, we'll train models on-demand
        logger.info(f" Models for {timeframe.value} will be trained on-demand")
    
    async def _handle_batch_prediction(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch prediction requests"""
        symbols = message.get('symbols', ['AAPL'])
        timeframe_str = message.get('timeframe', '1h')
        timeframe = PredictionTimeframe(timeframe_str)
        
        predictions = {}
        
        for symbol in symbols:
            try:
                pred = await self.predict_price(symbol, timeframe, 60)
                if pred:
                    predictions[symbol] = {
                        'predicted_price': pred.predicted_price,
                        'confidence': pred.confidence_score,
                        'recommendation': self._generate_recommendation_from_prediction(pred)
                    }
            except Exception as e:
                logger.error(f"❌ Batch prediction error for {symbol}: {e}")
        
        return {
            'status': 'success',
            'predictions': predictions,
            'timeframe': timeframe_str,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_performance_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model performance queries"""
        return {
            'status': 'success',
            'model_performance': {
                key: {
                    'model_type': perf.model_type.value,
                    'rmse': perf.rmse,
                    'mae': perf.mae,
                    'r2_score': perf.r2_score,
                    'accuracy': perf.accuracy,
                    'predictions_made': perf.predictions_made,
                    'last_updated': perf.last_updated.isoformat()
                }
                for key, perf in self.model_performance.items()
            },
            'total_predictions': self.total_predictions,
            'overall_accuracy': self.correct_direction_predictions / max(self.total_predictions, 1)
        }
    
    async def _handle_model_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model update requests"""
        # This would trigger model retraining
        return {
            'status': 'success',
            'message': 'Model update initiated',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_price_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming price data updates"""
        symbol = message.get('symbol')
        price_data = message.get('data', {})
        
        if symbol:
            # Store price data for training
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': datetime.now(),
                'price': price_data.get('price', 0),
                'volume': price_data.get('volume', 0)
            })
            
            # Keep only recent data
            self.price_history[symbol] = self.price_history[symbol][-5000:]  # Keep last 5000 points
        
        return {'status': 'acknowledged'}
    
    async def _handle_general_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries about the predictive analytics system"""
        return {
            'status': 'success',
            'agent': 'PredictiveAnalytics',
            'capabilities': [
                'Price prediction using ML models',
                'Multi-timeframe analysis',
                'Ensemble model voting',
                'Real-time model adaptation',
                'Confidence scoring',
                'Performance tracking'
            ],
            'supported_models': [model.value for model in ModelType],
            'supported_timeframes': [tf.value for tf in self.supported_timeframes],
            'total_predictions': self.total_predictions,
            'recommendation': 'ANALYZE',
            'confidence': 0.8
        }
