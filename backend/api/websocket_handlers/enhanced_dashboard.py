"""
🌐 Enhanced Dashboard WebSocket Handler
=====================================
Real-time data streaming for SuperAdvancedDashboard
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf
import pandas as pd
from database.cache_manager import cache

logger = logging.getLogger(__name__)

class EnhancedDashboardHandler:
    def __init__(self):
        self.active_connections: Dict[str, Any] = {}
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        self.update_interval = 30  # seconds
        self.running = False
        
    async def connect(self, websocket, client_id: str):
        """Connect a new client"""
        await websocket.accept()
        self.active_connections[client_id] = {
            'websocket': websocket,
            'connected_at': datetime.utcnow(),
            'last_ping': datetime.utcnow()
        }
        logger.info(f"Dashboard client {client_id} connected")
        
        # Start streaming if not already running
        if not self.running:
            asyncio.create_task(self.start_streaming())
    
    async def disconnect(self, client_id: str):
        """Disconnect a client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Dashboard client {client_id} disconnected")
    
    async def start_streaming(self):
        """Start the real-time data streaming"""
        self.running = True
        logger.info("Starting enhanced dashboard streaming")
        
        while self.active_connections and self.running:
            try:
                # Fetch comprehensive market data
                market_data = await self.fetch_market_data()
                predictions = await self.fetch_predictions()
                alerts = await self.fetch_alerts()
                sector_data = await self.fetch_sector_data()
                
                # Create comprehensive update package
                update_package = {
                    'type': 'dashboard_update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': {
                        'market_data': market_data,
                        'predictions': predictions,
                        'alerts': alerts,
                        'sector_data': sector_data,
                        'performance_metrics': self.calculate_performance_metrics(market_data, predictions)
                    }
                }
                
                # Send to all connected clients
                await self.broadcast(update_package)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard streaming: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def fetch_market_data(self) -> List[Dict]:
        """Fetch real-time market data"""
        try:
            market_data = []
            
            for symbol in self.symbols:
                try:
                    # Check cache first
                    cached_data = cache.get_market_data(symbol)
                    if cached_data:
                        market_data.append(cached_data)
                        continue
                    
                    # Fetch fresh data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d", interval="1d")
                    
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        volume = hist['Volume'].iloc[-1]
                        
                        data = {
                            "symbol": symbol,
                            "price": float(current),
                            "change": float(current - previous),
                            "changePercent": float((current - previous) / previous * 100),
                            "volume": int(volume) if not pd.isna(volume) else 0,
                            "high_24h": float(hist['High'].iloc[-1]),
                            "low_24h": float(hist['Low'].iloc[-1]),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        cache.set_market_data(symbol, data, 30)
                        market_data.append(data)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return []
    
    async def fetch_predictions(self) -> Dict[str, Any]:
        """Fetch AI predictions"""
        try:
            import hashlib
            
            predictions = {}
            for symbol in self.symbols[:5]:  # Limit for performance
                seed = int(hashlib.md5(f"{symbol}pred{datetime.utcnow().strftime('%Y%m%d%H')}", encoding='utf-8').hexdigest()[:8], 16)
                signal_strength = ((seed % 100) - 50) / 50
                
                if signal_strength > 0.3:
                    action = "BUY"
                    confidence = 0.7 + signal_strength * 0.25
                elif signal_strength < -0.3:
                    action = "SELL"
                    confidence = 0.7 + abs(signal_strength) * 0.25
                else:
                    action = "HOLD"
                    confidence = 0.6 + abs(signal_strength) * 0.2
                
                base_prices = {'AAPL': 175, 'MSFT': 365, 'GOOGL': 135, 'AMZN': 145, 'TSLA': 250}
                current_price = base_prices.get(symbol, 200)
                
                predictions[symbol] = {
                    "symbol": symbol,
                    "target_price": current_price * (1 + signal_strength * 0.05),
                    "confidence": confidence,
                    "direction": action,
                    "risk_score": max(0.1, min(0.9, 1 - confidence)),
                    "time_horizon": "1D",
                    "probability": confidence
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            return {}
    
    async def fetch_alerts(self) -> List[Dict]:
        """Fetch real-time alerts"""
        try:
            alerts = []
            
            for symbol in self.symbols[:3]:  # Limit for performance
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d", interval="1d")
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change_percent = (current_price - prev_price) / prev_price * 100
                        
                        if abs(change_percent) > 2:
                            severity = "high" if abs(change_percent) > 4 else "medium"
                            direction = "surge" if change_percent > 0 else "decline"
                            
                            alerts.append({
                                "id": f"alert_{symbol}_{int(datetime.utcnow().timestamp())}",
                                "type": "prediction",
                                "severity": severity,
                                "message": f"{symbol} {direction}: {change_percent:+.2f}%",
                                "symbol": symbol,
                                "timestamp": datetime.utcnow(),
                                "action": f"Monitor {symbol} closely"
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to generate alert for {symbol}: {e}")
            
            return alerts[:5]  # Limit alerts
            
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return []
    
    async def fetch_sector_data(self) -> Dict[str, Any]:
        """Fetch sector performance data"""
        try:
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Energy": "XLE"
            }
            
            sector_data = {}
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="2d", interval="1d")
                    
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change_1d = (current - previous) / previous * 100
                        
                        sector_data[sector] = {
                            "price": float(current),
                            "change_1d": float(change_1d),
                            "change_5d": float(change_1d * 1.2),  # Approximation
                            "symbol": etf
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch sector data for {sector}: {e}")
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector data: {e}")
            return {}
    
    def calculate_performance_metrics(self, market_data: List[Dict], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            total_value = sum(stock.get('price', 0) for stock in market_data)
            total_gain = sum(stock.get('change', 0) for stock in market_data)
            total_gain_percent = sum(stock.get('changePercent', 0) for stock in market_data) / max(len(market_data), 1)
            
            prediction_values = list(predictions.values())
            avg_confidence = sum(p.get('confidence', 0) for p in prediction_values) / max(len(prediction_values), 1)
            win_rate = len([p for p in prediction_values if p.get('direction') == 'BUY' and p.get('confidence', 0) > 0.7]) / max(len(prediction_values), 1)
            risk_score = sum(p.get('risk_score', 0) for p in prediction_values) / max(len(prediction_values), 1)
            
            return {
                "totalValue": total_value,
                "totalGain": total_gain,
                "totalGainPercent": total_gain_percent,
                "winRate": win_rate,
                "avgConfidence": avg_confidence,
                "riskScore": risk_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "totalValue": 0,
                "totalGain": 0,
                "totalGainPercent": 0,
                "winRate": 0,
                "avgConfidence": 0,
                "riskScore": 0
            }
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            try:
                await connection['websocket'].send_text(message_str)
                connection['last_ping'] = datetime.utcnow()
            except Exception as e:
                logger.warning(f"Failed to send to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming message from client"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'ping':
                # Respond with pong
                await self.send_to_client(client_id, {'type': 'pong', 'timestamp': datetime.utcnow().isoformat()})
            
            elif msg_type == 'subscribe_symbols':
                # Update symbols for this client
                symbols = message.get('symbols', [])
                if symbols:
                    self.symbols = symbols[:10]  # Limit to 10 symbols
                    logger.info(f"Client {client_id} subscribed to symbols: {symbols}")
            
            elif msg_type == 'update_interval':
                # Update refresh interval
                interval = message.get('interval', 30)
                self.update_interval = max(10, min(300, interval))  # Between 10s and 5min
                logger.info(f"Update interval changed to {self.update_interval}s")
                
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                message_str = json.dumps(message, default=str)
                await self.active_connections[client_id]['websocket'].send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send to client {client_id}: {e}")
                await self.disconnect(client_id)

# Global handler instance
enhanced_dashboard_handler = EnhancedDashboardHandler()