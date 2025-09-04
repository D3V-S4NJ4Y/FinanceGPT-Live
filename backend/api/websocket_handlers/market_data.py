"""
📈 Market Data WebSocket Handler
==============================
Real-time market data provider for FinanceGPT Live via WebSockets
"""

import asyncio
import json
import logging
import random
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta

# Import shared types
from api.websocket import WebSocketManager

# Configure logging
logger = logging.getLogger(__name__)

class MarketDataHandler:
    """
    Handler for real-time market data WebSocket messages
    
    Features:
    - Symbol subscription management
    - Real-time market data streaming
    - Market indices updates
    - Trading volume updates
    - Technical indicators streaming
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.active_subscriptions: Dict[str, Dict[str, Any]] = {}
        self.default_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "JPM", "V", "PYPL"]
        self.indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
        self.last_market_update = datetime.utcnow()
        self.last_full_update = datetime.utcnow() - timedelta(minutes=5)  # Force initial update
        self.market_cache = {}
        self.update_task = None
        
    async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """Process a message from a WebSocket client"""
        try:
            if not message or "type" not in message:
                logger.warning(f"Invalid message format from client {client_id}")
                return
                
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                # Handle symbol subscription
                symbols = message.get("symbols", [])
                if not symbols:
                    symbols = self.default_symbols.copy()
                    
                await self.subscribe_client(client_id, symbols)
                
            elif msg_type == "unsubscribe":
                # Handle symbol unsubscription
                symbols = message.get("symbols", [])
                if not symbols:
                    # Unsubscribe from all
                    if client_id in self.active_subscriptions:
                        del self.active_subscriptions[client_id]
                else:
                    await self.unsubscribe_client(client_id, symbols)
                    
            elif msg_type == "request":
                # Handle specific data requests
                data_type = message.get("data")
                if data_type == "market_snapshot":
                    await self.send_market_snapshot(client_id, message.get("symbols", []))
                elif data_type == "indices":
                    await self.send_indices_data(client_id)
                elif data_type == "stock_details":
                    symbol = message.get("symbol")
                    if symbol:
                        await self.send_stock_details(client_id, symbol)
                else:
                    logger.warning(f"Unknown data request type: {data_type}")
                    
            else:
                logger.warning(f"Unknown message type from client {client_id}: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            
    async def subscribe_client(self, client_id: str, symbols: List[str]):
        """Subscribe a client to specific symbols"""
        if client_id not in self.active_subscriptions:
            self.active_subscriptions[client_id] = {
                "symbols": set(symbols),
                "last_update": datetime.utcnow()
            }
        else:
            self.active_subscriptions[client_id]["symbols"].update(symbols)
            self.active_subscriptions[client_id]["last_update"] = datetime.utcnow()
            
        # Make a copy to avoid modification issues
        current_symbols = list(self.active_subscriptions[client_id]["symbols"])
            
        # Send confirmation
        await self.websocket_manager.send_personal_message(client_id, {
            "type": "subscription_confirmed",
            "data": {
                "symbols": current_symbols,
                "message": f"Subscribed to {len(current_symbols)} symbols"
            }
        })
        
        # Send initial data
        await self.send_market_snapshot(client_id, current_symbols)
        
    async def unsubscribe_client(self, client_id: str, symbols: List[str]):
        """Unsubscribe a client from specific symbols"""
        if client_id in self.active_subscriptions:
            for symbol in symbols:
                if symbol in self.active_subscriptions[client_id]["symbols"]:
                    self.active_subscriptions[client_id]["symbols"].remove(symbol)
                    
            # Update timestamp
            self.active_subscriptions[client_id]["last_update"] = datetime.utcnow()
            
            # Make a copy to avoid modification issues
            current_symbols = list(self.active_subscriptions[client_id]["symbols"])
            
            # Send confirmation
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "unsubscription_confirmed",
                "data": {
                    "symbols": current_symbols,
                    "message": f"Unsubscribed from specified symbols. Remaining: {len(current_symbols)}"
                }
            })
            
            # If no symbols left, remove subscription
            if not self.active_subscriptions[client_id]["symbols"]:
                del self.active_subscriptions[client_id]
                
    async def send_market_snapshot(self, client_id: str, symbols: List[str]):
        """Send current market data for requested symbols"""
        try:
            # If no symbols specified, use subscribed or default
            if not symbols:
                if client_id in self.active_subscriptions:
                    symbols = list(self.active_subscriptions[client_id]["symbols"])
                else:
                    symbols = self.default_symbols.copy()
            
            # Import the market_data module for direct access to functions
            from api.routes.market_data import get_realtime_batch
            
            # Get current market data
            market_data = await get_realtime_batch(symbols)
            
            # Format data for frontend
            stocks_data = []
            for symbol, data in market_data.items():
                stocks_data.append({
                    "symbol": symbol,
                    "price": data.get("price", 0),
                    "change": data.get("change", 0),
                    "changePercent": data.get("change_percent", 0),
                    "volume": data.get("volume", 0),
                    "lastUpdate": data.get("last_update", datetime.utcnow().isoformat())
                })
                
            # Send to client
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "market_snapshot",
                "data": {
                    "stocks": stocks_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error sending market snapshot: {e}")
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "error",
                "message": "Failed to get market data",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    async def send_indices_data(self, client_id: str):
        """Send market indices data to client"""
        try:
            # Import the market_data module for direct access to functions
            from api.routes.market_data import get_realtime_batch
            
            # Get indices data
            indices_data = await get_realtime_batch(self.indices)
            
            # Format data for frontend
            formatted_indices = []
            for symbol, data in indices_data.items():
                formatted_indices.append({
                    "symbol": symbol,
                    "name": self.get_index_name(symbol),
                    "price": data.get("price", 0),
                    "change": data.get("change", 0),
                    "changePercent": data.get("change_percent", 0),
                    "lastUpdate": data.get("last_update", datetime.utcnow().isoformat())
                })
                
            # Send to client
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "indices_data",
                "data": {
                    "indices": formatted_indices,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error sending indices data: {e}")
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "error",
                "message": "Failed to get indices data",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    async def send_stock_details(self, client_id: str, symbol: str):
        """Send detailed stock information to client"""
        try:
            # Import the market_data module for direct access to functions
            from api.routes.market_data import get_stock_details
            
            # Get stock details
            stock_details = await get_stock_details(symbol)
            
            # Send to client
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "stock_details",
                "data": stock_details,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error sending stock details: {e}")
            await self.websocket_manager.send_personal_message(client_id, {
                "type": "error",
                "message": f"Failed to get details for {symbol}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    def get_index_name(self, symbol: str) -> str:
        """Get friendly name for market index"""
        index_names = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^RUT": "Russell 2000",
            "^VIX": "VIX"
        }
        return index_names.get(symbol, symbol)
        
    async def start_update_task(self):
        """Start the periodic update task"""
        if self.update_task is None:
            self.update_task = asyncio.create_task(self.run_periodic_updates())
            logger.info("Started market data update task")
            
    async def run_periodic_updates(self):
        """Run periodic updates for all clients"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Send updates to all subscribed clients
                await self.broadcast_market_updates()
                
                # Check if full refresh needed (every 5 minutes)
                if (current_time - self.last_full_update).total_seconds() > 300:
                    await self.refresh_market_cache()
                    self.last_full_update = current_time
                    
                # Clean up stale subscriptions (no activity for 10 minutes)
                stale_time = current_time - timedelta(minutes=10)
                stale_clients = []
                
                for client_id, subscription in self.active_subscriptions.items():
                    if subscription["last_update"] < stale_time:
                        stale_clients.append(client_id)
                        
                for client_id in stale_clients:
                    logger.info(f"Removing stale subscription for client {client_id}")
                    if client_id in self.active_subscriptions:
                        del self.active_subscriptions[client_id]
                    
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                
            # Wait before next update
            await asyncio.sleep(5)
            
    async def refresh_market_cache(self):
        """Refresh the market data cache with latest data"""
        try:
            # Get all unique symbols from subscriptions
            all_symbols = set()
            for client_id, subscription in self.active_subscriptions.items():
                all_symbols.update(subscription["symbols"])
                
            # Add indices
            all_symbols.update(self.indices)
            
            # If no symbols, use defaults
            if not all_symbols:
                all_symbols = set(self.default_symbols + self.indices)
                
            # Import the market_data module for direct access to functions
            from api.routes.market_data import get_realtime_batch
            
            # Get market data
            self.market_cache = await get_realtime_batch(list(all_symbols))
            self.last_market_update = datetime.utcnow()
            
            logger.info(f"Refreshed market cache with {len(self.market_cache)} symbols")
            
        except Exception as e:
            logger.error(f"Error refreshing market cache: {e}")
            
    async def broadcast_market_updates(self):
        """Broadcast market updates to all subscribed clients"""
        try:
            current_time = datetime.utcnow()
            
            # Only update if we have data and if it's time for an update
            if not self.market_cache or (current_time - self.last_market_update).total_seconds() < 5:
                return
                
            # Send updates to each client based on their subscriptions
            for client_id, subscription in list(self.active_subscriptions.items()):
                client_symbols = subscription["symbols"]
                if not client_symbols:
                    continue
                    
                # Format data for this client
                stocks_data = []
                for symbol in client_symbols:
                    if symbol in self.market_cache:
                        data = self.market_cache[symbol]
                        stocks_data.append({
                            "symbol": symbol,
                            "price": data.get("price", 0),
                            "change": data.get("change", 0),
                            "changePercent": data.get("change_percent", 0),
                            "volume": data.get("volume", 0),
                            "lastUpdate": data.get("last_update", datetime.utcnow().isoformat())
                        })
                        
                if stocks_data:
                    # Send update to client
                    await self.websocket_manager.send_personal_message(client_id, {
                        "type": "market_update",
                        "data": {
                            "stocks": stocks_data,
                            "timestamp": current_time.isoformat()
                        }
                    })
                    
        except Exception as e:
            logger.error(f"Error broadcasting market updates: {e}")
            
    async def handle_connection_closed(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_subscriptions:
            del self.active_subscriptions[client_id]
            logger.info(f"Removed market data subscription for disconnected client {client_id}")
