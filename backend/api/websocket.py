import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Individual WebSocket connection manager"""
    
    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.subscriptions: Set[str] = set()
        self.last_ping = datetime.utcnow()
        self.is_active = True
        
    async def send_personal_message(self, message: dict):
        """Send message to this specific connection with comprehensive error handling"""
        if not self.is_active:
            return False
            
        try:
            # More robust connection state check
            from fastapi.websockets import WebSocketState
            if self.websocket.client_state != WebSocketState.CONNECTED:
                # Suppress debug logs for normal disconnections
                self.is_active = False
                return False
                
            await self.websocket.send_text(json.dumps(message))
            return True
        except ConnectionClosedError:
            # Normal disconnection - suppress log
            self.is_active = False
            return False
        except RuntimeError as e:
            if "close message has been sent" in str(e).lower():
                # Expected error when client disconnects - suppress log
                self.is_active = False
                return False
            # Only log unexpected runtime errors at debug level
            logger.debug(f"❌ Runtime error sending message to {self.client_id}: {e}")
            self.is_active = False
            return False
        except Exception as e:
            # Log other unexpected errors but at debug level to reduce noise
            logger.debug(f"❌ Failed to send message to {self.client_id}: {e}")
            self.is_active = False
            return False
            
    async def ping(self):
        """Send ping to check connection health"""
        try:
            await self.websocket.send_text(json.dumps({
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            }))
            self.last_ping = datetime.utcnow()
            return True
        except:
            self.is_active = False
            return False

class WebSocketManager:
    
    def __init__(self, max_connections: int = 100):
        # Active connections by client ID
        self.active_connections: Dict[str, ConnectionManager] = {}
        
        # Topic subscriptions: topic -> set of client IDs
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Message queue for offline clients
        self.message_queue: Dict[str, List[dict]] = {}
        
        # Connection limiting
        self.max_connections = max_connections
        
        # Connection statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "failed_sends": 0,
            "rejected_connections": 0,
            "last_cleanup": datetime.utcnow().isoformat(),
            "startup_time": datetime.utcnow().isoformat()
        }
        
        # Initialize background task references
        self._health_check_task = None
        self._cleanup_task = None
        
        # Initialize handlers registry
        self.handlers = {}
        
    def register_handler(self, handler_name: str, handler_instance):
        """Register a message handler for a specific type of messages"""
        self.handlers[handler_name] = handler_instance
        logger.info(f"✅ Registered handler for '{handler_name}'")
        
    async def handle_client_message(self, client_id: str, message: dict):
        """Route client message to appropriate handler based on message type"""
        try:
            if not isinstance(message, dict):
                logger.warning(f"❌ Invalid message format from client {client_id}")
                return
                
            handler_name = message.get('handler')
            if not handler_name:
                logger.warning(f"❌ No handler specified in message from client {client_id}")
                return
                
            handler = self.handlers.get(handler_name)
            if not handler:
                logger.warning(f"❌ Unknown handler '{handler_name}' requested by client {client_id}")
                return
                
            # Call the handler's handle_client_message method
            await handler.handle_client_message(client_id, message)
            
        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")
        
    async def start_background_tasks(self):
        """Start background tasks when event loop is available"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept new WebSocket connection with connection limiting"""
        
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            logger.warning(f"⚠️ Connection limit reached ({len(self.active_connections)}/{self.max_connections}). Rejecting connection.")
            self.stats["rejected_connections"] += 1
            await websocket.close(code=1013, reason="Server overloaded - too many connections")
            raise Exception("Connection limit exceeded")
        
        # Check if client already exists (prevent duplicates)
        if client_id and client_id in self.active_connections:
            # Check if the existing connection is still active
            connection = self.active_connections[client_id]
            try:
                from fastapi.websockets import WebSocketState
                if connection.is_active and connection.websocket.client_state == WebSocketState.CONNECTED:
                    logger.info(f" Client {client_id} already connected. Using existing connection.")
                    return client_id
                else:
                    # Connection exists but is not active, clean it up
                    logger.info(f" Client {client_id} has stale connection. Replacing it.")
                    await self.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error checking existing connection for {client_id}: {e}")
                await self.disconnect(client_id)
            
        # Note: Do not call accept() here as it should be called in the endpoint first
        
        # Small delay to ensure WebSocket is fully established
        await asyncio.sleep(0.05)
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
            
        # Create connection manager
        connection = ConnectionManager(websocket, client_id)
        self.active_connections[client_id] = connection
        
        # Update stats
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.active_connections)
        
        # Send welcome message
        await connection.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f" Connected to FinanceGPT Live! ({len(self.active_connections)}/{self.max_connections})"
        })
        
        # Send queued messages if any
        if client_id in self.message_queue:
            for message in self.message_queue[client_id]:
                await connection.send_personal_message(message)
            del self.message_queue[client_id]
            
        logger.info(f"✅ Client {client_id} connected. Total: {len(self.active_connections)}")
        return client_id
        
    async def disconnect(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_connections:
            # Remove from all subscriptions
            for topic in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[topic]:
                    self.subscriptions[topic].discard(client_id)
                    if not self.subscriptions[topic]:
                        del self.subscriptions[topic]
                        
            # Remove connection
            del self.active_connections[client_id]
            self.stats["active_connections"] = len(self.active_connections)
            
            logger.info(f" Client {client_id} disconnected. Total: {len(self.active_connections)}")
            
    async def subscribe(self, client_id: str, topic: str):
        """Subscribe client to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
            
        self.subscriptions[topic].add(client_id)
        
        # Send confirmation
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_personal_message({
                "type": "subscription_confirmed",
                "topic": topic,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        logger.info(f" Client {client_id} subscribed to {topic}")
        
    async def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe client from a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
                
        logger.info(f" Client {client_id} unsubscribed from {topic}")
        
    async def send_personal_message(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            if connection.is_active:
                success = await connection.send_personal_message(message)
                if success:
                    self.stats["messages_sent"] += 1
                else:
                    # Connection failed, disconnect
                    await self.disconnect(client_id)
                    self.stats["failed_sends"] += 1
            else:
                # Connection inactive, remove it
                await self.disconnect(client_id)
        else:
            # Queue message for offline client
            if client_id not in self.message_queue:
                self.message_queue[client_id] = []
            self.message_queue[client_id].append(message)
            
    async def broadcast_to_topic(self, topic: str, message: dict):
        """Broadcast message to all subscribers of a topic"""
        if topic not in self.subscriptions:
            return
            
        message["timestamp"] = datetime.utcnow().isoformat()
        disconnected_clients = []
        
        for client_id in list(self.subscriptions[topic]):
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                if connection.is_active:
                    success = await connection.send_personal_message(message)
                    if success:
                        self.stats["messages_sent"] += 1
                    else:
                        disconnected_clients.append(client_id)
                        self.stats["failed_sends"] += 1
                else:
                    disconnected_clients.append(client_id)
                    
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
            
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        message["timestamp"] = datetime.utcnow().isoformat()
        disconnected_clients = []
        
        for client_id, connection in list(self.active_connections.items()):
            if connection.is_active:
                success = await connection.send_personal_message(message)
                if success:
                    self.stats["messages_sent"] += 1
                else:
                    disconnected_clients.append(client_id)
                    self.stats["failed_sends"] += 1
            else:
                disconnected_clients.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
            
    async def broadcast_market_update(self, data: dict):
        """Specialized broadcast for market data updates"""
        market_message = {
            "type": "market_update",
            "data": data,
            "source": "real_time_feed"
        }
        await self.broadcast_to_topic("market_data", market_message)
        
    async def broadcast_news_update(self, data: dict):
        """Specialized broadcast for news updates"""
        news_message = {
            "type": "news_update", 
            "data": data,
            "source": "news_feed"
        }
        await self.broadcast_to_topic("news", news_message)
        
    async def broadcast_ai_signal(self, data: dict):
        """Specialized broadcast for AI trading signals"""
        signal_message = {
            "type": "ai_signal",
            "data": data,
            "source": "ai_agents"
        }
        await self.broadcast_to_topic("signals", signal_message)
        
    async def get_stats(self) -> dict:
        """Get WebSocket manager statistics"""
        return {
            **self.stats,
            "topics": list(self.subscriptions.keys()),
            "topic_subscribers": {
                topic: len(subscribers) 
                for topic, subscribers in self.subscriptions.items()
            }
        }
        
    async def _health_check_loop(self):
        """Background task to check connection health and cleanup dead connections"""
        while True:
            try:
                # Perform connection cleanup
                disconnected_count = await self.cleanup_dead_connections()
                
                if disconnected_count > 0:
                    logger.info(f" Health check removed {disconnected_count} dead connections")
                
                # Log stats periodically
                if len(self.active_connections) > 0:
                    logger.info(f" WebSocket Stats: {len(self.active_connections)}/{self.max_connections} active connections")
                    
                # Reduce check frequency when near connection limit
                if len(self.active_connections) > self.max_connections * 0.8:
                    await asyncio.sleep(5)  # Check more frequently when near limit
                else:
                    await asyncio.sleep(10)  # Regular check interval
                    
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(10)
                
    async def cleanup_dead_connections(self):
        """Immediately cleanup dead connections"""
        disconnected = []
        for client_id, connection in list(self.active_connections.items()):
            try:
                from fastapi.websockets import WebSocketState
                if not connection.is_active or connection.websocket.client_state != WebSocketState.CONNECTED:
                    disconnected.append(client_id)
            except:
                disconnected.append(client_id)
                
        for client_id in disconnected:
            await self.disconnect(client_id)
            
        self.stats["last_cleanup"] = datetime.utcnow().isoformat()
        return len(disconnected)
                
    async def _cleanup_loop(self):
        """Background task to cleanup old queued messages"""
        while True:
            try:
                # Remove messages older than 1 hour
                current_time = datetime.utcnow()
                for client_id in list(self.message_queue.keys()):
                    messages = self.message_queue[client_id]
                    # Filter out old messages (simplified - in production, add timestamps)
                    if len(messages) > 100:  # Keep only latest 100 messages
                        self.message_queue[client_id] = messages[-100:]
                        
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                await asyncio.sleep(300)
