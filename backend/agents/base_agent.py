from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.start_time = datetime.now()
        self.message_count = 0
        self.error_count = 0
        self.last_activity = datetime.now()
        self.is_active = True
        self.agent_id = name.lower().replace(' ', '_')  # Add agent_id attribute
        self.memory = []  # Add memory for storing events
        
        # Agent-specific capabilities
        self.capabilities: List[AgentCapability] = []
        
        logger.info(f"✅ {self.name} agent initialized")
    
    def update_status(self, status: str, message: str = ""):
        self.last_activity = datetime.now()
        if status == "active":
            self.is_active = True
        elif status == "error":
            self.error_count += 1
        
        if message:
            logger.debug(f"{self.name} status: {status} - {message}")
    
    def add_to_memory(self, event: str, data: Dict[str, Any] = None):
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {}
        }
        self.memory.append(memory_entry)
        
        # Keep only last 100 memory entries to prevent memory bloat
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
        
        logger.debug(f"{self.name} memory: {event}")
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.message_count += 1
            self.last_activity = datetime.now()
            
            # Add metadata
            enhanced_message = {
                **message,
                'agent_name': self.name,
                'timestamp': datetime.now().isoformat(),
                'message_id': f"{self.name}_{self.message_count}"
            }
            
            # Process message
            response = await self.process_message(enhanced_message)
            
            # Add response metadata
            if isinstance(response, dict):
                response.update({
                    'agent_name': self.name,
                    'processing_time_ms': (datetime.now() - self.last_activity).total_seconds() * 1000,
                    'message_id': enhanced_message['message_id']
                })
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in {self.name} agent: {e}")
            
            return {
                'status': 'error',
                'agent_name': self.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def add_capability(self, capability: AgentCapability):
        """Add capability to agent"""
        self.capabilities.append(capability)
        logger.info(f"Added capability '{capability.name}' to {self.name}")
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get list of agent capabilities"""
        return [
            {
                'name': cap.name,
                'description': cap.description,
                'input_types': cap.input_types,
                'output_types': cap.output_types
            }
            for cap in self.capabilities
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        uptime = datetime.now() - self.start_time
        
        # Calculate success rate
        total_messages = self.message_count
        success_rate = ((total_messages - self.error_count) / total_messages * 100) if total_messages > 0 else 100
        
        return {
            'agent_name': self.name,
            'status': 'healthy' if self.is_active and success_rate > 90 else 'degraded',
            'uptime_seconds': uptime.total_seconds(),
            'message_count': self.message_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'last_activity': self.last_activity.isoformat(),
            'capabilities_count': len(self.capabilities),
            'version': self.version
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        uptime = datetime.now() - self.start_time
        
        return {
            'agent_name': self.name,
            'uptime_hours': uptime.total_seconds() / 3600,
            'messages_processed': self.message_count,
            'messages_per_hour': self.message_count / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 0 else 0,
            'error_rate': (self.error_count / self.message_count * 100) if self.message_count > 0 else 0,
            'last_activity_minutes_ago': (datetime.now() - self.last_activity).total_seconds() / 60,
            'is_active': self.is_active
        }
    
    def get_latest_alerts(self) -> List[Dict[str, Any]]:
        return []
    
    def get_latest_signals(self) -> List[Dict[str, Any]]:
        return []
    
    async def start(self):
        """Start the agent"""
        self.is_active = True
        self.start_time = datetime.now()
        logger.info(f" {self.name} agent started")
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        logger.info(f" {self.name} agent stopped")
    
    async def restart(self):
        """Restart the agent"""
        await self.stop()
        await self.start()
        logger.info(f" {self.name} agent restarted")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.message_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        logger.info(f" {self.name} metrics reset")
    
    def __str__(self) -> str:
        return f"{self.name} Agent (v{self.version})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
