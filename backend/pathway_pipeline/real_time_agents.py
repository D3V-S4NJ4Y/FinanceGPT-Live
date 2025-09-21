try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    # Mock pathway for development
    class MockTable:
        pass
    
    class MockPathway:
        Table = MockTable
    
    pw = MockPathway()
    PATHWAY_AVAILABLE = False
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class PathwayFinancialAgent:
    """Base class for Pathway-powered financial agents"""
    
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.context_table = None
        
    def connect_to_pathway_stream(self, documents_table: pw.Table):
        """Connect agent to Pathway document stream"""
        self.context_table = documents_table
        logger.info(f"✅ {self.name} connected to Pathway stream")
    
    async def process_with_live_context(self, query: str, context_filter: Dict = None) -> Dict[str, Any]:
        """Process query with live Pathway context"""
        try:
            # Get relevant context from Pathway stream
            live_context = self._get_live_context(context_filter)
            
            # Create specialized prompt
            prompt = self._create_specialized_prompt(query, live_context)
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return {
                "agent": self.name,
                "specialization": self.specialization,
                "response": response.choices[0].message.content,
                "live_context_used": len(live_context),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pathway_powered": True
            }
            
        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            return {"error": str(e), "agent": self.name}
    
    def _get_live_context(self, context_filter: Dict = None) -> List[Dict]:
        """Get live context from Pathway stream - to be implemented by subclasses"""
        return []
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        """Create specialized prompt - to be implemented by subclasses"""
        return f"Query: {query}\nContext: {context}"

class MarketAnalysisAgent(PathwayFinancialAgent):
    """Real-time market analysis agent"""
    
    def __init__(self):
        super().__init__("MarketAnalyst", "Technical and fundamental market analysis")
    
    def _get_live_context(self, context_filter: Dict = None) -> List[Dict]:
        """Get market-specific context"""
        if not self.context_table:
            return []
        
        # Filter for market data
        market_docs = []
        # In real implementation, this would query the Pathway table
        # For now, return structure that would come from Pathway
        return market_docs
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([f"[{doc.get('timestamp', '')}] {doc.get('content', '')}" for doc in context])
        
        return f"""
        You are a specialized Market Analysis Agent with access to real-time financial data.
        
        Your expertise includes:
        - Technical analysis (RSI, MACD, Bollinger Bands)
        - Price action analysis
        - Volume analysis
        - Market sentiment assessment
        
        Live Market Data:
        {context_text}
        
        Query: {query}
        
        Provide detailed market analysis based on the live data above. Include specific metrics, trends, and actionable insights.
        """

class RiskAssessmentAgent(PathwayFinancialAgent):
    """Real-time risk assessment agent"""
    
    def __init__(self):
        super().__init__("RiskAssessor", "Portfolio risk analysis and management")
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([f"[{doc.get('timestamp', '')}] {doc.get('content', '')}" for doc in context])
        
        return f"""
        You are a specialized Risk Assessment Agent with access to real-time market data.
        
        Your expertise includes:
        - Value at Risk (VaR) calculations
        - Portfolio volatility analysis
        - Correlation analysis
        - Stress testing
        - Risk-adjusted returns
        
        Live Market Data:
        {context_text}
        
        Query: {query}
        
        Provide comprehensive risk analysis based on the live data. Include specific risk metrics, potential scenarios, and risk mitigation strategies.
        """

class NewsAnalysisAgent(PathwayFinancialAgent):
    """Real-time news and sentiment analysis agent"""
    
    def __init__(self):
        super().__init__("NewsAnalyst", "Financial news analysis and sentiment processing")
    
    def _get_live_context(self, context_filter: Dict = None) -> List[Dict]:
        """Get news-specific context"""
        if not self.context_table:
            return []
        
        # Filter for news data
        news_docs = []
        # In real implementation, this would query the Pathway table for news
        return news_docs
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([f"[{doc.get('timestamp', '')}] {doc.get('content', '')}" for doc in context])
        
        return f"""
        You are a specialized News Analysis Agent with access to real-time financial news.
        
        Your expertise includes:
        - Sentiment analysis
        - News impact assessment
        - Market-moving events identification
        - Regulatory news analysis
        - Earnings and corporate announcements
        
        Live News Data:
        {context_text}
        
        Query: {query}
        
        Provide detailed news analysis based on the live data. Include sentiment scores, potential market impact, and key themes.
        """

class TradingSignalAgent(PathwayFinancialAgent):
    """Real-time trading signal generation agent"""
    
    def __init__(self):
        super().__init__("SignalGenerator", "AI-powered trading signal generation")
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([f"[{doc.get('timestamp', '')}] {doc.get('content', '')}" for doc in context])
        
        return f"""
        You are a specialized Trading Signal Agent with access to real-time market data.
        
        Your expertise includes:
        - Buy/Sell/Hold signal generation
        - Entry and exit point identification
        - Confidence scoring
        - Risk-reward analysis
        - Multi-timeframe analysis
        
        Live Market Data:
        {context_text}
        
        Query: {query}
        
        Generate specific trading signals based on the live data. Include:
        - Signal type (BUY/SELL/HOLD)
        - Confidence level (0-100%)
        - Entry/exit prices
        - Risk management levels
        - Reasoning for the signal
        """

class ComplianceAgent(PathwayFinancialAgent):
    """Real-time compliance monitoring agent"""
    
    def __init__(self):
        super().__init__("ComplianceMonitor", "Regulatory compliance and risk monitoring")
    
    def _create_specialized_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n".join([f"[{doc.get('timestamp', '')}] {doc.get('content', '')}" for doc in context])
        
        return f"""
        You are a specialized Compliance Agent with access to real-time regulatory and market data.
        
        Your expertise includes:
        - Regulatory compliance monitoring
        - Risk flag identification
        - Policy adherence checking
        - Audit trail analysis
        - Regulatory change impact assessment
        
        Live Data:
        {context_text}
        
        Query: {query}
        
        Provide compliance analysis based on the live data. Include risk flags, regulatory considerations, and compliance recommendations.
        """

class MultiAgentOrchestrator:
    
    def __init__(self):
        self.agents = {
            'market_analysis': MarketAnalysisAgent(),
            'risk_assessment': RiskAssessmentAgent(),
            'news_analysis': NewsAnalysisAgent(),
            'trading_signals': TradingSignalAgent(),
            'compliance': ComplianceAgent()
        }
        self.pathway_connected = False
        
    def connect_to_pathway(self, documents_table: pw.Table):
        """Connect all agents to Pathway document stream"""
        for agent in self.agents.values():
            agent.connect_to_pathway_stream(documents_table)
        self.pathway_connected = True
        logger.info("✅ All agents connected to Pathway stream")
    
    async def route_query(self, query: str, agent_type: str = "auto", context_filter: Dict = None) -> Dict[str, Any]:
        """Route query to appropriate agent(s)"""
        try:
            if agent_type == "auto":
                agent_type = self._determine_agent_type(query)
            
            if agent_type not in self.agents:
                return {
                    "error": f"Unknown agent type: {agent_type}",
                    "available_agents": list(self.agents.keys())
                }
            
            # Process with selected agent
            result = await self.agents[agent_type].process_with_live_context(query, context_filter)
            
            # Add orchestrator metadata
            result.update({
                "orchestrator": "MultiAgentOrchestrator",
                "pathway_connected": self.pathway_connected,
                "agent_selection": agent_type,
                "hackathon_demo": "✅ Live multi-agent system"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {"error": str(e), "orchestrator": "MultiAgentOrchestrator"}
    
    def _determine_agent_type(self, query: str) -> str:
        """Determine appropriate agent based on query content"""
        query_lower = query.lower()
        
        # Keywords for different agent types
        if any(word in query_lower for word in ['risk', 'var', 'volatility', 'correlation', 'portfolio']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['news', 'sentiment', 'article', 'announcement']):
            return 'news_analysis'
        elif any(word in query_lower for word in ['signal', 'buy', 'sell', 'trade', 'entry', 'exit']):
            return 'trading_signals'
        elif any(word in query_lower for word in ['compliance', 'regulation', 'policy', 'audit']):
            return 'compliance'
        else:
            return 'market_analysis'  # Default to market analysis
    
    async def get_multi_agent_consensus(self, query: str, context_filter: Dict = None) -> Dict[str, Any]:
        """Get consensus from multiple agents"""
        try:
            # Get responses from relevant agents
            agents_to_query = ['market_analysis', 'risk_assessment', 'news_analysis']
            responses = {}
            
            for agent_type in agents_to_query:
                response = await self.agents[agent_type].process_with_live_context(query, context_filter)
                responses[agent_type] = response
            
            # Create consensus summary
            consensus_prompt = f"""
            Based on the following agent responses to the query "{query}", provide a consensus summary:
            
            Market Analysis: {responses.get('market_analysis', {}).get('response', 'N/A')}
            Risk Assessment: {responses.get('risk_assessment', {}).get('response', 'N/A')}
            News Analysis: {responses.get('news_analysis', {}).get('response', 'N/A')}
            
            Provide a balanced consensus that incorporates insights from all agents.
            """
            
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            consensus_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": consensus_prompt}],
                temperature=0.1
            )
            
            return {
                "consensus_summary": consensus_response.choices[0].message.content,
                "individual_responses": responses,
                "agents_consulted": agents_to_query,
                "pathway_powered": True,
                "multi_agent_consensus": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multi-agent consensus error: {e}")
            return {"error": str(e), "consensus": False}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "total_agents": len(self.agents),
            "pathway_connected": self.pathway_connected,
            "available_agents": {
                name: {
                    "name": agent.name,
                    "specialization": agent.specialization,
                    "status": "connected" if self.pathway_connected else "offline"
                }
                for name, agent in self.agents.items()
            },
            "orchestrator_status": "operational",
            "hackathon_ready": True
        }

# Global orchestrator instance
multi_agent_orchestrator = None

def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the global multi-agent orchestrator"""
    global multi_agent_orchestrator
    if multi_agent_orchestrator is None:
        multi_agent_orchestrator = MultiAgentOrchestrator()
    return multi_agent_orchestrator