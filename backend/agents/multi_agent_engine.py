#!/usr/bin/env python3
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json

logger = logging.getLogger("MultiAgentEngine")

class ConsensusLevel(Enum):
    UNANIMOUS = "unanimous"      # All agents agree (100%)
    STRONG = "strong"           # 80%+ agreement
    MODERATE = "moderate"       # 60%+ agreement
    WEAK = "weak"              # 40%+ agreement
    CONFLICTED = "conflicted"   # <40% agreement

class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ANALYZE = "analyze"
    ALERT = "alert"

@dataclass
class AgentInput:
    agent_id: str
    recommendation: str
    confidence: float
    reasoning: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    weight: float = 1.0

@dataclass
class ConsensusResult:
    action: ActionType
    consensus_level: ConsensusLevel
    confidence: float
    participating_agents: List[str]
    reasoning_summary: List[str]
    voting_breakdown: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

class MultiAgentCollaborationEngine:
    
    def __init__(self):
        self.agents = {}
        self.agent_performance = defaultdict(lambda: {
            'total_predictions': 0,
            'correct_predictions': 0,
            'average_confidence': 0.0,
            'weight': 1.0,
            'recent_performance': deque(maxlen=100)
        })
        
        # Collaboration history
        self.collaboration_history = deque(maxlen=1000)
        self.consensus_cache = {}
        
        # Performance tracking
        self.successful_consensus = 0
        self.total_consensus = 0
        
        logger.info("🤖 Multi-Agent Collaboration Engine initialized")
    
    async def register_agent(self, agent_id: str, agent_instance):
        """Register an AI agent with the collaboration engine"""
        self.agents[agent_id] = agent_instance
        logger.info(f"✅ Agent {agent_id} registered for collaboration")
    
    async def request_consensus(self, 
        symbol: str, 
        analysis_type: str,
        market_data: Dict[str, Any],
        timeout: float = 5.0) -> ConsensusResult:
        logger.info(f"🔄 Requesting consensus for {symbol} - {analysis_type}")
        
        # Gather agent inputs in parallel
        agent_inputs = await self._gather_agent_inputs(
            symbol, analysis_type, market_data, timeout
        )
        
        if not agent_inputs:
            return self._create_fallback_consensus(symbol, analysis_type)
        
        # Calculate consensus
        consensus = await self._calculate_consensus(agent_inputs, symbol, analysis_type)
        
        # Update collaboration history
        self.collaboration_history.append({
            'symbol': symbol,
            'analysis_type': analysis_type,
            'consensus': consensus,
            'timestamp': datetime.now()
        })
        
        # Cache result
        cache_key = f"{symbol}_{analysis_type}_{int(datetime.now().timestamp() // 30)}"  # 30-second cache
        self.consensus_cache[cache_key] = consensus
        
        self.total_consensus += 1
        logger.info(f"✅ Consensus reached for {symbol}: {consensus.action.value} ({consensus.consensus_level.value})")
        
        return consensus
    
    async def _gather_agent_inputs(self, 
        symbol: str, 
        analysis_type: str,
        market_data: Dict[str, Any],
        timeout: float) -> List[AgentInput]:
        """Gather inputs from all agents in parallel"""
        tasks = []
        
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(
                self._get_agent_input(agent_id, agent, symbol, analysis_type, market_data)
            )
            tasks.append(task)
        
        # Wait for all agents with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Filter successful responses
            agent_inputs = []
            for result in results:
                if isinstance(result, AgentInput):
                    # Apply dynamic weighting
                    result.weight = self.agent_performance[result.agent_id]['weight']
                    agent_inputs.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"⚠️ Agent failed to respond: {result}")
            
            return agent_inputs
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout waiting for agent responses ({timeout}s)")
            return []
    
    async def _get_agent_input(self, 
        agent_id: str, 
        agent, 
        symbol: str,
        analysis_type: str,
        market_data: Dict[str, Any]) -> AgentInput:
        """Get input from a specific agent"""
        try:
            # Create agent-specific message
            message = {
                'type': 'consensus_request',
                'symbol': symbol,
                'analysis_type': analysis_type,
                'market_data': market_data,
                'request_id': f"consensus_{symbol}_{int(datetime.now().timestamp())}"
            }
            
            # Get agent response
            response = await agent.process_message(message)
            
            # Parse agent response
            recommendation = response.get('recommendation', 'HOLD')
            confidence = response.get('confidence', 0.5)
            reasoning = response.get('reasoning', [f"{agent_id} analysis"])
            data = response.get('data', {})
            
            return AgentInput(
                agent_id=agent_id,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                data=data,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting input from {agent_id}: {e}")
            raise e
    
    async def _calculate_consensus(self, 
        agent_inputs: List[AgentInput],
        symbol: str,
        analysis_type: str) -> ConsensusResult:
        """Calculate consensus from agent inputs using advanced algorithms"""
        
        # Voting system with weighted scores
        action_votes = defaultdict(float)
        total_weight = 0
        reasoning_combined = []
        participating_agents = []
        voting_breakdown = {}
        
        for input_data in agent_inputs:
            # Map recommendation to action
            action = self._map_recommendation_to_action(input_data.recommendation)
            
            # Weight the vote by confidence and agent performance
            weighted_vote = input_data.confidence * input_data.weight
            action_votes[action.value] += weighted_vote
            total_weight += input_data.weight
            
            # Collect metadata
            participating_agents.append(input_data.agent_id)
            reasoning_combined.extend(input_data.reasoning)
            
            voting_breakdown[input_data.agent_id] = {
                'recommendation': input_data.recommendation,
                'confidence': input_data.confidence,
                'weight': input_data.weight,
                'weighted_vote': weighted_vote
            }
        
        # Determine winning action
        if not action_votes:
            winning_action = ActionType.HOLD
            consensus_confidence = 0.5
        else:
            winning_action_str = max(action_votes.items(), key=lambda x: x[1])[0]
            winning_action = ActionType(winning_action_str)
            consensus_confidence = action_votes[winning_action_str] / total_weight if total_weight > 0 else 0.5
        
        # Calculate consensus level
        consensus_level = self._determine_consensus_level(action_votes, total_weight)
        
        # Create reasoning summary (top reasons)
        reasoning_summary = list(set(reasoning_combined))[:5]  # Top 5 unique reasons
        
        return ConsensusResult(
            action=winning_action,
            consensus_level=consensus_level,
            confidence=min(consensus_confidence, 0.95),
            participating_agents=participating_agents,
            reasoning_summary=reasoning_summary,
            voting_breakdown=voting_breakdown,
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'analysis_type': analysis_type,
                'total_agents': len(agent_inputs),
                'total_weight': total_weight,
                'vote_distribution': dict(action_votes)
            }
        )
    
    def _map_recommendation_to_action(self, recommendation: str) -> ActionType:
        """Map agent recommendation to action type"""
        recommendation = recommendation.upper()
        
        if recommendation in ['BUY', 'STRONG_BUY', 'BULLISH']:
            return ActionType.BUY
        elif recommendation in ['SELL', 'STRONG_SELL', 'BEARISH']:
            return ActionType.SELL
        elif recommendation in ['ANALYZE', 'RESEARCH', 'INVESTIGATE']:
            return ActionType.ANALYZE
        elif recommendation in ['ALERT', 'WARNING', 'CAUTION']:
            return ActionType.ALERT
        else:
            return ActionType.HOLD
    
    def _determine_consensus_level(self, action_votes: Dict[str, float], total_weight: float) -> ConsensusLevel:
        """Determine consensus level based on vote distribution"""
        if not action_votes or total_weight == 0:
            return ConsensusLevel.CONFLICTED
        
        # Calculate percentage of winning vote
        max_votes = max(action_votes.values())
        winning_percentage = max_votes / total_weight
        
        if winning_percentage >= 1.0:
            return ConsensusLevel.UNANIMOUS
        elif winning_percentage >= 0.8:
            return ConsensusLevel.STRONG
        elif winning_percentage >= 0.6:
            return ConsensusLevel.MODERATE
        elif winning_percentage >= 0.4:
            return ConsensusLevel.WEAK
        else:
            return ConsensusLevel.CONFLICTED
    
    def _create_fallback_consensus(self, symbol: str, analysis_type: str) -> ConsensusResult:
        """Create fallback consensus when no agents respond"""
        return ConsensusResult(
            action=ActionType.HOLD,
            consensus_level=ConsensusLevel.CONFLICTED,
            confidence=0.1,
            participating_agents=[],
            reasoning_summary=["No agent responses available"],
            voting_breakdown={},
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'analysis_type': analysis_type,
                'fallback': True
            }
        )
    
    async def update_agent_performance(self, agent_id: str, was_correct: bool, confidence: float):
        """Update agent performance metrics for dynamic weighting"""
        perf = self.agent_performance[agent_id]
        
        perf['total_predictions'] += 1
        if was_correct:
            perf['correct_predictions'] += 1
        
        perf['recent_performance'].append(1 if was_correct else 0)
        
        # Update average confidence
        perf['average_confidence'] = (
            (perf['average_confidence'] * (perf['total_predictions'] - 1) + confidence) 
            / perf['total_predictions']
        )
        
        # Calculate accuracy
        accuracy = perf['correct_predictions'] / perf['total_predictions']
        recent_accuracy = np.mean(perf['recent_performance']) if perf['recent_performance'] else 0.5
        
        # Update dynamic weight (combination of overall accuracy and recent performance)
        perf['weight'] = (accuracy * 0.3 + recent_accuracy * 0.7) * (1 + perf['average_confidence'])
        
        logger.info(f"📊 Updated {agent_id} performance: {accuracy:.2%} accuracy, weight: {perf['weight']:.2f}")
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get comprehensive collaboration statistics"""
        return {
            'total_consensus_requests': self.total_consensus,
            'successful_consensus_rate': self.successful_consensus / max(self.total_consensus, 1),
            'registered_agents': len(self.agents),
            'agent_performance': dict(self.agent_performance),
            'recent_collaborations': len(self.collaboration_history),
            'cache_size': len(self.consensus_cache)
        }
    
    def get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Get agent performance rankings"""
        rankings = []
        
        for agent_id, perf in self.agent_performance.items():
            accuracy = perf['correct_predictions'] / max(perf['total_predictions'], 1)
            rankings.append({
                'agent_id': agent_id,
                'accuracy': accuracy,
                'weight': perf['weight'],
                'total_predictions': perf['total_predictions'],
                'average_confidence': perf['average_confidence'],
                'recent_performance': np.mean(perf['recent_performance']) if perf['recent_performance'] else 0.5
            })
        
        # Sort by weight (overall performance)
        rankings.sort(key=lambda x: x['weight'], reverse=True)
        
        return rankings
    
    async def get_consensus_history(self, 
        symbol: Optional[str] = None,
        hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical consensus data"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        history = []
        for record in self.collaboration_history:
            if record['timestamp'] >= cutoff:
                if symbol is None or record['symbol'] == symbol:
                    consensus = record['consensus']
                    history.append({
                        'symbol': record['symbol'],
                        'analysis_type': record['analysis_type'],
                        'action': consensus.action.value,
                        'consensus_level': consensus.consensus_level.value,
                        'confidence': consensus.confidence,
                        'participating_agents': len(consensus.participating_agents),
                        'timestamp': consensus.timestamp.isoformat()
                    })
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)

# Singleton instance of the MultiAgentEngine
_multi_agent_engine_instance = None

def get_multi_agent_engine():
    global _multi_agent_engine_instance
    
    if _multi_agent_engine_instance is None:
        # Use the class defined in this module
        _multi_agent_engine_instance = MultiAgentCollaborationEngine()
        
    return _multi_agent_engine_instance
