import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Comprehensive market context for LLM reasoning"""
    current_prices: Dict[str, float]
    price_changes: Dict[str, float]
    volumes: Dict[str, int]
    technical_indicators: Dict[str, Dict[str, float]]
    news_sentiment: Dict[str, float]
    sector_performance: Dict[str, float]
    market_regime: str
    volatility_index: float
    economic_indicators: Dict[str, float]

class AdvancedLLMEngine:
    """Advanced LLM Engine with financial reasoning capabilities"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.reasoning_cache = {}
        self.market_memory = []
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query with advanced reasoning"""
        
        # 1. Gather comprehensive market context
        market_context = await self._gather_market_context(context.get('symbols', []))
        
        # 2. Analyze query intent and complexity
        query_analysis = self._analyze_query_intent(query)
        
        # 3. Apply multi-step reasoning
        reasoning_steps = await self._multi_step_reasoning(query, market_context, query_analysis)
        
        # 4. Generate comprehensive response
        response = await self._generate_response(query, reasoning_steps, market_context)
        
        return response
    
    async def _gather_market_context(self, symbols: List[str]) -> MarketContext:
        """Gather comprehensive real-time market data"""
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'VIX']
        
        # Parallel data fetching for speed
        tasks = [
            self._fetch_price_data(symbols),
            self._fetch_technical_indicators(symbols),
            self._fetch_sector_data(),
            self._fetch_market_regime(),
            self._fetch_economic_indicators()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_data = results[0] if not isinstance(results[0], Exception) else {}
        technical_data = results[1] if not isinstance(results[1], Exception) else {}
        sector_data = results[2] if not isinstance(results[2], Exception) else {}
        market_regime = results[3] if not isinstance(results[3], Exception) else "Unknown"
        economic_data = results[4] if not isinstance(results[4], Exception) else {}
        
        return MarketContext(
            current_prices={s: price_data.get(s, {}).get('price', 0) for s in symbols},
            price_changes={s: price_data.get(s, {}).get('change_percent', 0) for s in symbols},
            volumes={s: price_data.get(s, {}).get('volume', 0) for s in symbols},
            technical_indicators=technical_data,
            news_sentiment={s: 0.5 for s in symbols},  # Placeholder for news sentiment
            sector_performance=sector_data,
            market_regime=market_regime,
            volatility_index=price_data.get('VIX', {}).get('price', 20),
            economic_indicators=economic_data
        )
    
    async def _fetch_price_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch real-time price data"""
        price_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                info = ticker.info
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    
                    price_data[symbol] = {
                        'price': current_price,
                        'change': current_price - prev_price,
                        'change_percent': change_percent,
                        'volume': int(hist['Volume'].iloc[-1]),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'sector': info.get('sector', 'Unknown')
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                
        return price_data
    
    async def _fetch_technical_indicators(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate advanced technical indicators"""
        technical_data = {}
        
        for symbol in symbols[:5]:  # Limit to prevent API overload
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                
                if len(hist) >= 50:
                    closes = hist['Close'].values
                    volumes = hist['Volume'].values
                    
                    # RSI
                    rsi = self._calculate_rsi(closes)
                    
                    # MACD
                    macd_line, signal_line = self._calculate_macd(closes)
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)
                    
                    # Volume indicators
                    volume_sma = np.mean(volumes[-20:])
                    volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1
                    
                    # Momentum
                    momentum_10 = (closes[-1] / closes[-11] - 1) * 100 if len(closes) > 10 else 0
                    momentum_20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) > 20 else 0
                    
                    # Volatility
                    returns = np.diff(closes) / closes[:-1]
                    volatility = np.std(returns) * np.sqrt(252) * 100
                    
                    technical_data[symbol] = {
                        'rsi': rsi[-1] if len(rsi) > 0 else 50,
                        'macd': macd_line[-1] if len(macd_line) > 0 else 0,
                        'macd_signal': signal_line[-1] if len(signal_line) > 0 else 0,
                        'bb_position': (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if len(bb_upper) > 0 else 0.5,
                        'volume_ratio': volume_ratio,
                        'momentum_10d': momentum_10,
                        'momentum_20d': momentum_20,
                        'volatility': volatility,
                        'support_level': np.min(closes[-20:]),
                        'resistance_level': np.max(closes[-20:])
                    }
            except Exception as e:
                logger.warning(f"Failed to calculate technical indicators for {symbol}: {e}")
                
        return technical_data
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to understand intent and complexity"""
        
        query_lower = query.lower()
        
        # Intent classification
        intents = {
            'price_analysis': any(word in query_lower for word in ['price', 'cost', 'value', 'worth']),
            'technical_analysis': any(word in query_lower for word in ['technical', 'chart', 'pattern', 'indicator', 'rsi', 'macd']),
            'risk_assessment': any(word in query_lower for word in ['risk', 'safe', 'dangerous', 'volatile', 'stability']),
            'prediction': any(word in query_lower for word in ['predict', 'forecast', 'future', 'will', 'expect']),
            'comparison': any(word in query_lower for word in ['compare', 'vs', 'versus', 'better', 'worse']),
            'portfolio': any(word in query_lower for word in ['portfolio', 'holdings', 'allocation', 'diversification']),
            'news_impact': any(word in query_lower for word in ['news', 'earnings', 'announcement', 'event']),
            'sector_analysis': any(word in query_lower for word in ['sector', 'industry', 'technology', 'healthcare'])
        }
        
        # Extract symbols mentioned
        symbols_mentioned = []
        common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        for symbol in common_symbols:
            if symbol.lower() in query_lower:
                symbols_mentioned.append(symbol)
        
        # Complexity assessment
        complexity_indicators = [
            len(query.split()) > 10,  # Long query
            sum(intents.values()) > 2,  # Multiple intents
            len(symbols_mentioned) > 1,  # Multiple symbols
            any(word in query_lower for word in ['why', 'how', 'explain', 'analyze', 'detailed'])
        ]
        
        complexity = 'high' if sum(complexity_indicators) >= 3 else 'medium' if sum(complexity_indicators) >= 1 else 'low'
        
        return {
            'intents': intents,
            'symbols_mentioned': symbols_mentioned,
            'complexity': complexity,
            'requires_calculation': any(word in query_lower for word in ['calculate', 'compute', 'ratio', 'percentage']),
            'time_sensitive': any(word in query_lower for word in ['now', 'current', 'today', 'latest'])
        }
    
    async def _multi_step_reasoning(self, query: str, context: MarketContext, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply multi-step reasoning to complex queries"""
        
        reasoning_steps = []
        
        # Step 1: Data Analysis
        if analysis['intents']['price_analysis'] or analysis['intents']['technical_analysis']:
            price_analysis = self._analyze_price_movements(context, analysis['symbols_mentioned'])
            reasoning_steps.append({
                'step': 'price_analysis',
                'findings': price_analysis,
                'confidence': 0.9
            })
        
        # Step 2: Technical Pattern Recognition
        if analysis['intents']['technical_analysis']:
            technical_patterns = self._identify_technical_patterns(context, analysis['symbols_mentioned'])
            reasoning_steps.append({
                'step': 'technical_patterns',
                'findings': technical_patterns,
                'confidence': 0.85
            })
        
        # Step 3: Risk Assessment
        if analysis['intents']['risk_assessment']:
            risk_analysis = self._assess_risk_factors(context, analysis['symbols_mentioned'])
            reasoning_steps.append({
                'step': 'risk_assessment',
                'findings': risk_analysis,
                'confidence': 0.8
            })
        
        # Step 4: Market Context Integration
        market_context_analysis = self._analyze_market_context(context)
        reasoning_steps.append({
            'step': 'market_context',
            'findings': market_context_analysis,
            'confidence': 0.75
        })
        
        # Step 5: Predictive Analysis
        if analysis['intents']['prediction']:
            predictions = self._generate_predictions(context, analysis['symbols_mentioned'])
            reasoning_steps.append({
                'step': 'predictions',
                'findings': predictions,
                'confidence': 0.7
            })
        
        return reasoning_steps
    
    def _analyze_price_movements(self, context: MarketContext, symbols: List[str]) -> Dict[str, Any]:
        """Analyze price movements with advanced logic"""
        
        if not symbols:
            symbols = list(context.current_prices.keys())[:5]
        
        analysis = {
            'summary': '',
            'individual_analysis': {},
            'market_sentiment': 'neutral'
        }
        
        total_change = 0
        positive_moves = 0
        
        for symbol in symbols:
            if symbol in context.current_prices:
                price = context.current_prices[symbol]
                change = context.price_changes[symbol]
                volume = context.volumes[symbol]
                
                # Individual stock analysis
                if abs(change) > 5:
                    strength = 'strong'
                elif abs(change) > 2:
                    strength = 'moderate'
                else:
                    strength = 'weak'
                
                direction = 'bullish' if change > 0 else 'bearish' if change < 0 else 'neutral'
                
                # Volume analysis
                volume_context = 'high' if volume > 1000000 else 'normal' if volume > 100000 else 'low'
                
                analysis['individual_analysis'][symbol] = {
                    'price': price,
                    'change_percent': change,
                    'direction': direction,
                    'strength': strength,
                    'volume_context': volume_context,
                    'interpretation': f"{symbol} shows {strength} {direction} movement with {volume_context} volume"
                }
                
                total_change += change
                if change > 0:
                    positive_moves += 1
        
        # Overall market sentiment
        avg_change = total_change / len(symbols) if symbols else 0
        positive_ratio = positive_moves / len(symbols) if symbols else 0
        
        if avg_change > 1 and positive_ratio > 0.6:
            analysis['market_sentiment'] = 'bullish'
        elif avg_change < -1 and positive_ratio < 0.4:
            analysis['market_sentiment'] = 'bearish'
        else:
            analysis['market_sentiment'] = 'neutral'
        
        analysis['summary'] = f"Market showing {analysis['market_sentiment']} sentiment with {positive_ratio:.1%} of stocks positive"
        
        return analysis
    
    def _identify_technical_patterns(self, context: MarketContext, symbols: List[str]) -> Dict[str, Any]:
        """Identify technical patterns using advanced algorithms"""
        
        patterns = {
            'summary': '',
            'patterns_found': [],
            'signals': []
        }
        
        for symbol in symbols:
            if symbol in context.technical_indicators:
                indicators = context.technical_indicators[symbol]
                
                # RSI patterns
                rsi = indicators.get('rsi', 50)
                if rsi > 70:
                    patterns['patterns_found'].append(f"{symbol}: Overbought (RSI: {rsi:.1f})")
                    patterns['signals'].append(f"{symbol}: Consider taking profits")
                elif rsi < 30:
                    patterns['patterns_found'].append(f"{symbol}: Oversold (RSI: {rsi:.1f})")
                    patterns['signals'].append(f"{symbol}: Potential buying opportunity")
                
                # MACD patterns
                macd = indicators.get('macd', 0)
                macd_signal = indicators.get('macd_signal', 0)
                if macd > macd_signal and macd > 0:
                    patterns['patterns_found'].append(f"{symbol}: Bullish MACD crossover")
                    patterns['signals'].append(f"{symbol}: Upward momentum building")
                elif macd < macd_signal and macd < 0:
                    patterns['patterns_found'].append(f"{symbol}: Bearish MACD crossover")
                    patterns['signals'].append(f"{symbol}: Downward pressure increasing")
                
                # Bollinger Band patterns
                bb_position = indicators.get('bb_position', 0.5)
                if bb_position > 0.9:
                    patterns['patterns_found'].append(f"{symbol}: Near upper Bollinger Band")
                    patterns['signals'].append(f"{symbol}: Potential resistance level")
                elif bb_position < 0.1:
                    patterns['patterns_found'].append(f"{symbol}: Near lower Bollinger Band")
                    patterns['signals'].append(f"{symbol}: Potential support level")
        
        patterns['summary'] = f"Identified {len(patterns['patterns_found'])} technical patterns across {len(symbols)} symbols"
        
        return patterns
    
    def _assess_risk_factors(self, context: MarketContext, symbols: List[str]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        
        risk_analysis = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'risk_score': 50,
            'recommendations': []
        }
        
        risk_score = 0
        total_factors = 0
        
        # Market volatility risk
        vix = context.volatility_index
        if vix > 30:
            risk_analysis['risk_factors'].append(f"High market volatility (VIX: {vix:.1f})")
            risk_score += 80
        elif vix > 20:
            risk_analysis['risk_factors'].append(f"Elevated volatility (VIX: {vix:.1f})")
            risk_score += 60
        else:
            risk_score += 30
        total_factors += 1
        
        # Individual stock risks
        for symbol in symbols:
            if symbol in context.technical_indicators:
                indicators = context.technical_indicators[symbol]
                volatility = indicators.get('volatility', 20)
                
                if volatility > 40:
                    risk_analysis['risk_factors'].append(f"{symbol}: High volatility ({volatility:.1f}%)")
                    risk_score += 70
                elif volatility > 25:
                    risk_score += 50
                else:
                    risk_score += 30
                total_factors += 1
        
        # Market regime risk
        if context.market_regime in ['Bear Market', 'High Volatility']:
            risk_analysis['risk_factors'].append(f"Challenging market regime: {context.market_regime}")
            risk_score += 75
            total_factors += 1
        elif context.market_regime == 'Transitional':
            risk_score += 55
            total_factors += 1
        else:
            risk_score += 35
            total_factors += 1
        
        # Calculate final risk score
        final_risk_score = risk_score / total_factors if total_factors > 0 else 50
        risk_analysis['risk_score'] = int(final_risk_score)
        
        if final_risk_score > 70:
            risk_analysis['overall_risk'] = 'high'
            risk_analysis['recommendations'].append("Consider reducing position sizes")
            risk_analysis['recommendations'].append("Implement strict stop-loss orders")
        elif final_risk_score > 50:
            risk_analysis['overall_risk'] = 'medium'
            risk_analysis['recommendations'].append("Monitor positions closely")
            risk_analysis['recommendations'].append("Maintain diversification")
        else:
            risk_analysis['overall_risk'] = 'low'
            risk_analysis['recommendations'].append("Favorable risk environment")
            risk_analysis['recommendations'].append("Consider strategic opportunities")
        
        return risk_analysis
    
    def _analyze_market_context(self, context: MarketContext) -> Dict[str, Any]:
        """Analyze broader market context"""
        
        market_analysis = {
            'regime': context.market_regime,
            'key_factors': [],
            'outlook': 'neutral'
        }
        
        # Analyze sector performance
        if context.sector_performance:
            best_sector = max(context.sector_performance.items(), key=lambda x: x[1])
            worst_sector = min(context.sector_performance.items(), key=lambda x: x[1])
            
            market_analysis['key_factors'].append(f"Best performing sector: {best_sector[0]} (+{best_sector[1]:.1f}%)")
            market_analysis['key_factors'].append(f"Worst performing sector: {worst_sector[0]} ({worst_sector[1]:.1f}%)")
        
        # VIX analysis
        vix = context.volatility_index
        if vix < 15:
            market_analysis['key_factors'].append("Low fear index suggests complacency")
            market_analysis['outlook'] = 'cautiously optimistic'
        elif vix > 25:
            market_analysis['key_factors'].append("Elevated fear index indicates uncertainty")
            market_analysis['outlook'] = 'cautious'
        
        # Market breadth analysis
        positive_stocks = sum(1 for change in context.price_changes.values() if change > 0)
        total_stocks = len(context.price_changes)
        breadth = positive_stocks / total_stocks if total_stocks > 0 else 0.5
        
        if breadth > 0.7:
            market_analysis['key_factors'].append("Strong market breadth supports rally")
            market_analysis['outlook'] = 'optimistic'
        elif breadth < 0.3:
            market_analysis['key_factors'].append("Weak market breadth suggests selling pressure")
            market_analysis['outlook'] = 'pessimistic'
        
        return market_analysis
    
    def _generate_predictions(self, context: MarketContext, symbols: List[str]) -> Dict[str, Any]:
        """Generate AI-powered predictions"""
        
        predictions = {
            'short_term': {},
            'medium_term': {},
            'confidence_levels': {},
            'key_assumptions': []
        }
        
        for symbol in symbols:
            if symbol in context.current_prices and symbol in context.technical_indicators:
                price = context.current_prices[symbol]
                indicators = context.technical_indicators[symbol]
                
                # Short-term prediction (1-5 days)
                rsi = indicators.get('rsi', 50)
                momentum = indicators.get('momentum_10d', 0)
                
                if rsi < 30 and momentum > -5:
                    short_term_direction = 'up'
                    short_term_magnitude = 3 + abs(momentum) * 0.5
                    confidence = 75
                elif rsi > 70 and momentum < 5:
                    short_term_direction = 'down'
                    short_term_magnitude = 2 + abs(momentum) * 0.3
                    confidence = 70
                else:
                    short_term_direction = 'sideways'
                    short_term_magnitude = 1
                    confidence = 60
                
                predictions['short_term'][symbol] = {
                    'direction': short_term_direction,
                    'magnitude': f"{short_term_magnitude:.1f}%",
                    'target_price': price * (1 + short_term_magnitude/100 * (1 if short_term_direction == 'up' else -1 if short_term_direction == 'down' else 0))
                }
                
                predictions['confidence_levels'][symbol] = confidence
        
        predictions['key_assumptions'] = [
            "Predictions based on technical analysis",
            "Market conditions remain stable",
            "No major news events",
            "Historical patterns continue"
        ]
        
        return predictions
    
    async def _generate_response(self, query: str, reasoning_steps: List[Dict[str, Any]], context: MarketContext) -> Dict[str, Any]:
        """Generate comprehensive response using all reasoning steps"""
        
        response_parts = []
        
        # Synthesize findings from all reasoning steps
        for step in reasoning_steps:
            step_type = step['step']
            findings = step['findings']
            confidence = step['confidence']
            
            if step_type == 'price_analysis':
                response_parts.append(f" **Price Analysis**: {findings['summary']}")
                
            elif step_type == 'technical_patterns':
                if findings['patterns_found']:
                    response_parts.append(f" **Technical Patterns**: {findings['summary']}")
                    for signal in findings['signals'][:3]:  # Top 3 signals
                        response_parts.append(f"  • {signal}")
                        
            elif step_type == 'risk_assessment':
                response_parts.append(f"⚠️ **Risk Assessment**: {findings['overall_risk'].upper()} risk level ({findings['risk_score']}/100)")
                for rec in findings['recommendations'][:2]:
                    response_parts.append(f"  • {rec}")
                    
            elif step_type == 'market_context':
                response_parts.append(f" **Market Context**: {findings['regime']} regime detected")
                for factor in findings['key_factors'][:2]:
                    response_parts.append(f"  • {factor}")
                    
            elif step_type == 'predictions':
                response_parts.append(" **AI Predictions**:")
                for symbol, pred in list(findings['short_term'].items())[:3]:
                    confidence = findings['confidence_levels'].get(symbol, 60)
                    response_parts.append(f"  • {symbol}: {pred['direction']} {pred['magnitude']} (confidence: {confidence}%)")
        
        # Generate final synthesis
        final_response = "\n".join(response_parts)
        
        # Add actionable recommendations
        final_response += "\n\n **Key Takeaways**:"
        
        # Extract key insights
        if any('bullish' in str(step['findings']) for step in reasoning_steps):
            final_response += "\n  • Market conditions show bullish signals"
        if any('risk' in str(step['findings']) for step in reasoning_steps):
            final_response += "\n  • Monitor risk factors closely"
        if any('volatility' in str(step['findings']) for step in reasoning_steps):
            final_response += "\n  • Elevated volatility requires caution"
        
        return {
            'response': final_response,
            'confidence': np.mean([step['confidence'] for step in reasoning_steps]),
            'reasoning_steps': reasoning_steps,
            'data_sources': ['Yahoo Finance', 'Technical Analysis', 'Market Data'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for technical calculations
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[~np.isnan(rsi)]
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.values, signal_line.values
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2):
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.values, sma.values, lower_band.values
    
    async def _fetch_sector_data(self) -> Dict[str, float]:
        """Fetch sector performance data"""
        # Simplified sector data - in production, use real sector ETFs
        return {
            'Technology': 1.2,
            'Healthcare': 0.8,
            'Financial': -0.3,
            'Energy': 2.1,
            'Consumer': 0.5
        }
    
    async def _fetch_market_regime(self) -> str:
        """Determine current market regime"""
        try:
            # Use VIX and SPY to determine regime
            vix = yf.Ticker('VIX').history(period='5d')
            spy = yf.Ticker('SPY').history(period='30d')
            
            if not vix.empty and not spy.empty:
                current_vix = vix['Close'].iloc[-1]
                spy_change = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1) * 100
                
                if current_vix > 30:
                    return "High Volatility"
                elif spy_change > 5:
                    return "Bull Market"
                elif spy_change < -5:
                    return "Bear Market"
                else:
                    return "Consolidation"
        except:
            pass
        
        return "Unknown"
    
    async def _fetch_economic_indicators(self) -> Dict[str, float]:
        """Fetch key economic indicators"""
        # Placeholder for economic data - in production, integrate with FRED API
        return {
            'interest_rate': 5.25,
            'inflation_rate': 3.2,
            'unemployment_rate': 3.8,
            'gdp_growth': 2.1
        }

# Global instance
advanced_llm_engine = AdvancedLLMEngine()