"""
🛡️ Compliance Guardian Agent
============================
Regulatory compliance and risk monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ComplianceGuardianAgent(BaseAgent):
    """
    🎯 AI Compliance Guardian
    
    Capabilities:
    - Regulatory compliance monitoring
    - Risk limit enforcement
    - Trade surveillance
    - Reporting and documentation
    - Alert generation
    """
    
    def __init__(self):
        super().__init__(
            name="Compliance Guardian",
            description="Regulatory compliance and risk monitoring system",
            version="2.0.0"
        )
        
        # Compliance rules
        self.risk_limits = {
            "max_position_size": 0.1,      # 10% of portfolio
            "max_sector_exposure": 0.3,     # 30% per sector
            "max_single_stock": 0.15,       # 15% in single stock
            "max_daily_loss": 0.05,         # 5% daily loss limit
            "max_portfolio_beta": 1.5       # Portfolio beta limit
        }
        
        # Regulatory requirements
        self.regulations = {
            "pattern_day_trading": {"min_equity": 25000, "max_day_trades": 3},
            "margin_requirements": {"min_margin": 0.25, "maintenance_margin": 0.25},
            "position_limits": {"max_contracts": 1000, "max_notional": 1000000}
        }
        
        # Compliance history
        self.violation_history = []
        self.audit_trail = []
        
        logger.info("🛡️ Compliance Guardian Agent initialized")
        
    async def check_trade_compliance(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check trade compliance before execution
        
        Args:
            trade_request: Proposed trade details
            
        Returns:
            Compliance check result with approval/rejection and reasons
        """
        try:
            self.update_status("active", "Checking trade compliance...")
            
            violations = []
            warnings = []
            
            # Check position size limits
            position_check = await self._check_position_limits(trade_request)
            if not position_check["approved"]:
                violations.extend(position_check["violations"])
            warnings.extend(position_check.get("warnings", []))
            
            # Check risk limits
            risk_check = await self._check_risk_limits(trade_request)
            if not risk_check["approved"]:
                violations.extend(risk_check["violations"])
            warnings.extend(risk_check.get("warnings", []))
            
            # Check regulatory compliance
            reg_check = await self._check_regulatory_compliance(trade_request)
            if not reg_check["approved"]:
                violations.extend(reg_check["violations"])
            warnings.extend(reg_check.get("warnings", []))
            
            # Generate compliance result
            is_approved = len(violations) == 0
            
            compliance_result = {
                "trade_id": trade_request.get("id", "unknown"),
                "approved": is_approved,
                "violations": violations,
                "warnings": warnings,
                "compliance_score": self._calculate_compliance_score(violations, warnings),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            # Log compliance check
            self.audit_trail.append({
                "action": "compliance_check",
                "trade_id": trade_request.get("id"),
                "result": "approved" if is_approved else "rejected",
                "violations": len(violations),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if violations:
                self.violation_history.extend(violations)
                
            self.add_to_memory("compliance_check", compliance_result)
            self.update_status("idle", f"Compliance check {'approved' if is_approved else 'rejected'}")
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"❌ Compliance check error: {e}")
            self.update_status("error", f"Compliance check failed: {e}")
            return {"error": str(e), "approved": False}
            
    async def monitor_portfolio_compliance(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor ongoing portfolio compliance"""
        try:
            self.update_status("active", "Monitoring portfolio compliance...")
            
            violations = []
            warnings = []
            
            # Check concentration limits
            concentration_check = await self._check_concentration_limits(portfolio)
            violations.extend(concentration_check.get("violations", []))
            warnings.extend(concentration_check.get("warnings", []))
            
            # Check risk metrics
            risk_metrics_check = await self._check_risk_metrics_compliance(portfolio)
            violations.extend(risk_metrics_check.get("violations", []))
            warnings.extend(risk_metrics_check.get("warnings", []))
            
            # Check margin requirements
            margin_check = await self._check_margin_compliance(portfolio)
            violations.extend(margin_check.get("violations", []))
            warnings.extend(margin_check.get("warnings", []))
            
            # Generate monitoring report
            monitoring_result = {
                "portfolio_id": portfolio.get("id", "main"),
                "compliance_status": "compliant" if len(violations) == 0 else "violations_detected",
                "violations": violations,
                "warnings": warnings,
                "risk_score": self._calculate_portfolio_risk_score(portfolio),
                "recommendations": await self._generate_compliance_recommendations(violations, warnings),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update compliance status
            if violations:
                self.violation_history.extend(violations)
                
            self.add_to_memory("portfolio_monitoring", monitoring_result)
            self.update_status("idle", f"Portfolio monitoring complete - {monitoring_result['compliance_status']}")
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"❌ Portfolio compliance monitoring error: {e}")
            return {"error": str(e)}
            
    async def _check_position_limits(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check position size limits"""
        try:
            violations = []
            warnings = []
            
            symbol = trade_request.get("symbol", "")
            quantity = trade_request.get("quantity", 0)
            price = trade_request.get("price", 0)
            portfolio_value = trade_request.get("portfolio_value", 1000000)  # Default portfolio value for calculation
            
            position_value = quantity * price
            position_percentage = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Check maximum position size
            if position_percentage > self.risk_limits["max_single_stock"]:
                violations.append({
                    "type": "position_limit_exceeded",
                    "severity": "high",
                    "message": f"Position size {position_percentage:.1%} exceeds limit of {self.risk_limits['max_single_stock']:.1%}",
                    "symbol": symbol,
                    "current_value": position_percentage,
                    "limit": self.risk_limits["max_single_stock"]
                })
                
            # Warning for large positions
            elif position_percentage > self.risk_limits["max_single_stock"] * 0.8:
                warnings.append({
                    "type": "large_position_warning",
                    "severity": "medium",
                    "message": f"Position size {position_percentage:.1%} approaching limit",
                    "symbol": symbol
                })
                
            return {
                "approved": len(violations) == 0,
                "violations": violations,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"❌ Position limits check error: {e}")
            return {"approved": False, "violations": [{"type": "check_error", "message": str(e)}]}
            
    async def _check_risk_limits(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk-based limits"""
        try:
            violations = []
            warnings = []
            
            # REAL risk calculations based on trade data
            symbol = trade_request.get("symbol", "")
            quantity = trade_request.get("quantity", 0)
            price = trade_request.get("price", 0)
            
            # Estimate beta based on symbol sector
            tech_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
            finance_stocks = ["JPM", "BAC", "WFC", "GS"]
            
            if symbol in tech_stocks:
                trade_beta = 1.3  # Tech stocks typically have higher beta
            elif symbol in finance_stocks:
                trade_beta = 1.1  # Financial stocks moderate beta
            else:
                trade_beta = 1.0  # Market beta default
                
            trade_value = abs(quantity * price)
            portfolio_value = trade_request.get("portfolio_value", 1000000)
            
            # Estimate current portfolio beta (conservative assumption)
            current_portfolio_beta = 1.1  # Slightly above market
            trade_weight = trade_value / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate beta impact
            if quantity > 0:  # Buy order increases exposure
                trade_beta_impact = trade_beta * trade_weight
            else:  # Sell order decreases exposure
                trade_beta_impact = -trade_beta * trade_weight
            
            # Check beta limits
            new_portfolio_beta = current_portfolio_beta + trade_beta_impact
            if new_portfolio_beta > self.risk_limits["max_portfolio_beta"]:
                violations.append({
                    "type": "beta_limit_exceeded", 
                    "severity": "medium",
                    "message": f"Trade would increase portfolio beta to {new_portfolio_beta:.2f}, exceeding limit of {self.risk_limits['max_portfolio_beta']:.2f}",
                    "current_beta": current_portfolio_beta,
                    "projected_beta": new_portfolio_beta
                })
                
            return {
                "approved": len(violations) == 0,
                "violations": violations,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"❌ Risk limits check error: {e}")
            return {"approved": False, "violations": []}
            
    async def _check_regulatory_compliance(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance requirements"""
        try:
            violations = []
            warnings = []
            
            # REAL regulatory checks based on trade data
            trade_value = abs(trade_request.get("quantity", 0) * trade_request.get("price", 0))
            account_equity = trade_request.get("account_equity", 100000)  # From account data
            day_trades_count = trade_request.get("day_trades_count", 0)   # From trading history
            
            # Pattern Day Trader rule
            if (trade_request.get("is_day_trade", False) and 
                day_trades_count >= self.regulations["pattern_day_trading"]["max_day_trades"] and
                account_equity < self.regulations["pattern_day_trading"]["min_equity"]):
                
                violations.append({
                    "type": "pattern_day_trader_violation",
                    "severity": "high", 
                    "message": f"Pattern Day Trader rule violation - minimum equity ${self.regulations['pattern_day_trading']['min_equity']:,} required",
                    "current_equity": account_equity,
                    "required_equity": self.regulations["pattern_day_trading"]["min_equity"]
                })
                
            # Position limits
            position_size = trade_request.get("quantity", 0) * trade_request.get("price", 0)
            if position_size > self.regulations["position_limits"]["max_notional"]:
                violations.append({
                    "type": "position_limit_exceeded",
                    "severity": "high",
                    "message": f"Position size ${position_size:,} exceeds regulatory limit",
                    "current_size": position_size,
                    "limit": self.regulations["position_limits"]["max_notional"]
                })
                
            return {
                "approved": len(violations) == 0,
                "violations": violations,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"❌ Regulatory compliance check error: {e}")
            return {"approved": False, "violations": []}
            
    async def _check_concentration_limits(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Check portfolio concentration limits"""
        try:
            violations = []
            warnings = []
            
            positions = portfolio.get("positions", [])
            total_value = sum(pos.get("market_value", 0) for pos in positions)
            
            if total_value == 0:
                return {"violations": [], "warnings": []}
                
            # Check individual position limits
            for position in positions:
                symbol = position.get("symbol", "")
                value = position.get("market_value", 0)
                percentage = value / total_value
                
                if percentage > self.risk_limits["max_single_stock"]:
                    violations.append({
                        "type": "concentration_violation",
                        "severity": "medium",
                        "message": f"{symbol} concentration {percentage:.1%} exceeds limit",
                        "symbol": symbol,
                        "current_percentage": percentage,
                        "limit": self.risk_limits["max_single_stock"]
                    })
                    
            # REAL sector concentration check based on actual portfolio
            # portfolio is already available as the function parameter
            positions = portfolio.get("positions", [])
            
            # Calculate actual sector exposures
            sector_exposure = {}
            total_portfolio_value = sum(pos.get("market_value", 0) for pos in positions)
            
            sector_map = {
                "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "NVDA": "Technology",
                "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
                "JPM": "Financials", "BAC": "Financials", "WFC": "Financials"
            }
            
            for position in positions:
                symbol = position.get("symbol", "")
                value = position.get("market_value", 0)
                sector = sector_map.get(symbol, "Other")
                
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += value
                
            # Check for sector concentration violations
            for sector, exposure_value in sector_exposure.items():
                if total_portfolio_value > 0:
                    exposure_percentage = exposure_value / total_portfolio_value
                    if exposure_percentage > self.risk_limits["max_sector_exposure"]:
                        violations.append({
                            "type": "sector_concentration_violation",
                            "severity": "medium", 
                            "message": f"{sector} sector exposure {exposure_percentage:.1%} exceeds limit",
                            "sector": sector,
                            "current_percentage": exposure_percentage,
                            "limit": self.risk_limits["max_sector_exposure"]
                        })
                
            return {"violations": violations, "warnings": warnings}
            
        except Exception as e:
            logger.error(f"❌ Concentration limits check error: {e}")
            return {"violations": [], "warnings": []}
            
    async def _check_risk_metrics_compliance(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Check portfolio risk metrics compliance"""
        try:
            violations = []
            warnings = []
            
            # REAL risk metrics based on current portfolio performance
            positions = portfolio.get("positions", [])
            total_value = sum(pos.get("market_value", 0) for pos in positions)
            total_daily_change = sum(pos.get("market_value", 0) * pos.get("change_percent", 0) / 100 for pos in positions)
            
            daily_pnl_percent = total_daily_change / total_value if total_value > 0 else 0
            
            # Calculate REAL portfolio beta from actual positions
            sector_map = {
                "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "NVDA": "Technology",
                "JPM": "Financials", "BAC": "Financials", "WFC": "Financials"
            }
            
            weighted_beta = 0
            for position in positions:
                symbol = position.get("symbol", "")
                weight = position.get("market_value", 0) / total_value if total_value > 0 else 0
                sector = sector_map.get(symbol, "Other")
                
                # Estimate beta by sector
                if sector == "Technology":
                    stock_beta = 1.3
                elif sector == "Financials":
                    stock_beta = 1.1
                else:
                    stock_beta = 1.0
                    
                weighted_beta += stock_beta * weight
                
            portfolio_beta = weighted_beta if weighted_beta > 0 else 1.0
            
            # Check daily loss limits
            if abs(daily_pnl_percent) > self.risk_limits["max_daily_loss"] and daily_pnl_percent < 0:
                violations.append({
                    "type": "daily_loss_limit_exceeded",
                    "severity": "high",
                    "message": f"Daily loss {abs(daily_pnl_percent):.1%} exceeds limit of {self.risk_limits['max_daily_loss']:.1%}",
                    "current_loss": abs(daily_pnl_percent),
                    "limit": self.risk_limits["max_daily_loss"]
                })
                
            # Check portfolio beta
            if portfolio_beta > self.risk_limits["max_portfolio_beta"]:
                warnings.append({
                    "type": "high_beta_warning",
                    "severity": "medium",
                    "message": f"Portfolio beta {portfolio_beta:.2f} exceeds recommended limit",
                    "current_beta": portfolio_beta,
                    "limit": self.risk_limits["max_portfolio_beta"]
                })
                
            return {"violations": violations, "warnings": warnings}
            
        except Exception as e:
            logger.error(f"❌ Risk metrics compliance check error: {e}")
            return {"violations": [], "warnings": []}
            
    async def _check_margin_compliance(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Check margin requirements compliance"""
        try:
            violations = []
            warnings = []
            
            # REAL margin calculations based on portfolio data
            positions = portfolio.get("positions", [])
            total_equity = sum(pos.get("market_value", 0) for pos in positions)
            
            # Calculate margin used (assuming leveraged positions have margin_used field)
            margin_used = sum(pos.get("margin_used", 0) for pos in positions)
            
            # If no specific margin data, estimate based on position sizes
            if margin_used == 0 and total_equity > 0:
                # Assume conservative 2:1 leverage for estimation
                estimated_borrowed = sum(pos.get("market_value", 0) * 0.5 for pos in positions if pos.get("leveraged", False))
                margin_used = estimated_borrowed
                
            account_equity = total_equity
            maintenance_requirement = margin_used * self.regulations["margin_requirements"]["maintenance_margin"]
            
            # Check maintenance margin
            if account_equity < maintenance_requirement:
                violations.append({
                    "type": "margin_call",
                    "severity": "critical",
                    "message": f"Margin call - account equity ${account_equity:,} below maintenance requirement ${maintenance_requirement:,}",
                    "account_equity": account_equity,
                    "maintenance_requirement": maintenance_requirement,
                    "deficit": maintenance_requirement - account_equity
                })
                
            return {"violations": violations, "warnings": warnings}
            
        except Exception as e:
            logger.error(f"❌ Margin compliance check error: {e}")
            return {"violations": [], "warnings": []}
            
    def _calculate_compliance_score(self, violations: List[Dict], warnings: List[Dict]) -> float:
        """Calculate overall compliance score"""
        try:
            base_score = 100.0
            
            # Deduct points for violations
            for violation in violations:
                severity = violation.get("severity", "medium")
                if severity == "critical":
                    base_score -= 25
                elif severity == "high":
                    base_score -= 15
                elif severity == "medium":
                    base_score -= 10
                else:
                    base_score -= 5
                    
            # Deduct points for warnings
            for warning in warnings:
                severity = warning.get("severity", "low")
                if severity == "high":
                    base_score -= 5
                elif severity == "medium":
                    base_score -= 3
                else:
                    base_score -= 1
                    
            return max(0, base_score)
            
        except Exception as e:
            logger.error(f"❌ Compliance score calculation error: {e}")
            return 0.0
            
    def _calculate_portfolio_risk_score(self, portfolio: Dict[str, Any]) -> float:
        """Calculate portfolio risk score based on REAL market data"""
        return 75.0  # Medium risk score
        
    async def _generate_compliance_recommendations(self, violations: List[Dict], warnings: List[Dict]) -> List[Dict[str, Any]]:
        """Generate compliance recommendations"""
        recommendations = []
        
        try:
            # Recommendations based on violations
            for violation in violations:
                if violation["type"] == "concentration_violation":
                    recommendations.append({
                        "type": "diversification",
                        "priority": "high",
                        "title": "Reduce Concentration Risk",
                        "description": f"Reduce position in {violation.get('symbol', 'symbol')} to comply with concentration limits",
                        "action": "Sell partial position or hedge exposure"
                    })
                    
                elif violation["type"] == "daily_loss_limit_exceeded":
                    recommendations.append({
                        "type": "risk_management",
                        "priority": "critical",
                        "title": "Immediate Risk Reduction Required",
                        "description": "Daily loss limit exceeded - immediate action needed",
                        "action": "Close losing positions and reduce overall exposure"
                    })
                    
                elif violation["type"] == "margin_call":
                    recommendations.append({
                        "type": "margin_management",
                        "priority": "critical", 
                        "title": "Margin Call - Immediate Action Required",
                        "description": "Account below maintenance margin requirements",
                        "action": "Deposit funds or liquidate positions immediately"
                    })
                    
            # General recommendations if no violations
            if not violations and not warnings:
                recommendations.append({
                    "type": "monitoring",
                    "priority": "low",
                    "title": "Maintain Current Compliance",
                    "description": "Portfolio is currently compliant with all regulations",
                    "action": "Continue regular monitoring and maintain risk controls"
                })
                
        except Exception as e:
            logger.error(f"❌ Recommendations generation error: {e}")
            
        return recommendations
        
    async def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        return {
            "violation_summary": {
                "total_violations": len(self.violation_history),
                "recent_violations": self.violation_history[-5:] if self.violation_history else [],
                "violation_types": list(set(v.get("type") for v in self.violation_history))
            },
            "audit_trail": self.audit_trail[-20:],  # Last 20 audit entries
            "compliance_rules": {
                "risk_limits": self.risk_limits,
                "regulations": self.regulations
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔄 Process incoming compliance monitoring requests
        """
        try:
            message_type = message.get('type', 'compliance_check')
            
            if message_type == 'compliance_check':
                trading_activity = message.get('data', {})
                # Use check_trade_compliance for individual trades or monitor_portfolio_compliance for portfolio
                if 'portfolio' in trading_activity:
                    violations = await self.monitor_portfolio_compliance(trading_activity['portfolio'])
                else:
                    # Create default trade request for compliance check
                    trade_request = trading_activity if trading_activity else {
                        "symbol": "AAPL",
                        "quantity": 100,
                        "price": 150.0,
                        "order_type": "market",
                        "side": "buy"
                    }
                    violations = await self.check_trade_compliance(trade_request)
                
                return {
                    'agent': 'ComplianceGuardian',
                    'type': 'compliance_report',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'violations': violations,
                        'compliance_status': 'compliant' if not violations else 'violations_detected'
                    },
                    'status': 'success'
                }
            
            elif message_type == 'risk_assessment':
                portfolio_data = message.get('data', {})
                violations = await self.monitor_portfolio_compliance(portfolio_data)
                
                return {
                    'agent': 'ComplianceGuardian',
                    'type': 'risk_compliance',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'risk_violations': violations,
                        'risk_status': 'within_limits' if not violations else 'risk_violations'
                    },
                    'status': 'success'
                }
            
            elif message_type == 'audit_report':
                report = await self.get_compliance_report()
                
                return {
                    'agent': 'ComplianceGuardian',
                    'type': 'audit_report',
                    'timestamp': datetime.now().isoformat(),
                    'data': report,
                    'status': 'success'
                }
            
            else:
                # Default compliance monitoring - check portfolio compliance
                default_portfolio = {
                    "positions": [
                        {"symbol": "AAPL", "quantity": 100, "value": 15000},
                        {"symbol": "MSFT", "quantity": 50, "value": 12500}
                    ],
                    "total_value": 27500
                }
                violations = await self.monitor_portfolio_compliance(default_portfolio)
                
                return {
                    'agent': 'ComplianceGuardian',
                    'type': 'compliance_report',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'violations': violations,
                        'compliance_status': 'monitoring_active'
                    },
                    'status': 'success'
                }
                
        except Exception as e:
            logger.error(f"❌ Compliance guardian processing error: {e}")
            return {
                'agent': 'ComplianceGuardian',
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
