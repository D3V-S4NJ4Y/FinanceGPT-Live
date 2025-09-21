import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .live_ai_agent import get_live_ai_app
from .realtime_rag_engine import market_streamer, pathway_rag_engine
import logging

logger = logging.getLogger(__name__)

def create_pathway_app():
    """Create Pathway-powered FastAPI application"""
    
    app = FastAPI(
        title="FinanceGPT Pathway LiveAI",
        description="Real-time financial AI powered by Pathway streaming engine",
        version="1.0.0-pathway"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount Live AI agent
    live_ai_app = get_live_ai_app()
    app.mount("/api/live-ai", live_ai_app)
    
    @app.on_event("startup")
    async def startup_event():
        """Start Pathway streaming on startup"""
        logger.info(" Starting Pathway LiveAI streaming...")
        
        # Start market data streaming
        asyncio.create_task(market_streamer.start_streaming())
        
        logger.info("✅ Pathway LiveAI system started")
    
    @app.get("/")
    async def root():
        return {
            "system": "FinanceGPT Pathway LiveAI",
            "status": "operational",
            "features": [
                "Real-time market data streaming",
                "Live vector embeddings",
                "Dynamic RAG responses",
                "Multi-agent orchestration"
            ]
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "pathway": "active"}
    
    return app

def run_pathway_server():
    """Run the Pathway server"""
    app = create_pathway_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

if __name__ == "__main__":
    run_pathway_server()