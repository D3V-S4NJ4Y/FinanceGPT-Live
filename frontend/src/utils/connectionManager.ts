// Connection State Manager - Prevent Multiple API Calls
class ConnectionManager {
  private static instance: ConnectionManager;
  private isConnected: boolean = true;
  private lastHealthCheck: number = 0;
  private healthCheckInProgress: boolean = false;
  
  static getInstance(): ConnectionManager {
    if (!ConnectionManager.instance) {
      ConnectionManager.instance = new ConnectionManager();
    }
    return ConnectionManager.instance;
  }
  
  async checkConnection(): Promise<boolean> {
    const now = Date.now();
    
    // Return cached result if checked within last 30 seconds
    if (now - this.lastHealthCheck < 30000) {
      return this.isConnected;
    }
    
    // Prevent multiple simultaneous health checks
    if (this.healthCheckInProgress) {
      return this.isConnected;
    }
    
    this.healthCheckInProgress = true;
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // Quick 3s timeout
      
      const response = await fetch('http://127.0.0.1:8001/health', {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      this.isConnected = response.ok;
      this.lastHealthCheck = now;
      
      console.log(`🔌 Connection check: ${this.isConnected ? 'Online' : 'Offline'}`);
      
    } catch (error) {
      this.isConnected = false;
      this.lastHealthCheck = now;
      console.log('🔌 Connection check: Failed');
    } finally {
      this.healthCheckInProgress = false;
    }
    
    return this.isConnected;
  }
  
  getConnectionStatus(): boolean {
    return this.isConnected;
  }
  
  setConnectionStatus(status: boolean): void {
    this.isConnected = status;
    this.lastHealthCheck = Date.now();
  }
}

export default ConnectionManager;
