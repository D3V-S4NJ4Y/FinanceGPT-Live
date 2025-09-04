import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    console.error('ErrorBoundary: Caught error:', error);
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary: Component stack trace:', errorInfo.componentStack);
    console.error('ErrorBoundary: Error details:', error);
    this.setState({
      error,
      errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-red-900 text-white flex items-center justify-center">
          <div className="bg-red-800/90 rounded-lg p-8 max-w-2xl mx-4">
            <h1 className="text-2xl font-bold mb-4">🚨 Application Error</h1>
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2">Error:</h3>
              <pre className="bg-black/30 p-3 rounded text-sm overflow-auto">
                {this.state.error?.toString()}
              </pre>
            </div>
            {this.state.errorInfo && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Component Stack:</h3>
                <pre className="bg-black/30 p-3 rounded text-sm overflow-auto max-h-40">
                  {this.state.errorInfo.componentStack}
                </pre>
              </div>
            )}
            <button 
              onClick={() => window.location.reload()} 
              className="bg-red-600 hover:bg-red-500 px-4 py-2 rounded transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
