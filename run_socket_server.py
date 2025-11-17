#!/usr/bin/env python3
"""
Simple script to run the socket server with proper environment setup.
"""
import os
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run the socket server."""
    try:
        # Import and run the socket server
        from socket_server import app, socketio
        
        print("🚀 Starting Privacy Policy Chatbot Socket Server...")
        print("📱 Web interface: http://localhost:5000")
        print("🔌 Socket.IO endpoint: ws://localhost:5000/socket.io/")
        print("Press Ctrl+C to stop the server\n")
        
        # Run the server
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()