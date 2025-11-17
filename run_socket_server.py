#!/usr/bin/env python3
"""
Simple script to run the raw socket server with proper environment setup.
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
        from socket_server import start_server
        
        print("🚀 Starting Privacy Policy Chatbot Socket Server...")
        print("� Socket endpoint: tcp://51.38.38.66:5000")
        print("� Protocol: JSON messages over TCP (newline-delimited)")
        print("Press Ctrl+C to stop the server\n")
        
        # Run the server
        start_server(
            host='51.38.38.66',
            port=5000
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()