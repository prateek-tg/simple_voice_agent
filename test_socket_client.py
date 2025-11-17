#!/usr/bin/env python3
"""
Test client for Socket.IO server communication.
"""
import socketio
import time
import sys

# Server configuration
SERVER_URL = 'http://51.38.38.66:5000'

# Create a Socket.IO client
sio = socketio.Client()

# Track connection status
connected = False
response_received = False

@sio.event
def connect():
    """Handle connection event."""
    global connected
    connected = True
    print(f"✅ Successfully connected to {SERVER_URL}")
    print(f"   Session ID: {sio.sid}\n")

@sio.event
def disconnect():
    """Handle disconnection event."""
    print("❌ Disconnected from server\n")

@sio.on('status')
def on_status(data):
    """Handle status messages."""
    print(f"📢 Status: {data.get('message', 'No message')}")
    print(f"   Type: {data.get('type', 'unknown')}\n")

@sio.on('query_received')
def on_query_received(data):
    """Handle query acknowledgment."""
    print(f"📨 Query received by server")
    print(f"   Status: {data.get('status', 'unknown')}")
    print(f"   Timestamp: {data.get('timestamp', 'N/A')}\n")

@sio.on('bot_response')
def on_bot_response(data):
    """Handle bot response."""
    global response_received
    response_received = True
    print(f"🤖 Bot Response:")
    print(f"   {data.get('message', 'No message')}\n")
    print(f"   Original Query: {data.get('original_query', 'N/A')}")
    print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"   Is Goodbye: {data.get('is_goodbye', False)}\n")

@sio.on('session_ending')
def on_session_ending(data):
    """Handle session ending notification."""
    print(f"👋 Session Ending:")
    print(f"   {data.get('message', 'No message')}\n")

@sio.on('health_status')
def on_health_status(data):
    """Handle health check response."""
    print(f"❤️  Health Status:")
    for key, value in data.items():
        print(f"   {key}: {value}")
    print()

@sio.on('session_stats')
def on_session_stats(data):
    """Handle session statistics."""
    print(f"📊 Session Statistics:")
    for key, value in data.items():
        print(f"   {key}: {value}")
    print()

@sio.on('error')
def on_error(data):
    """Handle error messages."""
    print(f"⚠️  Error:")
    print(f"   Message: {data.get('message', 'Unknown error')}")
    if 'error' in data:
        print(f"   Details: {data.get('error')}")
    print()

def test_connection():
    """Test basic connection to the server."""
    print(f"🔌 Testing connection to {SERVER_URL}...")
    print(f"   Attempting to connect...\n")
    
    try:
        sio.connect(SERVER_URL, wait_timeout=10)
        time.sleep(2)  # Wait for connection to stabilize
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_health_check():
    """Test health check endpoint."""
    print("🏥 Testing health check...")
    sio.emit('health_check')
    time.sleep(2)

def test_send_query(query):
    """Test sending a query to the chatbot."""
    global response_received
    response_received = False
    
    print(f"💬 Sending query: '{query}'")
    sio.emit('user_query', {'message': query})
    
    # Wait for response (max 30 seconds)
    timeout = 30
    elapsed = 0
    while not response_received and elapsed < timeout:
        time.sleep(1)
        elapsed += 1
    
    if not response_received:
        print(f"⚠️  No response received within {timeout} seconds\n")
    
    return response_received

def test_get_stats():
    """Test getting session statistics."""
    print("📊 Requesting session statistics...")
    sio.emit('get_stats')
    time.sleep(2)

def interactive_mode():
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE")
    print("="*60)
    print("Type your questions or 'quit' to exit\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Exiting interactive mode...")
                break
            
            print()  # Add spacing
            test_send_query(query)
            print()  # Add spacing after response
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break

def main():
    """Main test function."""
    print("="*60)
    print("🧪 SOCKET.IO SERVER TEST CLIENT")
    print("="*60)
    print()
    
    # Test connection
    if not test_connection():
        sys.exit(1)
    
    print("="*60)
    print("🧪 Running automated tests...")
    print("="*60)
    print()
    
    # Test health check
    test_health_check()
    
    # Test a simple query
    test_send_query("What is your privacy policy?")
    
    # Test getting stats
    test_get_stats()
    
    # Ask if user wants interactive mode
    print("\n" + "="*60)
    try:
        choice = input("Would you like to enter interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
    except KeyboardInterrupt:
        print("\n")
    
    # Disconnect
    print("\n" + "="*60)
    print("🔌 Disconnecting from server...")
    sio.disconnect()
    print("✅ Test completed!")
    print("="*60)

if __name__ == '__main__':
    main()
