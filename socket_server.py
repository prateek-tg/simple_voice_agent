"""
Raw Python socket server for chatbot communication.
"""
import logging
import os
import sys
from pathlib import Path
import socket
import json
import threading
import time

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from chatbot import ChatBot
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance and active clients
chatbot_instance = None
active_clients = {}

def get_chatbot():
    """Get or create chatbot instance."""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = ChatBot()
        # Start a session for the chatbot
        chatbot_instance.start_session()
    return chatbot_instance

def send_message(client_socket, message_type, data):
    """Send a JSON message to the client."""
    try:
        message = json.dumps({
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        })
        client_socket.sendall((message + '\n').encode('utf-8'))
        return True
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False

def handle_client(client_socket, client_address):
    """Handle individual client connection."""
    client_id = f"{client_address[0]}:{client_address[1]}"
    logger.info(f"Client connected: {client_id}")
    
    active_clients[client_id] = client_socket
    
    # Send welcome message
    send_message(client_socket, 'status', {
        'message': 'Connected to Privacy Policy Chatbot',
        'type': 'success',
        'client_id': client_id
    })
    
    buffer = ""
    
    try:
        while True:
            # Receive data from client
            data = client_socket.recv(4096).decode('utf-8')
            
            if not data:
                logger.info(f"Client disconnected: {client_id}")
                break
            
            buffer += data
            
            # Process complete messages (separated by newlines)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    message = json.loads(line)
                    handle_message(client_socket, message, client_id)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {client_id}: {e}")
                    send_message(client_socket, 'error', {
                        'message': 'Invalid JSON format'
                    })
    
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    
    finally:
        # Clean up
        if client_id in active_clients:
            del active_clients[client_id]
        client_socket.close()
        logger.info(f"Connection closed: {client_id}")

def handle_message(client_socket, message, client_id):
    """Handle different message types from client."""
    msg_type = message.get('type', '')
    data = message.get('data', {})
    
    logger.info(f"Received message from {client_id}: {msg_type}")
    
    if msg_type == 'user_query':
        handle_user_query(client_socket, data, client_id)
    
    elif msg_type == 'health_check':
        handle_health_check(client_socket)
    
    elif msg_type == 'get_stats':
        handle_get_stats(client_socket)
    
    else:
        send_message(client_socket, 'error', {
            'message': f'Unknown message type: {msg_type}'
        })

def handle_user_query(client_socket, data, client_id):
    """Handle incoming user query."""
    try:
        user_message = data.get('message', '').strip()
        
        if not user_message:
            send_message(client_socket, 'error', {
                'message': 'Empty message received'
            })
            return
        
        logger.info(f"Processing query from {client_id}: {user_message}")
        
        # Send acknowledgment
        send_message(client_socket, 'query_received', {
            'message': user_message,
            'status': 'processing'
        })
        
        # Get chatbot instance and process message
        chatbot = get_chatbot()
        response = chatbot.process_message(user_message)
        
        logger.info(f"Generated response: {response[:100]}...")
        
        # Check if this was a goodbye intent
        chatbot_agent = chatbot.agent
        intent = chatbot_agent.classify_intent(user_message)
        is_goodbye = (intent.value == 'goodbye')
        
        # Send the bot response
        send_message(client_socket, 'bot_response', {
            'message': response,
            'original_query': user_message,
            'is_goodbye': is_goodbye
        })
        
        # If goodbye, end session
        if is_goodbye:
            logger.info("Goodbye detected - ending session")
            send_message(client_socket, 'session_ending', {
                'message': 'Thank you for using our privacy assistant. Session will end shortly.'
            })
            chatbot.end_session()
            
            # Close connection after brief delay
            time.sleep(2)
            client_socket.close()
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        send_message(client_socket, 'error', {
            'message': 'Sorry, I encountered an error processing your request.',
            'error': str(e)
        })

def handle_health_check(client_socket):
    """Handle health check request."""
    try:
        chatbot = get_chatbot()
        health_status = chatbot.health_check()
        send_message(client_socket, 'health_status', health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        send_message(client_socket, 'error', {
            'message': 'Health check failed',
            'error': str(e)
        })

def handle_get_stats(client_socket):
    """Handle request for session statistics."""
    try:
        chatbot = get_chatbot()
        stats = chatbot.get_session_stats()
        send_message(client_socket, 'session_stats', stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        send_message(client_socket, 'error', {
            'message': 'Failed to get statistics',
            'error': str(e)
        })

def check_environment():
    """Check if all required environment variables are set."""
    issues = []
    
    # Check OpenAI API key
    if not config.openai_api_key:
        issues.append("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
    
    # Check if data file exists
    if not os.path.exists(config.data_file_path):
        issues.append(f"Data file not found: {config.data_file_path}")
    
    return issues

def start_server(host='0.0.0.0', port=5000):
    """Start the socket server."""
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        
        logger.info(f"🚀 Socket server started on {host}:{port}")
        print(f"🚀 Socket server listening on {host}:{port}")
        print(f"🔌 Waiting for connections...")
        print(f"Press Ctrl+C to stop\n")
        
        while True:
            # Accept client connection
            client_socket, client_address = server_socket.accept()
            
            # Handle client in a separate thread
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address),
                daemon=True
            )
            client_thread.start()
    
    except KeyboardInterrupt:
        print("\n� Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Server error: {e}")
    finally:
        server_socket.close()
        logger.info("Server socket closed")

if __name__ == '__main__':
    try:
        # Check environment
        issues = check_environment()
        if issues:
            print("❌ Environment check failed:")
            for issue in issues:
                print(f"   • {issue}")
            sys.exit(1)
        
        # Start the raw socket server
        start_server(host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)