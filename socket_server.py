"""
Socket.IO server for real-time chatbot communication.
"""
import logging
import os
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import time

from chatbot import ChatBot
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global chatbot instance
chatbot_instance = None

def get_chatbot():
    """Get or create chatbot instance."""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = ChatBot()
        # Start a session for the chatbot
        chatbot_instance.start_session()
    return chatbot_instance

def format_response_for_web(response):
    """Format the chatbot response for better web display."""
    if not response:
        return response
    
    # Add line breaks after sentences for better readability
    formatted = response.replace('. ', '.\n\n')
    
    # Add line breaks before certain phrases for structure
    structure_phrases = [
        'First off,', 'However,', 'As for', 'And just a quick note',
        'So,', 'In simple terms,', 'For example,', 'Additionally,',
        'Important:', 'Note:', 'Remember:'
    ]
    
    for phrase in structure_phrases:
        formatted = formatted.replace(phrase, f'\n{phrase}')
    
    # Clean up multiple line breaks
    formatted = '\n'.join(line.strip() for line in formatted.split('\n') if line.strip())
    
    return formatted

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to Privacy Policy Chatbot', 'type': 'success'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('user_query')
def handle_user_query(data):
    """
    Handle incoming user query and emit response.
    
    Events:
    - 'user_query': When a query comes in
    - 'bot_response': When a response is ready
    """
    try:
        user_message = data.get('message', '').strip()
        
        if not user_message:
            emit('error', {'message': 'Empty message received'})
            return
        
        logger.info(f"Received query: {user_message}")
        
        # Emit acknowledgment that query was received
        emit('query_received', {
            'message': user_message,
            'timestamp': time.time(),
            'status': 'processing'
        })
        
        # Get chatbot instance and process message
        chatbot = get_chatbot()
        
        # Process the message (this might take some time)
        response = chatbot.process_message(user_message)
        
        logger.info(f"Generated response: {response[:100]}...")
        
        # Format response for better readability
        formatted_response = format_response_for_web(response)
        
        # Check if this was a goodbye intent - if so, prepare to end session
        chatbot_agent = chatbot.agent
        intent = chatbot_agent.classify_intent(user_message)
        is_goodbye = (intent.value == 'goodbye')
        
        # Emit the bot response
        emit('bot_response', {
            'message': formatted_response,
            'timestamp': time.time(),
            'original_query': user_message,
            'is_goodbye': is_goodbye
        })
        
        # If this was a goodbye, end the session after a brief delay
        if is_goodbye:
            logger.info("Goodbye detected - ending session")
            # Let the client know to prepare for disconnection
            emit('session_ending', {
                'message': 'Thank you for using our privacy assistant. Session will end shortly.',
                'timestamp': time.time()
            })
            
            # End the chatbot session
            chatbot.end_session()
            
            # Disconnect the client after a brief delay (2 seconds)
            import threading
            def disconnect_client():
                import time
                time.sleep(2)
                try:
                    from flask_socketio import disconnect
                    disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting client: {e}")
            
            threading.Thread(target=disconnect_client).start()
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        emit('error', {
            'message': 'Sorry, I encountered an error processing your request.',
            'error': str(e),
            'timestamp': time.time()
        })

@socketio.on('health_check')
def handle_health_check():
    """Handle health check request."""
    try:
        chatbot = get_chatbot()
        health_status = chatbot.health_check()
        emit('health_status', health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        emit('error', {'message': 'Health check failed', 'error': str(e)})

@socketio.on('get_stats')
def handle_get_stats():
    """Handle request for session statistics."""
    try:
        chatbot = get_chatbot()
        stats = chatbot.get_session_stats()
        emit('session_stats', stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        emit('error', {'message': 'Failed to get statistics', 'error': str(e)})

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

if __name__ == '__main__':
    try:
        # Check environment
        issues = check_environment()
        if issues:
            print("❌ Environment check failed:")
            for issue in issues:
                print(f"   • {issue}")
            sys.exit(1)
        
        print("🚀 Starting Socket.IO server...")
        print("📱 Chat interface will be available at: http://localhost:5000")
        
        # Run the SocketIO server
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Set to True for development
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)