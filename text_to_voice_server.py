#!/usr/bin/env python3
"""
Text-to-Voice Socket.IO Server for Frontend Integration
Accepts text queries and streams audio responses to browser
"""

import socketio
import logging
import asyncio
import os
import sys
import time
import requests
import base64
from pathlib import Path
from aiohttp import web
from typing import Iterator

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from chatbot import ChatBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Socket.IO server with CORS enabled
sio = socketio.AsyncServer(
    cors_allowed_origins='*',  # Allow all origins for development
    async_mode='aiohttp',
    logger=True,
    engineio_logger=True
)

app = web.Application()
sio.attach(app)

# Store client sessions
clients = {}

# Track active responses for interruption handling
active_responses = {}  # {sid: {'task': asyncio.Task, 'interrupted': bool}}


class TextToVoiceHandler:
    """Handles text-to-voice conversion and streaming."""
    
    def __init__(self):
        """Initialize TTS handler."""
        self.baseten_api_key = os.environ.get("BASETEN_API_KEY")
        if not self.baseten_api_key:
            raise ValueError("BASETEN_API_KEY not found in environment variables")
    
    async def text_to_speech_stream(self, text: str) -> Iterator[bytes]:
        """
        Convert text to speech and return audio stream.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio chunks as bytes
        """
        start = time.perf_counter()
        
        try:
            # Make async request to TTS API
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"https://model-rwnyv243.api.baseten.co/development/predict",
                    headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
                    json={"text": text, "language": "en", "chunk_size": 20},
                    stream=True,
                )
            )
            
            end = time.perf_counter()
            logger.info(f"Time to make TTS POST request: {end-start}s")
            
            if res.status_code != 200:
                logger.error(f"TTS API error: {res.text}")
                return
            
            first = True
            for chunk in res.iter_content(chunk_size=512):
                if first:
                    end = time.perf_counter()
                    logger.info(f"Time to first audio chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
            
            logger.info(f"⏱️ TTS response elapsed: {res.elapsed}")
            
        except Exception as e:
            logger.error(f"Error in text_to_speech_stream: {e}")


# Initialize TTS handler
try:
    tts_handler = TextToVoiceHandler()
except ValueError as e:
    logger.error(f"TTS initialization failed: {e}")
    sys.exit(1)


@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection."""
    logger.info(f"🔗 Client {sid} connected")
    if auth:
        logger.info(f"📋 Connection data: {auth}")
    
    try:
        # Create individual session for this client
        chatbot = ChatBot()
        session_id = chatbot.start_session()  # Get session_id
        
        clients[sid] = {
            'chatbot': chatbot,
            'session_id': session_id  # Store separately
        }
        
        # Initialize interruption tracking for this client
        active_responses[sid] = {'task': None, 'interrupted': False}
        
        # Send connection status only (no greeting message)
        await sio.emit('status', {
            'message': 'Connected',
            'type': 'success'
        }, room=sid)
        
        logger.info(f"✅ Session created for client {sid}")
        
        # No automatic greeting - wait for user input
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error creating session for {sid}: {e}")
        
        await sio.emit('error', {
            'message': f'Failed to create session: {error_message}'
        }, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"🔌 Client {sid} disconnected")
    
    if sid in clients:
        client_data = clients[sid]
        chatbot = client_data['chatbot']
        session_id = client_data['session_id']
        try:
            chatbot.end_session(session_id)  # Pass session_id
        except:
            pass
        del clients[sid]
        logger.info(f"🗑️ Session cleaned up for client {sid}")
    
    # Clean up interruption tracking
    if sid in active_responses:
        del active_responses[sid]


@sio.event
async def text_query(sid, data):
    """Handle text query from client."""
    try:
        if sid not in clients:
            await sio.emit('error', {
                'message': 'Session not found'
            }, room=sid)
            return
        
        # INTERRUPTION HANDLING: Cancel previous response if still active
        if sid in active_responses and active_responses[sid]['task']:
            logger.info(f"⚠️ Interrupting previous response for client {sid}")
            active_responses[sid]['interrupted'] = True
            
            # Cancel the previous task
            previous_task = active_responses[sid]['task']
            if previous_task and not previous_task.done():
                previous_task.cancel()
                try:
                    await previous_task
                except asyncio.CancelledError:
                    logger.info(f"✅ Previous response cancelled for {sid}")
            
            # Send interruption signal to client
            await sio.emit('response_interrupted', {
                'message': 'Previous response interrupted'
            }, room=sid)
        
        # Extract query text
        if isinstance(data, dict):
            query_text = data.get('message', '') or data.get('query', '') or data.get('text', '')
        else:
            query_text = str(data)
        
        if not query_text or not query_text.strip():
            await sio.emit('error', {
                'message': 'Empty query'
            }, room=sid)
            return
        
        client_data = clients[sid]
        chatbot = client_data['chatbot']
        session_id = client_data['session_id']  # Get session_id
        
        logger.info(f"💬 Client {sid}: '{query_text}'")
        
        # Send acknowledgment
        await sio.emit('query_received', {
            'message': query_text,
            'status': 'processing'
        }, room=sid)
        
        # Create a new task for this response
        async def process_and_respond():
            try:
                # Reset interruption flag
                active_responses[sid]['interrupted'] = False
                
                # Process query through chatbot
                logger.info(f"🔄 Processing query for {sid}...")
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    chatbot.process_message,
                    query_text,
                    session_id  # Pass session_id
                )
                
                # Check if interrupted during processing
                if active_responses[sid]['interrupted']:
                    logger.info(f"⚠️ Response interrupted during processing for {sid}")
                    return
                
                logger.info(f"✅ Client {sid} processing completed")
                
                # Send text response
                await sio.emit('text_response', {
                    'message': response,
                    'original_query': query_text,
                    'type': 'response'
                }, room=sid)
                
                logger.info(f"📤 Sent text to client {sid}: {response[:50]}...")
                
                # Check again before audio streaming
                if active_responses[sid]['interrupted']:
                    logger.info(f"⚠️ Response interrupted before audio for {sid}")
                    return
                
                # Generate and stream audio response
                await stream_audio_to_client(sid, response)
                
            except asyncio.CancelledError:
                logger.info(f"⚠️ Response task cancelled for {sid}")
                raise
            except Exception as e:
                logger.error(f"Error processing query for {sid}: {e}")
                await sio.emit('error', {
                    'message': f'Processing error: {str(e)}'
                }, room=sid)
        
        # Start the task and track it
        task = asyncio.create_task(process_and_respond())
        active_responses[sid]['task'] = task
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Task was cancelled due to interruption
            
    except Exception as e:
        logger.error(f"Error handling query for {sid}: {e}")
        await sio.emit('error', {
            'message': str(e)
        }, room=sid)


async def stream_audio_to_client(sid: str, text: str):
    """
    Generate audio from text and stream it to the client.
    Supports interruption - stops streaming if client sends new query.
    
    Args:
        sid: Client session ID
        text: Text to convert to audio
    """
    try:
        logger.info(f"🔊 Generating audio for client {sid}")
        
        # Signal start of audio stream
        await sio.emit('audio_start', {
            'message': 'Starting audio stream...',
            'text_length': len(text)
        }, room=sid)
        
        # Generate and stream audio chunks
        chunk_count = 0
        async for audio_chunk in tts_handler.text_to_speech_stream(text):
            # Check for interruption before sending each chunk
            if sid in active_responses and active_responses[sid]['interrupted']:
                logger.info(f"⚠️ Audio streaming interrupted for client {sid} at chunk {chunk_count}")
                await sio.emit('audio_interrupted', {
                    'message': 'Audio playback interrupted',
                    'chunks_sent': chunk_count
                }, room=sid)
                return
            
            if audio_chunk:
                # Encode audio chunk as base64 for transmission
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                await sio.emit('audio_chunk', {
                    'data': audio_b64,
                    'chunk_id': chunk_count,
                    'format': 's16le',  # 16-bit signed little-endian
                    'sample_rate': 24000
                }, room=sid)
                
                chunk_count += 1
        
        # Only send completion if not interrupted
        if sid in active_responses and not active_responses[sid]['interrupted']:
            # Signal end of audio stream
            await sio.emit('audio_end', {
                'message': 'Audio stream completed',
                'total_chunks': chunk_count
            }, room=sid)
            
            logger.info(f"🎵 Audio streaming completed for client {sid} ({chunk_count} chunks)")
        
    except asyncio.CancelledError:
        logger.info(f"⚠️ Audio streaming cancelled for client {sid}")
        await sio.emit('audio_interrupted', {
            'message': 'Audio playback cancelled'
        }, room=sid)
        raise
    except Exception as e:
        logger.error(f"Error streaming audio to client {sid}: {e}")
        await sio.emit('error', {
            'message': f'Audio streaming error: {str(e)}'
        }, room=sid)


# Health check endpoint
async def health(request):
    """Health check endpoint."""
    return web.Response(text='Text-to-Voice Socket.IO Server Running')

# Serve the HTML frontend
async def serve_frontend(request):
    """Serve the HTML frontend."""
    try:
        html_path = Path(__file__).parent / 'static' / 'text_to_voice.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    except FileNotFoundError:
        return web.Response(text='Frontend HTML file not found', status=404)

# Add routes
app.router.add_get('/', serve_frontend)
app.router.add_get('/health', health)


def check_environment():
    """Check if all required environment variables are set."""
    issues = []
    
    # Check OpenAI API key
    from config import config
    if not config.openai_api_key:
        issues.append("OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
    
    # Check Baseten API key
    if not os.environ.get("BASETEN_API_KEY"):
        issues.append("BASETEN_API_KEY is not set. Please set BASETEN_API_KEY environment variable.")
    
    # Check if data file exists
    if not os.path.exists(config.data_file_path):
        issues.append(f"Data file not found: {config.data_file_path}")
    
    return issues


def main(host='0.0.0.0', port=5000):
    """Start the Text-to-Voice Socket.IO server."""
    logger.info(f"🚀 Starting Text-to-Voice Socket.IO Server on {host}:{port}")
    logger.info("💬 Each client gets their own session")
    logger.info("🔊 Audio streams in real-time to browser")
    logger.info("🌐 CORS enabled for frontend integration")
    logger.info("-" * 50)
    
    web.run_app(app, host=host, port=port)


if __name__ == '__main__':
    try:
        # Check environment
        issues = check_environment()
        if issues:
            print("❌ Environment check failed:")
            for issue in issues:
                print(f"   • {issue}")
            sys.exit(1)
        
        # Start the Socket.IO server
        main(host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)