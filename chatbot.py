"""
Main chatbot interface that orchestrates the conversation flow.
"""
import logging
import sys
from typing import Optional, Dict, Any

from agent import ChatbotAgent
from session_manager import SessionManager, session_manager
from vectorstore.chromadb_client import ChromaDBClient
from config import config

logger = logging.getLogger(__name__)


class ChatBot:
    """Main chatbot interface that manages the conversation flow."""
    
    def __init__(self):
        """Initialize the chatbot with all required components."""
        try:
            # Initialize components
            self.chromadb_client = ChromaDBClient()
            self.session_manager = session_manager
            self.agent = ChatbotAgent(self.chromadb_client)
            
            # Check if ChromaDB has data
            if self.chromadb_client.is_collection_empty():
                logger.warning("ChromaDB collection is empty. Please run data initialization first.")
                print("\n⚠️  Warning: No data found in ChromaDB. Please run 'python initialise_data.py' first.")
                
            self.session_id: Optional[str] = None
            
            logger.info("ChatBot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {e}")
            raise
    
    def start_session(self) -> str:
        """
        Start a new chat session.
        
        Returns:
            Session ID
        """
        try:
            self.session_id = self.session_manager.create_session()
            logger.info(f"Started new session: {self.session_id}")
            
            # Display welcome message
            welcome_message = """
🤖 Privacy Policy Assistant

Hello! I'm here to help you understand TechGropse's privacy policy and answer any questions about how we handle your personal data.

You can ask me about:
- Data collection practices
- Cookie usage
- Your privacy rights
- Contact information
- Security measures

Type 'quit', 'exit', or 'bye' to end our conversation.
            """
            
            print(welcome_message)
            return self.session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise
    
    def end_session(self):
        """End the current session and clear cache."""
        try:
            if self.session_id:
                self.session_manager.clear_session(self.session_id)
                logger.info(f"Ended session: {self.session_id}")
                print("\n👋 Session ended. Thank you for using our privacy assistant!")
                self.session_id = None
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    def process_message(self, user_input: str) -> str:
        """
        Process a user message through the complete flow.
        
        Args:
            user_input: User's input message
            
        Returns:
            Bot response
        """
        try:
            if not self.session_id:
                raise ValueError("No active session. Please start a session first.")

            # Update session activity
            self.session_manager.update_session_activity(self.session_id)

            # Append user message to session history
            try:
                self.session_manager.append_message_to_history(self.session_id, 'user', user_input)
            except Exception:
                logger.debug("Unable to append message to history")

            # Step 1: Check cache first
            cached_response = self.session_manager.get_cached_response(self.session_id, user_input)

            if cached_response:
                # Dynamic acknowledgment referencing the previous topic
                try:
                    last_query = self.session_manager.get_last_user_query(self.session_id)
                except Exception:
                    last_query = None

                def make_ack(prev_q: Optional[str]) -> str:
                    import random
                    templates = [
                        "As I mentioned earlier,: ",
                        "To reiterate about, ",
                        "As we discussed before regarding, ",
                        "Following up on your earlier question about, "
                    ]
                    t = random.choice(templates)
                    if prev_q:
                        preview = prev_q.strip()
                        if len(preview) > 60:
                            preview = preview[:57] + '...'
                        return t.format(topic=preview)
                    return random.choice(["As I mentioned earlier, ", "As we discussed before, "])

                prefix = make_ack(last_query)
                final_cached = f"{prefix}{cached_response}"

                # Append bot response to history
                try:
                    self.session_manager.append_message_to_history(self.session_id, 'bot', final_cached)
                except Exception:
                    logger.debug("Unable to append cached bot response to history")

                logger.debug("Returning cached response with dynamic acknowledgment")
                return final_cached

            # Step 2: Process through agent (provide last user query for follow-ups)
            # Since we already appended current user_input to history, we need to skip it
            # to get the PREVIOUS question (skip_current=True)
            last_user = self.session_manager.get_last_user_query(self.session_id, skip_current=True)
            result = self.agent.process_user_input(user_input, last_user_query=last_user)

            response = result.get('response', '')
            intent = result.get('intent', '')
            needs_caching = result.get('needs_caching', True)

            # Step 3: Cache response if needed
            if needs_caching and intent == 'query' and len(response) > 50:
                # Only cache substantial query responses
                self.session_manager.cache_query_response(self.session_id, user_input, response)

            # Append bot response to history
            try:
                self.session_manager.append_message_to_history(self.session_id, 'bot', response)
            except Exception:
                logger.debug("Unable to append bot response to history")

            # Log interaction
            logger.info(f"Session {self.session_id}: Intent={intent}, Cached=False")

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        
        Returns:
            Session statistics
        """
        try:
            if not self.session_id:
                return {"error": "No active session"}
            
            session_info = self.session_manager.get_session_info(self.session_id)
            collection_info = self.chromadb_client.get_collection_info()
            
            return {
                "session_id": self.session_id,
                "session_info": session_info,
                "collection_info": collection_info
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}
    
    def run_interactive(self):
        """
        Run the chatbot in interactive mode.
        """
        # Start session and enter the interactive REPL loop.
        try:
            self.start_session()

            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n\n👋 Goodbye! Thanks for using our privacy assistant.")
                    break

                if not user_input:
                    # empty input, prompt again
                    continue

                # Normal processing
                try:
                    response = self.process_message(user_input)
                    
                    # Display response
                    print(f"\n🤖 Bot: {response}")
                    
                    # Check if this was a goodbye intent - if so, end the session and exit
                    # We can check this by looking at the last result intent
                    if user_input.lower() in ["quit", "exit", "bye", "goodbye", "end", "thank you", "thanks", "that's all"]:
                        break
                    
                    # Also check for goodbye patterns in the input
                    goodbye_patterns = ['goodbye', 'bye', 'see you', 'farewell', 'good night', 'take care']
                    if any(pattern in user_input.lower() for pattern in goodbye_patterns):
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    print(f"\n❌ Error: {e}")
                    continue

        finally:
            # Ensure session is cleared on exit
            try:
                self.end_session()
            except Exception:
                pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all components.
        
        Returns:
            Health status of all components
        """
        health_status = {
            "chatbot": "healthy",
            "session_manager": "unknown",
            "chromadb": "unknown",
            "agent": "unknown"
        }
        
        try:
            # Check session manager (Redis)
            test_session = self.session_manager.create_session()
            if test_session:
                self.session_manager.clear_session(test_session)
                health_status["session_manager"] = "healthy"
            else:
                health_status["session_manager"] = "unhealthy"
        except Exception as e:
            health_status["session_manager"] = f"unhealthy: {e}"
        
        try:
            # Check ChromaDB
            collection_info = self.chromadb_client.get_collection_info()
            if collection_info:
                health_status["chromadb"] = "healthy"
            else:
                health_status["chromadb"] = "unhealthy"
        except Exception as e:
            health_status["chromadb"] = f"unhealthy: {e}"
        
        try:
            # Check agent (basic classification test)
            test_intent = self.agent.classify_intent("hello")
            if test_intent:
                health_status["agent"] = "healthy"
            else:
                health_status["agent"] = "unhealthy"
        except Exception as e:
            health_status["agent"] = f"unhealthy: {e}"
        
        # Overall health
        unhealthy_components = [
            comp for comp, status in health_status.items() 
            if comp != "chatbot" and not status.startswith("healthy")
        ]
        
        if unhealthy_components:
            health_status["chatbot"] = f"partial: {', '.join(unhealthy_components)} unhealthy"
        
        return health_status
    
    def __del__(self):
        """Cleanup when chatbot is destroyed."""
        try:
            if hasattr(self, 'session_id') and self.session_id:
                self.end_session()
        except Exception as e:
            logger.error(f"Error in chatbot cleanup: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        chatbot = ChatBot()
        
        # Check if we should run health check
        if len(sys.argv) > 1 and sys.argv[1] == "--health":
            health = chatbot.health_check()
            print("\n🏥 Health Check Results:")
            for component, status in health.items():
                status_emoji = "✅" if status == "healthy" else "❌"
                print(f"{status_emoji} {component}: {status}")
        else:
            # Run interactive mode
            chatbot.run_interactive()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)