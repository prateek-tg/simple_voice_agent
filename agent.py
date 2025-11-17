"""
CrewAI agent for intent classification, greeting handling, and query processing.
"""
import logging
from typing import Dict, Any, List
from enum import Enum

from crewai import Agent
from langchain_openai import ChatOpenAI

from config import config
from vectorstore.chromadb_client import ChromaDBClient

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Intent classification types."""
    GREETING = "greeting"
    FOLLOWUP = "followup"
    QUERY = "query"
    GOODBYE = "goodbye"
    UNCLEAR = "unclear"


class ChatbotAgent:
    """CrewAI-powered agent for handling user intents and generating responses."""
    
    def __init__(self, chromadb_client: ChromaDBClient):
        """
        Initialize the chatbot agent.
        
        Args:
            chromadb_client: ChromaDB client for document retrieval
        """
        self.chromadb_client = chromadb_client
        
        # Disable CrewAI telemetry and traces
        import os
        os.environ['CREWAI_TELEMETRY'] = 'false'
        os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.7,  # Increased for more creative, human-like responses
            openai_api_key=config.openai_api_key
        )
        
        # Create the intent classification agent (for reference, not used in new implementation)
        self.intent_agent = Agent(
            role='Alicia - TechGropse Privacy Policy Assistant',
            goal='As Alicia, represent TechGropse and help users understand our privacy policy and data practices',
            backstory="""You are Alicia, an official representative of TechGropse, speaking on behalf of the company. 
            You help users understand TechGropse's privacy policy, data collection practices, cookies usage, 
            and user rights. You always speak as 'we' when referring to TechGropse (e.g., 'we collect', 
            'our privacy policy', 'we use'). You are friendly, professional, and knowledgeable about 
            TechGropse's privacy practices. When greeting users, you introduce yourself as Alicia from TechGropse.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
    
    def classify_intent(self, user_input: str) -> IntentType:
        """
        Classify the user's intent using LLM.
        
        Args:
            user_input: User's message
            
        Returns:
            Classified intent type
        """
        try:
            # Use LLM to classify intent
            prompt = f"""You are a customer service assistant. Classify the user's intent into ONE of these categories:

1. GREETING - user is saying hello, hi, or starting the conversation
2. FOLLOWUP - user wants more details/information about the previous topic (phrases like "more info", "tell me more", "need more details", "elaborate", etc.)
3. QUERY - user is asking a specific question about privacy policy, data collection, cookies, etc.
4. GOODBYE - user is ending the conversation. This includes ANY expression of thanks, satisfaction, or ending like: "thank you", "thanks", "ok thank you", "that's all", "goodbye", "bye", "see you", "appreciate it", "perfect", "great thanks", "awesome", "all good", "done", "finished", etc.
5. UNCLEAR - unclear or ambiguous input

User input: "{user_input}"

Be VERY sensitive to goodbye hints - if there's ANY indication the user is satisfied or ending the conversation, classify as GOODBYE.

Respond with ONLY the category name (GREETING, FOLLOWUP, QUERY, GOODBYE, or UNCLEAR):"""

            response = self.llm.invoke(prompt)
            intent_text = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            # Map to IntentType
            if 'GREETING' in intent_text:
                return IntentType.GREETING
            elif 'FOLLOWUP' in intent_text:
                return IntentType.FOLLOWUP
            elif 'QUERY' in intent_text:
                return IntentType.QUERY
            elif 'GOODBYE' in intent_text:
                return IntentType.GOODBYE
            else:
                return IntentType.UNCLEAR
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            # Fallback to simple heuristic
            user_input_lower = user_input.lower().strip()
            if any(word in user_input_lower for word in ['hi', 'hello', 'hey']):
                return IntentType.GREETING
            elif any(phrase in user_input_lower for phrase in [
                'bye', 'goodbye', 'good bye', 'see you', 'thanks', 'thank you', 
                'ok thank you', 'okay thank you', 'that\'s all', 'thats all',
                'end', 'quit', 'exit', 'done', 'finished', 'all good',
                'appreciate it', 'perfect', 'great thanks', 'awesome thanks'
            ]):
                return IntentType.GOODBYE
            return IntentType.QUERY
    
    def handle_greeting(self) -> str:
        """
        Generate a greeting response using LLM.
        
        Returns:
            Greeting message
        """
        try:
            prompt = """You are Alicia, an official representative of TechGropse, speaking on behalf of the company.
A user just greeted you. Respond warmly and naturally, introducing yourself as Alicia from TechGropse 
and letting them know you're here to help with our privacy policy questions. Always use 'we', 'our', 
and 'us' when referring to TechGropse. Keep it conversational and welcoming (2-3 sentences max).

Generate a natural greeting response as Alicia:"""

            response = self.llm.invoke(prompt)
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            return "Hello! I'm Alicia from TechGropse, and I'm here to help you with any questions about our privacy policy. What would you like to know?"
    
    def handle_goodbye(self) -> str:
        """
        Generate a goodbye response using LLM.
        
        Returns:
            Goodbye message
        """
        try:
            prompt = """You are Alicia, an official representative of TechGropse, speaking on behalf of the company.
A user is ending the conversation. Say goodbye warmly and professionally as Alicia, thanking them for their 
interest in our privacy practices. Always use 'we', 'our', and 'us' when referring to TechGropse.
Keep it natural and brief (1-2 sentences).

Generate a natural goodbye response as Alicia:"""

            response = self.llm.invoke(prompt)
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
        except Exception as e:
            logger.error(f"Error generating goodbye: {e}")
            return "Thank you for your questions! If you need any more information, feel free to ask anytime. Have a great day!"
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from ChromaDB.
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            results = self.chromadb_client.search_similar_documents(query, n_results)

            # Filter results with reasonable similarity threshold
            filtered_results = []
            for result in results:
                # Only include results with reasonable similarity (distance < 1.5)
                if result.get('distance', 0) < 1.5:
                    filtered_results.append(result)

            logger.debug(f"Retrieved {len(filtered_results)} relevant documents for query")
            return filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_response_from_context(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on query and retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved document contexts
            
        Returns:
            Generated response
        """
        try:
            # Prepare context from retrieved documents
            context_text = ""
            if context_docs:
                context_chunks = []
                for i, doc in enumerate(context_docs):
                    chunk = doc['content'].strip()
                    if chunk:
                        context_chunks.append(f"Context {i+1}: {chunk}")
                context_text = "\n\n".join(context_chunks)
            
            if not context_text:
                # No context found - provide general guidance
                return """I don't have specific information about that topic in our privacy policy. 
                For detailed information about our privacy practices, please contact us directly at sales@techgropse.com 
                or visit our contact page. We'll be happy to assist you with any privacy-related questions."""
            
            # Use LLM directly for better control
            prompt = f"""You are Alicia, an official representative of TechGropse, speaking on behalf of the company about our privacy policy.

User Question: {query}

Relevant Information from Privacy Policy:
{context_text}

Instructions:
1. You are Alicia from TechGropse - always use "we", "our", "us" when referring to the company (e.g., "we collect", "our privacy policy", "we use")
2. Answer naturally and conversationally, like a helpful human would - don't be overly formal or robotic
3. Use the information provided, but explain it in a friendly, easy-to-understand way
4. You can use casual phrases like "So basically...", "Here's what that means...", "Let me explain...", "In simple terms..."
5. Break down complex information into digestible pieces
6. Use bullet points when listing multiple items, but wrap them in natural language
7. If something isn't fully covered, say so naturally without immediately suggesting they contact support
8. Be warm and personable while staying professional
9. Remember you're Alicia speaking FOR TechGropse, not ABOUT TechGropse
10. DO NOT include greetings like "Hi there!", "Hello!", "Hey!" - jump straight into answering the question
11. Start directly with acknowledgment or information, not with greeting phrases

Provide a natural, conversational response as Alicia representing TechGropse (NO greetings):"""

            # Get response from LLM directly
            response_text = None
            try:
                # Try different invocation methods
                if hasattr(self.llm, 'predict_messages'):
                    response_obj = self.llm.predict_messages([{"role": "user", "content": prompt}])
                    if isinstance(response_obj, list) and hasattr(response_obj[0], 'content'):
                        response_text = response_obj[0].content
                    else:
                        response_text = str(response_obj)
                elif hasattr(self.llm, 'generate'):
                    messages = [{"role": "user", "content": prompt}]
                    gen = self.llm.generate(messages)
                    try:
                        response_text = gen.generations[0][0].text
                    except Exception:
                        response_text = str(gen)
                else:
                    # Fallback to invoke
                    response = self.llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.debug(f"LLM call failed, trying fallback: {e}")
                try:
                    response = self.llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                except Exception as e2:
                    logger.error(f"All LLM invocation methods failed: {e2}")
                    response_text = None
            
            # Clean up the response
            if response_text:
                response_text = response_text.strip()
                
                # Remove greeting phrases that might have slipped through
                greeting_phrases = [
                    "Hi there!", "Hello!", "Hey!", "Hi!", "Hey there!",
                    "Good morning!", "Good afternoon!", "Good evening!",
                    "Greetings!", "Hello there!"
                ]
                
                for greeting in greeting_phrases:
                    if response_text.startswith(greeting):
                        response_text = response_text[len(greeting):].strip()
                        break
            
            # Ensure we have a reasonable response
            if not response_text or len(response_text) < 20:
                response_text = """I'd be happy to help with your privacy-related question. 
                For specific details about our privacy practices, please feel free to ask more specific questions 
                or contact us directly at sales@techgropse.com."""
            
            # Add engagement prompt at the end of detailed responses
            if response_text and len(response_text) > 100:  # Only for substantial responses
                engagement_prompts = [
                    "\n\nIs there anything else you'd like to know about this topic, or do you have other privacy questions?",
                    "\n\nWould you like more information on this, or do you have any other questions about our privacy practices?",
                    "\n\nDo you need more details about this, or is there something else regarding our privacy policy you'd like to know?",
                    "\n\nIs there anything else about this topic you'd like me to clarify, or do you have other privacy-related questions?"
                ]
                import random
                response_text += random.choice(engagement_prompts)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return """I apologize, but I'm having trouble processing your request right now. 
            For immediate assistance with privacy questions, please contact us at sales@techgropse.com."""
    
    def _generate_followup_response(self, original_query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a follow-up response that provides more detailed information.
        
        Args:
            original_query: The original question being followed up on
            context_docs: Retrieved document contexts
            
        Returns:
            Generated detailed response
        """
        try:
            # Prepare context from retrieved documents
            context_text = ""
            if context_docs:
                context_chunks = []
                for i, doc in enumerate(context_docs):
                    chunk = doc['content'].strip()
                    if chunk:
                        context_chunks.append(f"Context {i+1}: {chunk}")
                context_text = "\n\n".join(context_chunks)
            
            if not context_text:
                return """I don't have additional information on that topic in our privacy policy. 
                For more detailed information, please contact us directly at sales@techgropse.com."""
            
            # Use LLM to generate a more comprehensive response
            prompt = f"""You are Alicia, an official representative of TechGropse, speaking on behalf of the company about our privacy policy.

The user previously asked: "{original_query}"

They want MORE DETAILS about this topic now.

Relevant Information from Privacy Policy:
{context_text}

Instructions:
1. You are Alicia from TechGropse - always use "we", "our", "us" when referring to the company
2. Give them a more thorough, comprehensive answer in a natural, conversational tone
3. Say things like "Sure, let me give you more details..." or "Absolutely, here's a deeper dive..."
4. Include all the relevant specifics from the context
5. Break it down clearly with examples or explanations where helpful
6. Be thorough but keep it conversational - like explaining to a friend
7. Only mention contacting support if the info really isn't there
8. Remember you're Alicia speaking FOR TechGropse, not ABOUT TechGropse
9. DO NOT include greetings like "Hi there!", "Hello!" - start directly with the detailed information

Provide a detailed, friendly response as Alicia representing TechGropse (NO greetings):"""

            # Get response from LLM
            response_text = None
            try:
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Error invoking LLM for followup: {e}")
                return """I'd be happy to provide more information. For specific details, please contact us at sales@techgropse.com."""
            
            # Clean up the response
            if response_text:
                response_text = response_text.strip()
                
                # Remove greeting phrases that might have slipped through
                greeting_phrases = [
                    "Hi there!", "Hello!", "Hey!", "Hi!", "Hey there!",
                    "Good morning!", "Good afternoon!", "Good evening!",
                    "Greetings!", "Hello there!"
                ]
                
                for greeting in greeting_phrases:
                    if response_text.startswith(greeting):
                        response_text = response_text[len(greeting):].strip()
                        break
            
            if not response_text or len(response_text) < 20:
                response_text = """We'd be happy to provide more information. For specific details about our privacy practices, 
                please feel free to ask a more specific question or contact us directly at sales@techgropse.com."""
            
            # Add engagement prompt at the end of detailed follow-up responses
            if response_text and len(response_text) > 100:  # Only for substantial responses
                engagement_prompts = [
                    "\n\nDoes this answer your question completely, or would you like even more details about any specific aspect?",
                    "\n\nIs this the level of detail you were looking for, or would you like me to elaborate on any particular point?",
                    "\n\nDoes this cover what you wanted to know, or do you have follow-up questions about any of these points?",
                    "\n\nI hope this gives you the detail you were looking for! Is there anything specific you'd like me to expand on further?"
                ]
                import random
                response_text += random.choice(engagement_prompts)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating followup response: {e}")
            return """We apologize, but I'm having trouble retrieving more details right now. 
            For comprehensive information, please contact us at sales@techgropse.com."""
    
    def process_user_input(self, user_input: str, last_user_query: str = None) -> Dict[str, Any]:
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input: User's input message
            last_user_query: Optional previous user query for follow-up handling
            
        Returns:
            Dictionary with intent, response, and context information
        """
        try:
            # Classify intent
            intent = self.classify_intent(user_input)
            
            result = {
                'intent': intent.value,
                'user_input': user_input,
                'response': '',
                'context_docs': [],
                'needs_caching': True
            }
            
            # Handle based on intent
            if intent == IntentType.GREETING:
                result['response'] = self.handle_greeting()
                result['needs_caching'] = False  # Don't cache greetings
                
            elif intent == IntentType.GOODBYE:
                result['response'] = self.handle_goodbye()
                result['needs_caching'] = False  # Don't cache goodbyes
                
            elif intent == IntentType.QUERY:
                # Retrieve relevant documents
                context_docs = self.retrieve_relevant_documents(user_input)
                result['context_docs'] = context_docs
                
                # Generate response based on context
                result['response'] = self.generate_response_from_context(user_input, context_docs)
                
            elif intent == IntentType.FOLLOWUP:
                    # For follow-ups, we need the previous substantive question (not the current "need more info")
                    # The current user_input is already in history, so we need to skip it and get the one before
                    if last_user_query and last_user_query.lower().strip() == user_input.lower().strip():
                        # If last_user_query is the same as current input, we need to go back further
                        # This shouldn't happen with proper logic, but let's handle it
                        expanded_query = user_input
                    elif last_user_query:
                        # Construct a query that references the previous question and asks for more detail
                        expanded_query = f"{last_user_query} - provide more detailed information and additional context"
                    else:
                        expanded_query = user_input

                    # Get more results for follow-ups to provide comprehensive answers
                    context_docs = self.retrieve_relevant_documents(expanded_query, n_results=6)
                    result['context_docs'] = context_docs
                    
                    # Generate a more detailed response with explicit instruction to expand
                    result['response'] = self._generate_followup_response(last_user_query or user_input, context_docs)
                
            else:  # UNCLEAR intent
                result['response'] = """I'm not sure I understand your question. Could you please rephrase it? 
                I'm here to help with questions about our privacy policy, data collection, cookies, and your rights."""
                result['needs_caching'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {
                'intent': 'error',
                'user_input': user_input,
                'response': "We're sorry, I encountered an error processing your request. Please try again or contact us for support.",
                'context_docs': [],
                'needs_caching': False
            }