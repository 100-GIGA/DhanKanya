"""
Home page component for the DhanKanya application.

This module contains the UI components for the main home page,
which includes the chat interface for interacting with the AI assistant.
Now includes comprehensive voice interaction capabilities via WebSocket backend.
"""

import streamlit as st
import streamlit.components.v1 as components
import anthropic
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import re
import asyncio
import google.generativeai as genai

from app.services.ai_service import get_response, get_response_with_sources, process_voice_query, detect_language_from_text
from config.settings import SUPPORTED_VOICE_LANGUAGES, VOICE_ENABLED

# Constants
HINDI_FONT = "Noto Sans Devanagari"
ENGLISH_FONT = "Inter"
MAX_MESSAGE_LENGTH = 1024

# Supported Indian languages and their font mappings
INDIAN_LANGUAGES = {
    'hi': 'Noto Sans Devanagari',  # Hindi
    'mr': 'Noto Sans Devanagari',  # Marathi
    'ta': 'Noto Sans Tamil',       # Tamil
    'te': 'Noto Sans Telugu',      # Telugu
    'bn': 'Noto Sans Bengali',     # Bengali
    'gu': 'Noto Sans Gujarati',    # Gujarati
    'pa': 'Noto Sans Gurmukhi',    # Punjabi
    'ml': 'Noto Sans Malayalam',   # Malayalam
    'kn': 'Noto Sans Kannada',     # Kannada
    'en': 'Inter'                  # English
}

def detect_indian_language(text: str) -> Tuple[str, str]:
    """
    Detect if the text is in a supported Indian language or English.
    Uses a simple pattern-based approach to identify script.
    
    Args:
        text: The text to check
        
    Returns:
        Tuple[str, str]: A tuple containing (language_code, font_family)
        Defaults to English if no supported language is detected
    """
    # Devanagari Unicode range (Hindi, Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        return 'hi', INDIAN_LANGUAGES['hi']
    
    # Bengali Unicode range
    if re.search(r'[\u0980-\u09FF]', text):
        return 'bn', INDIAN_LANGUAGES['bn']
    
    # Gurmukhi Unicode range (Punjabi)
    if re.search(r'[\u0A00-\u0A7F]', text):
        return 'pa', INDIAN_LANGUAGES['pa']
    
    # Gujarati Unicode range
    if re.search(r'[\u0A80-\u0AFF]', text):
        return 'gu', INDIAN_LANGUAGES['gu']
    
    # Tamil Unicode range
    if re.search(r'[\u0B80-\u0BFF]', text):
        return 'ta', INDIAN_LANGUAGES['ta']
    
    # Telugu Unicode range
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'te', INDIAN_LANGUAGES['te']
    
    # Kannada Unicode range
    if re.search(r'[\u0C80-\u0CFF]', text):
        return 'kn', INDIAN_LANGUAGES['kn']
    
    # Malayalam Unicode range
    if re.search(r'[\u0D00-\u0D7F]', text):
        return 'ml', INDIAN_LANGUAGES['ml']
    
    # Default to English for any other script
    return 'en', INDIAN_LANGUAGES['en']

def render_sources(sources: List[Dict[str, Any]]) -> None:
    """
    Render source links from Linkup API in a Perplexity-style inline layout with dark theme.
    
    Args:
        sources: List of source dictionaries from Linkup
    """
    if not sources:
        return
    
    # Create inline numbered source badges like Perplexity using Streamlit columns
    st.markdown("**Sources:**")
    
    # Create horizontal layout for source badges
    cols = st.columns(min(len(sources), 8))  # Max 8 sources per row
    for i, source in enumerate(sources):
        if i < 8:  # Only show first 8 sources inline
            source_num = i + 1
            name = source.get("name", "Unknown Source")
            url = source.get("url", "")
            
            with cols[i]:
                if url:
                    st.markdown(f"[**{source_num}**]({url})", help=name)
                else:
                    st.markdown(f"**{source_num}**")
    
    # Add an expandable section for detailed source information
    with st.expander("📚 View detailed sources", expanded=False):
        for i, source in enumerate(sources):
            source_num = i + 1
            name = source.get("name", "Unknown Source")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            
            # Create source card using Streamlit container
            with st.container():
                st.markdown(f"**{source_num}. {name}**")
                
                if snippet:
                    st.caption(snippet)
                
                if url:
                    st.markdown(f"🔗 [Visit Source]({url})")
                else:
                    st.caption("No link available")
                
                if i < len(sources) - 1:  # Add separator if not last item
                    st.markdown("---")

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = "claude-3-haiku-20240307"
    if "voice_mode_enabled" not in st.session_state:
        st.session_state.voice_mode_enabled = False
    if "voice_backend_connected" not in st.session_state:
        st.session_state.voice_backend_connected = False

def render_message(message: Dict[str, Any]) -> None:
    """
    Render a single chat message with appropriate styling and voice features.
    
    Args:
        message: Dictionary containing message data
    """
    with st.chat_message(message["role"]):
        font_family = message.get("font_family", ENGLISH_FONT)
        
        # Check if this is a voice message
        is_voice = message.get("is_voice", False)
        voice_lang = message.get("voice_language", "en")
        message_type = message.get("message_type", "text")
        has_audio = message.get("has_audio", False)
        
        # Show voice/message type indicators
        if is_voice:
            lang_display = SUPPORTED_VOICE_LANGUAGES.get(voice_lang, "English")
            
            if message_type == "audio":
                st.markdown(f"🔊 **Audio Message** ({lang_display})")
                
                # Show audio player if audio data is available
                if "audio_data" in message:
                    try:
                        import base64
                        audio_bytes = base64.b64decode(message["audio_data"])
                        st.audio(audio_bytes, format="audio/wav", sample_rate=message.get("sample_rate", 24000))
                    except Exception as e:
                        st.caption(f"Audio playback error: {e}")
                
            elif message_type == "input_transcription":
                st.markdown(f"📝 **Input Transcription** ({lang_display})")
            elif message_type == "output_transcription":
                st.markdown(f"📝 **Output Transcription** ({lang_display})")
            else:
                st.markdown(f"🎤 **Voice Message** ({lang_display})")
        
        # Show timestamp if available
        if "timestamp" in message and message["timestamp"]:
            st.caption(f"⏰ {message['timestamp']}")
        
        # Render the message content with appropriate font
        content = message.get("content", "")
        if content:
            st.markdown(
                f'<div style="font-family: {font_family}; line-height: 1.6; font-size: 1.1em;">{content}</div>',
                unsafe_allow_html=True
            )
        
        # Display sources if available
        if message.get("sources"):
            render_sources(message["sources"])

def render_header() -> None:
    """Render the header section with logo and welcome message."""
    # Title with custom styling
    st.markdown("# DhanKanya: Financial Empowerment for Girls in India")
    
    # Center the logo using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./assets/images/logo.png", width=200)
    
    # Welcome message
    st.markdown("### Welcome to our AI-powered financial literacy application!")
    st.markdown("Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.")

def render_features() -> None:
    """Render the features section using Streamlit's native components."""
    st.markdown("### Key Features")
    
    # Create three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Interactive Budgeting")
        st.markdown("Track your income and expenses with our user-friendly tools")
    
    with col2:
        st.markdown("#### Educational Resources")
        st.markdown("Learn essential financial literacy concepts like saving and investing")
    
    with col3:
        st.markdown("#### Goal Setting")
        st.markdown("Plan and save for specific educational milestones")

def handle_text_input(prompt: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel]) -> None:
    """
    Handle text input from the user.
    
    Args:
        prompt: The user's text prompt
        provider: The LLM provider name
        client: The initialized LLM client
    """
    if st.session_state.is_processing:
        return
        
    st.session_state.is_processing = True
    
    # Detect language and font family for the input text
    lang_code, font_family = detect_indian_language(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "font_family": font_family,
        "lang_code": lang_code,
        "is_voice": False
    })
    
    # Display a loading spinner while getting the response
    with st.spinner("Thinking of a helpful response for you..."):
        # Get AI response with sources using the same language context
        response, sources = get_response_with_sources(prompt, provider, client, lang_code)
    
    # Add assistant message with the same font family and sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "font_family": font_family,
        "lang_code": lang_code,
        "is_voice": False,
        "sources": sources  # Add sources to message
    })
    
    st.session_state.is_processing = False
    st.rerun()

def render_integrated_voice_interface():
    """Render an integrated voice interface exactly like simple_voice_interface.py"""
    
    st.markdown("## 🎤 DhanKanya Voice Assistant")
    st.markdown("Simple trigger-based recording - backend handles all audio processing")
    
    # Use full width without any container constraints
    # Trigger-based voice interface with responsive dark design
    audio_interface_html = """
    <style>
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #1a1a1a;
        color: #ffffff;
        line-height: 1.6;
    }
    
    /* Main container - optimized width design */
    .main-voice-container {
        width: 80%;
        max-width: none;
        margin: 0 0 0 auto;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        min-height: 100vh;
        position: relative;
        border-radius: 15px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .main-voice-container {
            padding: 15px;
            margin: 0 0 0 auto;
            width: 85%;
        }
    }
    
    @media (max-width: 480px) {
        .main-voice-container {
            padding: 10px;
            margin: 0 0 0 auto;
            width: 100%;
        }
    }
    
    /* Status display */
    .status-display {
        font-size: clamp(1rem, 2.5vw, 1.25rem);
        font-weight: bold;
        margin: 20px auto;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        text-align: center;
        width: 100%;
        max-width: none;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
        transition: transform 0.3s ease;
    }
    
    .status-display:hover {
        transform: translateY(-2px);
    }
    
    /* Connection status */
    .connection-status {
        display: inline-block;
        padding: 12px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: clamp(0.875rem, 2vw, 1rem);
        margin: 0 auto;
        text-align: center;
        min-width: 150px;
        transition: all 0.3s ease;
    }
    
    .connected {
        background: #10b981;
        color: white;
    }
    
    .disconnected {
        background: #ef4444;
        color: white;
    }
    
    .connecting {
        background: #f59e0b;
        color: white;
    }
    
    /* Control panel - responsive flex layout */
    .control-panel {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin: 30px 0;
        flex-wrap: wrap;
        align-items: center;
    }
    
    @media (max-width: 768px) {
        .control-panel {
            flex-direction: column;
            gap: 15px;
        }
    }
    
    /* Buttons - responsive sizing */
    .control-btn {
        padding: clamp(12px, 2.5vw, 18px) clamp(20px, 5vw, 40px);
        border: none;
        border-radius: 50px;
        cursor: pointer;
        font-size: clamp(0.875rem, 2.5vw, 1.125rem);
        font-weight: bold;
        transition: all 0.3s ease;
        min-width: clamp(140px, 25vw, 180px);
        transform: scale(1);
        text-align: center;
    }
    
    .control-btn:hover:not(:disabled) {
        transform: scale(1.05);
    }
    
    .control-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none !important;
    }
    
    /* Button color schemes */
    .start-btn {
        background: linear-gradient(45deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .start-btn:hover:not(:disabled) {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    .stop-btn {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .stop-btn:hover:not(:disabled) {
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
    }
    
    .new-session-btn {
        background: linear-gradient(45deg, #6366f1, #4f46e5);
        color: white;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .new-session-btn:hover:not(:disabled) {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Chat container - enhanced scrollability and larger area */
    .chat-container {
        max-height: 85vh;
        min-height: 500px;
        overflow-y: auto;
        overflow-x: hidden;
        margin: 20px auto;
        padding: 25px;
        background: rgba(55, 65, 81, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(75, 85, 99, 0.3);
        text-align: left;
        color: #f3f4f6;
        width: 100%;
        max-width: none;
        scrollbar-width: thin;
        scrollbar-color: #4b5563 #374151;
        scroll-behavior: smooth;
        position: relative;
    }
    
    /* Custom scrollbar for webkit browsers - enhanced visibility */
    .chat-container::-webkit-scrollbar {
        width: 12px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #374151;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 6px;
        border: 2px solid #374151;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    
    .chat-container::-webkit-scrollbar-thumb:active {
        background: #9ca3af;
    }
    
    @media (max-width: 768px) {
        .chat-container {
            padding: 20px;
            max-height: 80vh;
            min-height: 450px;
            margin: 15px auto;
        }
    }
    
    @media (max-width: 480px) {
        .chat-container {
            padding: 15px;
            max-height: 75vh;
            min-height: 400px;
            margin: 10px auto;
        }
    }
    
    /* Message styles - enhanced for better readability */
    .message {
        margin: 20px 0;
        padding: 18px;
        border-radius: 12px;
        line-height: 1.6;
        word-wrap: break-word;
        animation: fadeIn 0.3s ease;
        max-width: 100%;
        overflow-wrap: break-word;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        border-left: 4px solid #3b82f6;
        color: #ffffff;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #7c2d12, #9a3412);
        border-left: 4px solid #ea580c;
        color: #ffffff;
    }
    
    .transcription-message {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-left: 4px solid #10b981;
        color: #ffffff;
    }
    
    .error-message {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 4px solid #ef4444;
        color: #ffffff;
    }
    
    /* Responsive text sizing */
    .message {
        font-size: clamp(0.875rem, 2vw, 1rem);
    }
    
    .message strong {
        font-size: clamp(0.9375rem, 2.2vw, 1.125rem);
    }
    
    /* Empty state styling */
    .empty-chat {
        text-align: center;
        color: #9ca3af;
        font-style: italic;
        padding: 40px;
        font-size: clamp(0.875rem, 2vw, 1rem);
    }
    
    /* Loading indicator */
    .loading {
        opacity: 0.7;
    }
    
    .loading::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60% { content: '...'; }
        80%, 100% { content: ''; }
    }
    </style>
    </style>
    
    <div class="main-voice-container">
        
        <div style="margin: 30px 0; text-align: center;">
            <div class="connection-status connecting" id="connectionStatus">
                🔄 Connecting...
            </div>
        </div>
        
        <div style="margin: 20px 0;">
            <div class="status-display" id="status">
                Ready to connect to DhanKanya backend...
            </div>
        </div>
        
        <div style="margin: 30px 0;">
            <div class="control-panel">
                <button class="control-btn start-btn" id="startBtn" onclick="startRecording()" disabled>
                    🎤 Start Recording
                </button>
                <button class="control-btn new-session-btn" id="newSessionBtn" onclick="startNewSession()" disabled>
                    🔄 Start New Session
                </button>
            </div>
        </div>
        
        <div style="margin: 20px auto;">
            <div class="chat-container" id="chatContainer">
                <div class="empty-chat">
                    💬 Your conversation with DhanKanya will appear here...
                </div>
            </div>
        </div>
    </div>
    
    <script>
    let websocket = null;
    let isRecording = false;
    let isConnected = false;
    
    // Get DOM elements
    const status = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const newSessionBtn = document.getElementById('newSessionBtn');
    const chatContainer = document.getElementById('chatContainer');
    const connectionStatus = document.getElementById('connectionStatus');
    
    function updateConnectionStatus(status_text, isConnected) {
        connectionStatus.textContent = status_text;
        connectionStatus.className = 'connection-status ' + (isConnected ? 'connected' : 'disconnected');
    }
    
    function addChatMessage(type, text) {
        // Only show errors and transcriptions as requested
        if (type !== 'error' && type !== 'input_transcription' && type !== 'output_transcription') {
            return;
        }
        
        // Clear empty state message if it exists
        const emptyState = chatContainer.querySelector('.empty-chat');
        if (emptyState) {
            emptyState.remove();
        }
        
        // Create new message element
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        
        let messageClass = '';
        let icon = '';
        let label = '';
        
        switch(type) {
            case 'input_transcription':
                messageClass = 'user-message';
                icon = '🎙️';
                label = 'You said';
                break;
            case 'output_transcription':
                messageClass = 'transcription-message';
                icon = '💬';
                label = 'DhanKanya said';
                break;
            case 'error':
                messageClass = 'error-message';
                icon = '❌';
                label = 'Error';
                break;
            default:
                return; // Don't show other message types
        }
        
        messageDiv.className += ' ' + messageClass;
        messageDiv.innerHTML = `
            <strong>${icon} ${label}:</strong>
            <div class="message-content">${text}</div>
        `;
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        console.log(`💬 Added ${type} message: "${text}"`);
    }
    
    function clearChat() {
        chatContainer.innerHTML = '<div class="empty-chat">💬 Your conversation with DhanKanya will appear here...</div>';
    }
    
    function startNewSession() {
        console.log('🔄 Starting new session');
        
        // Stop any playing audio first
        stopAllAudio();
        
        // Close existing WebSocket connection
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.close();
        }
        
        // Reset all state
        isRecording = false;
        isConnected = false;
        
        // Reset UI
        startBtn.disabled = true;
        newSessionBtn.disabled = true;
        
        // Clear chat
        clearChat();
        
        // Update status
        status.textContent = 'Starting new session...';
        status.style.color = '#ff9800';
        
        // Add status message
        addChatMessage('status', 'Starting new session... Please wait.');
        
        // Reconnect after a brief delay
        setTimeout(() => {
            connectWebSocket();
        }, 1000);
    }
    
    function closeVoiceInterface() {
        // This function can be used to close the interface if needed
        console.log('Closing voice interface');
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.close();
        }
    }
    
    // Global audio management for smooth sequential playback
    let audioContext = null;
    let audioQueue = [];
    let isPlayingAudio = false;
    let currentAudioSource = null;
    let nextPlayTime = 0;
    
    async function initAudioContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('🎵 Audio context initialized for smooth sequential playback');
        }
        
        // Resume context if suspended (required for many browsers)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
    }
    
    function playNextAudio() {
        if (audioQueue.length === 0) {
            isPlayingAudio = false;
            nextPlayTime = 0;
            return;
        }
        
        if (isPlayingAudio) {
            return; // Already playing, wait for current to finish
        }
        
        const audioBuffer = audioQueue.shift();
        isPlayingAudio = true;
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        
        // Apply gain for consistent volume
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0.8; // Slightly reduce volume for consistency
        
        // Connect: source -> gain -> destination
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Calculate precise timing for seamless playback
        const startTime = Math.max(audioContext.currentTime, nextPlayTime);
        nextPlayTime = startTime + audioBuffer.duration;
        
        // Store reference to current source
        currentAudioSource = source;
        
        source.onended = () => {
            console.log('🎵 Audio chunk finished, playing next...');
            isPlayingAudio = false;
            currentAudioSource = null;
            
            // Immediately try to play next chunk for seamless flow
            setTimeout(() => playNextAudio(), 0);
        };
        
        source.start(startTime);
        console.log(`🎵 Playing audio buffer smoothly, duration: ${audioBuffer.duration.toFixed(2)}s, ${audioQueue.length} remaining in queue`);
    }
    
    function stopAllAudio() {
        // Stop current audio if playing
        if (currentAudioSource) {
            try {
                currentAudioSource.stop();
                currentAudioSource = null;
            } catch (e) {
                console.log('Audio source already stopped');
            }
        }
        
        // Clear the queue
        audioQueue = [];
        isPlayingAudio = false;
        nextPlayTime = 0;
        console.log('🔇 All audio stopped and queue cleared');
    }
    
    function playAudioResponse(audioData, mimeType) {
        try {
            // Initialize audio context if needed
            initAudioContext();
            
            // Validate input data
            if (!audioData || audioData.length === 0) {
                console.error('Empty audio data received');
                return;
            }
            
            // Decode base64 audio data with validation
            let binaryData;
            try {
                binaryData = atob(audioData);
            } catch (e) {
                console.error('Invalid base64 audio data:', e);
                return;
            }
            
            const uint8Array = new Uint8Array(binaryData.length);
            for (let i = 0; i < binaryData.length; i++) {
                uint8Array[i] = binaryData.charCodeAt(i);
            }
            
            // Create audio blob with specific MIME type for better compatibility
            const mimeTypeToUse = mimeType || 'audio/wav';
            const audioBlob = new Blob([uint8Array], { type: mimeTypeToUse });
            
            // Convert to ArrayBuffer and decode with enhanced error handling
            audioBlob.arrayBuffer().then(arrayBuffer => {
                // Ensure we have a valid ArrayBuffer
                if (!arrayBuffer || arrayBuffer.byteLength === 0) {
                    throw new Error('Empty audio buffer received');
                }
                
                // Check if this looks like a WAV file (starts with RIFF header)
                const headerView = new Uint8Array(arrayBuffer.slice(0, 4));
                const isWav = String.fromCharCode(...headerView) === 'RIFF';
                
                if (isWav) {
                    console.log('🎵 Processing WAV audio format');
                } else {
                    console.log('🎵 Processing PCM/other audio format');
                }
                
                return audioContext.decodeAudioData(arrayBuffer);
            }).then(audioBuffer => {
                // Validate the decoded audio buffer
                if (!audioBuffer || audioBuffer.length === 0) {
                    throw new Error('Failed to decode audio buffer');
                }
                
                // Log audio details for debugging
                console.log(`🎵 Decoded audio: ${audioBuffer.duration.toFixed(3)}s, ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}ch`);
                
                // Add to queue for smooth playback
                audioQueue.push(audioBuffer);
                console.log(`🎵 Audio chunk queued (${audioQueue.length} total, duration: ${audioBuffer.duration.toFixed(2)}s)`);
                
                // Start playing if not already playing
                if (!isPlayingAudio) {
                    playNextAudio();
                }
            }).catch(error => {
                console.error('Error decoding audio chunk:', error);
                
                // Enhanced fallback to HTML5 audio with better error handling
                try {
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioElement = new Audio(audioUrl);
                    
                    // Set audio properties for better quality
                    audioElement.preload = 'auto';
                    audioElement.volume = 0.8;
                    
                    // Add error handling for HTML5 audio
                    audioElement.onerror = (e) => {
                        console.error('HTML5 audio error:', e);
                        URL.revokeObjectURL(audioUrl);
                        addChatMessage('error', 'Audio playback error. The audio format may be corrupted.');
                    };
                    
                    audioElement.play().then(() => {
                        console.log('🎵 Audio chunk playing via HTML5 (fallback)');
                    }).catch(e => {
                        console.error('HTML5 audio playback failed:', e);
                        addChatMessage('error', 'Audio playback failed. Please check your browser audio settings.');
                    });
                    
                    audioElement.onended = () => {
                        URL.revokeObjectURL(audioUrl);
                        console.log('🎵 HTML5 audio chunk finished');
                    };
                    
                } catch (fallbackError) {
                    console.error('Fallback audio creation failed:', fallbackError);
                    addChatMessage('error', 'Audio playback system error. Please refresh the page and try again.');
                }
            });
            
        } catch (error) {
            console.error('Error processing audio chunk:', error);
            addChatMessage('error', `Failed to process audio: ${error.message}`);
        }
    }
    
    async function connectWebSocket() {
        try {
            connectionStatus.textContent = '🔄 Connecting...';
            connectionStatus.className = 'connection-status connecting';
            
            // Connect to the backend WebSocket
            websocket = new WebSocket('ws://localhost:8000/ws');
            
            websocket.onopen = function() {
                console.log('✅ Connected to DhanKanya backend');
                updateConnectionStatus('✅ Connected', true);
                status.textContent = '✅ Connected to DhanKanya - Ready to record!';
                status.style.color = '#4CAF50';
                
                startBtn.disabled = false;
                newSessionBtn.disabled = false;
                isConnected = true;
                
                // Clear initial message and add welcome
                chatContainer.innerHTML = '';
                addChatMessage('status', 'Connected to DhanKanya! Click "Start Recording" to begin your conversation.');
                
                // Don't send any initial commands - wait for user to start recording
                console.log('🎯 Connection established, waiting for user to start recording...');
            };
            
            websocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    console.log('📨 Received from backend:', data);
                    
                    switch(data.type) {
                        case 'input_transcription':
                            addChatMessage('input_transcription', data.text, data.is_partial || false);
                            break;
                            
                        case 'output_transcription':
                            addChatMessage('output_transcription', data.text, data.is_partial || false);
                            break;
                            
                        case 'audio_data':
                            console.log('🎵 Received enhanced audio chunk from DhanKanya');
                            console.log(`Audio details: ${data.sample_rate}Hz, ${data.channels}ch, ~${data.duration_ms}ms`);
                            playAudioResponse(data.data, data.mime_type || 'audio/wav');
                            break;
                            
                        case 'status':
                            addChatMessage('status', data.message || data.text);
                            break;
                            
                        case 'session_ended':
                            addChatMessage('status', data.message || 'Session ended.');
                            // Reset recording state
                            isRecording = false;
                            startBtn.disabled = false;
                            stopBtn.disabled = true;
                            status.textContent = '✅ Session ended - Ready for new conversation!';
                            status.style.color = '#4CAF50';
                            break;
                            
                        case 'error':
                            console.error('❌ Backend error:', data.text || data.message);
                            addChatMessage('error', data.text || data.message || 'Unknown error');
                            break;
                            
                        // Handle legacy formats from working backend
                        case undefined:
                            if (data.status === 'session_ended') {
                                addChatMessage('status', data.message || 'Session ended.');
                                // Reset recording state
                                isRecording = false;
                                startBtn.disabled = false;
                                stopBtn.disabled = true;
                                status.textContent = '✅ Session ended - Ready for new conversation!';
                                status.style.color = '#4CAF50';
                            }
                            else if (data.audio_chunk) {
                                console.log('🎵 Received legacy audio chunk from DhanKanya');
                                playAudioResponse(data.audio_chunk, 'audio/wav');
                            }
                            if (data.input_transcript) {
                                addChatMessage('input_transcription', data.input_transcript, false);
                            }
                            if (data.output_transcript) {
                                addChatMessage('output_transcription', data.output_transcript, false);
                            }
                            break;
                            
                        default:
                            console.log('❓ Unknown message type:', data.type, data);
                    }
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                    console.log('Raw message:', event.data);
                }
            };
            
            websocket.onerror = function(error) {
                console.error('❌ WebSocket error:', error);
                updateConnectionStatus('❌ Error', false);
                status.textContent = '❌ Connection error - Check if backend is running on port 8000';
                status.style.color = '#f44336';
                
                addChatMessage('error', 'Connection error. Please ensure the backend server is running on port 8000.');
            };
            
            websocket.onclose = function(event) {
                console.log('🔌 WebSocket connection closed:', event.code, event.reason);
                updateConnectionStatus('🔌 Disconnected', false);
                
                status.textContent = '🔌 Disconnected from backend';
                status.style.color = '#f59e0b';
                
                isConnected = false;
                isRecording = false;
                
                // Reset start button to initial state
                startBtn.textContent = '🎤 Start Recording';
                startBtn.onclick = startRecording;
                startBtn.className = 'control-btn start-btn';
                startBtn.disabled = true;
                newSessionBtn.disabled = true;
                
                if (event.reason) {
                    addChatMessage('error', `Connection closed: ${event.reason} (Code: ${event.code})`);
                }
                
                // Auto-reconnect after 5 seconds if disconnected unexpectedly
                setTimeout(() => {
                    if (!isConnected) {
                        console.log('🔄 Attempting to reconnect...');
                        addChatMessage('status', 'Reconnecting to DhanKanya...');
                        connectWebSocket();
                    }
                }, 5000);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            updateConnectionStatus('❌ Failed', false);
            status.textContent = '❌ Failed to connect to backend';
            status.style.color = '#f44336';
        }
    }
    
    function startRecording() {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            addChatMessage('error', 'WebSocket not connected. Please wait for connection.');
            return;
        }
        
        console.log('🎤 Starting recording - requesting microphone access');
        
        // Send start recording command to backend FIRST
        websocket.send(JSON.stringify({
            action: "start_recording"
        }));
        console.log('📤 Sent start_recording command to backend');
        
        // Check for browser compatibility
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addChatMessage('error', 'Your browser does not support microphone access. Please use Chrome, Firefox, or Edge with HTTPS.');
            return;
        }
        
        // Request microphone access and start streaming
        navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                sampleSize: 16,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: true
            }
        }).then(stream => {
            console.log('🎤 Microphone access granted');
            
            // Create AudioContext for processing
            const recordingAudioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000 // Force 16kHz sample rate
            });
            const source = recordingAudioContext.createMediaStreamSource(stream);
            
            function setupScriptProcessor(audioContext, source) {
                // Create ScriptProcessorNode for audio processing
                const processor = audioContext.createScriptProcessor(1024, 1, 1);
                
                processor.onaudioprocess = function(event) {
                    if (!isRecording) return;
                    
                    const inputBuffer = event.inputBuffer;
                    const inputData = inputBuffer.getChannelData(0);
                    
                    // Convert to 16-bit PCM at 16kHz
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        // Convert float32 [-1,1] to int16 [-32768,32767]
                        pcmData[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32768)));
                    }
                    
                    // Send PCM audio data to backend
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(pcmData.buffer);
                    }
                };
                
                // Connect audio pipeline
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                // Store references for cleanup
                window.currentStream = stream;
                window.currentAudioContext = audioContext;
                window.currentProcessor = processor;
            }
            
            // Setup audio processing
            setupScriptProcessor(recordingAudioContext, source);
            
            isRecording = true;
            // Change start button to stop button
            startBtn.textContent = '⏹️ Stop Recording';
            startBtn.onclick = stopRecording;
            startBtn.className = 'control-btn stop-btn';
            newSessionBtn.disabled = true; // Disable new session while recording
            
            status.textContent = '🔴 Recording... DhanKanya is listening!';
            status.style.color = '#ef4444';
            
            addChatMessage('status', 'Recording started! Speak to DhanKanya now...');
            
        }).catch(error => {
            console.error('❌ Microphone access denied:', error);
            if (error.name === 'NotAllowedError') {
                addChatMessage('error', 'Microphone access denied. Please allow microphone access and try again.');
            } else if (error.name === 'NotFoundError') {
                addChatMessage('error', 'No microphone found. Please check your microphone connection.');
            } else if (error.name === 'NotSupportedError') {
                addChatMessage('error', 'Your browser does not support microphone access. Please use a modern browser.');
            } else {
                addChatMessage('error', `Microphone error: ${error.message}`);
            }
        });
    }
    
    function stopRecording() {
        console.log('⏹️ Stopping recording and ending session');
        
        isRecording = false;
        
        // Stop any currently playing audio to avoid confusion
        stopAllAudio();
        
        // Reset start button
        startBtn.textContent = '🎤 Start Recording';
        startBtn.onclick = startRecording;
        startBtn.className = 'control-btn start-btn';
        startBtn.disabled = true; // Will be enabled when new session starts
        newSessionBtn.disabled = false; // Enable new session button
        
        // Send stop recording command to backend
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                action: "stop_recording"
            }));
            console.log('📤 Sent stop_recording command to backend');
        }
        
        // Clean up audio resources
        if (window.currentStream) {
            window.currentStream.getTracks().forEach(track => track.stop());
            window.currentStream = null;
        }
        
        if (window.currentProcessor) {
            window.currentProcessor.disconnect();
            window.currentProcessor = null;
        }
        
        if (window.currentAudioContext) {
            window.currentAudioContext.close();
            window.currentAudioContext = null;
        }
        
        status.textContent = '⏹️ Recording stopped - Processing your message...';
        status.style.color = '#ff9800';
        
        addChatMessage('status', 'Recording stopped. DhanKanya is processing your message...');
    }
    
    // Auto-connect when page loads
    console.log('🚀 Starting DhanKanya Voice Interface...');
    connectWebSocket();
    
    </script>
        """
        
    # Apply aggressive full-width CSS that breaks out of all constraints
    st.markdown("""
    <style>
    /* Voice interface optimized width */
    .voice-interface iframe {
        width: 100% !important;
        max-width: none !important;
        margin: 0 auto !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.2) !important;
        position: relative !important;
    }
    
    /* Voice interface container uses optimal width */
    .voice-interface {
        width: 100% !important;
        max-width: none !important;
        margin: 0 auto !important;
        padding: 0 !important;
        position: relative !important;
        left: 0 !important;
        right: 0 !important;
    }
    
    /* Override all Streamlit container constraints */
    .main .block-container {
        max-width: none !important;
        padding: 0 !important;
    }
    
    /* Ensure no parent containers limit width */
    .stTabs, .stTabs > div, .stTabs [data-baseweb="tab-panel"] {
        width: 100% !important;
        max-width: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render the HTML component with full viewport and voice-interface class
    st.markdown('<div class="voice-interface">', unsafe_allow_html=True)
    components.html(audio_interface_html, height=800, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface(provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel]) -> None:
    """Render the chat interface with integrated voice and text capabilities."""
    st.markdown("### Chat with DhanKanya")
    st.markdown("Ask questions in Telugu, Tamil, English, or other Indian languages. Use voice or text - both are supported!")
    
    # Show current AI model
    st.caption(f"Currently using: **{provider}** with real-time voice processing")
    
    # Create tabs for different input methods at the top
    text_tab, voice_tab = st.tabs(["💬 Text Chat", "🎤 Voice Chat"])
    
    with text_tab:
        # Chat messages container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                render_message(message)
            
            # Also display voice messages if any
            if 'voice_messages' in st.session_state:
                for message in st.session_state.voice_messages:
                    render_message(message)
        
        # Text input form with inline submit button
        with st.form(key="chat_form", clear_on_submit=True):
            # Create columns for input field and submit button
            input_col, button_col = st.columns([5, 1])
            
            with input_col:
                prompt = st.text_input(
                    "Ask a question in Telugu, Tamil, English, or any Indian language",
                    key="chat_input",
                    label_visibility="collapsed",
                    disabled=st.session_state.is_processing
                )
            
            with button_col:
                submit_button = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                    disabled=st.session_state.is_processing
                )
            
            if submit_button and prompt:
                # Use regular text processing
                handle_text_input(prompt, provider, client)
    
    with voice_tab:
        # Apply aggressive full-width styling that breaks out of all constraints
        st.markdown("""
        <style>
        /* Force voice tab to full viewport width */
        .stTabs [data-baseweb="tab-panel"] iframe {
            width: 100vw !important;
            max-width: none !important;
            margin-left: calc(-50vw + 50%) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            position: relative !important;
        }
        
        /* Tab panel breaks out of all constraints */
        .stTabs [data-baseweb="tab-panel"][aria-labelledby*="Voice"] {
            padding: 0 !important;
            margin: 0 !important;
            width: 100vw !important;
            max-width: none !important;
            margin-left: calc(-50vw + 50%) !important;
            position: relative !important;
            left: 0 !important;
            right: 0 !important;
        }
        
        /* Voice interface in tabs uses full viewport */
        .stTabs .voice-interface {
            width: 100vw !important;
            max-width: none !important;
            margin-left: calc(-50vw + 50%) !important;
            padding: 0 !important;
            position: relative !important;
            left: 0 !important;
            right: 0 !important;
        }
        
        /* Override any tab container width restrictions */
        .stTabs, .stTabs > div {
            width: 100% !important;
            max-width: none !important;
            overflow: visible !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Show voice interface in place of chat interface
        render_integrated_voice_interface()

def render(provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel]) -> None:
    """
    Render the home page with integrated chat and voice interface.
    
    Args:
        provider: The LLM provider name ('Claude' or 'Gemini').
        client: The initialized LLM client for AI interaction.
    """
    # Initialize session state
    initialize_session_state()
    
    # Render page sections
    render_header()
    st.markdown("---")
    render_features()
    st.markdown("---")
    render_chat_interface(provider, client)
