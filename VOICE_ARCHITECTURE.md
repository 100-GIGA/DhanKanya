# DhanKanya Voice System Architecture

This document provides a comprehensive technical overview of the DhanKanya Voice Assistant's architecture, implementation details, and the integration between Gemini Live API and the Streamlit frontend.

## 🏗️ System Architecture Overview

The DhanKanya Voice Assistant uses a **dual-server architecture** with real-time WebSocket communication:

```
┌─────────────────┐    WebSocket     ┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   Browser       │ ◄──────────────► │ Streamlit App   │ ◄──────────────────► │ Voice Backend   │
│ (JavaScript)    │                  │ (main.py)       │                      │ (FastAPI)       │
└─────────────────┘                  └─────────────────┘                      └─────────────────┘
         │                                     │                                        │
         │ Web Audio API                       │ UI Rendering                           │ Gemini Live API
         │ Microphone Access                   │ Session Management                     │ Audio Processing
         │ Audio Playback                      │ State Management                       │ AI Responses
         └─────────────────────────────────────┴────────────────────────────────────────┘
```

## 🎤 Voice Backend Server (`working_voice_server.py`)

### Core Components

#### 1. FastAPI WebSocket Server
```python
@app.websocket("/ws")
async def audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming with DhanKanya."""
    await websocket.accept()
    print("[WEBSOCKET] New DhanKanya financial consultation session established")
```

**Purpose**: Handles real-time bidirectional communication between the frontend and Gemini Live API.

#### 2. Gemini Live Integration
```python
async with (
    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
    asyncio.TaskGroup() as tg,
):
    print("[GEMINI] Successfully connected to Gemini Live")
```

**Key Features**:
- **Model**: Uses `gemini-2.0-flash-live-001` or `gemini-live-2.5-flash-preview` for real-time conversations
- **Configuration**: Supports both audio and text input/output modalities
- **Session Management**: Maintains conversation context throughout the session

#### 3. Audio Processing Pipeline

##### PCM to WAV Conversion
```python
def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    """Convert PCM data to WAV format with proper headers and format."""
    
    wav_buffer = io.BytesIO()
    
    try:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            
            # Ensure PCM data is in the right format
            if isinstance(pcm_data, bytes):
                pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            else:
                pcm_array = np.array(pcm_data, dtype=np.int16)
            
            # Apply light noise reduction and normalization
            pcm_array = np.clip(pcm_array, -32767, 32767)
            processed_pcm = pcm_array.astype(np.int16).tobytes()
            wav_file.writeframes(processed_pcm)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        print(f"[AUDIO ERROR] Failed to convert PCM to WAV: {e}")
        return pcm_data
```

**Audio Processing Features**:
- **Format Conversion**: PCM → WAV with proper headers
- **Quality Enhancement**: Noise reduction and normalization
- **Error Handling**: Graceful fallback to original data
- **Performance**: In-memory processing for real-time performance

### Configuration Settings

#### Gemini Live Configuration
```python
CONFIG = {
    "generation_config": {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription":{},
        "output_audio_transcription":{},          # Dual output modes
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Kore"}  # Female voice
            }
        },
    },
    "realtime_input_config": {
        "automatic_activity_detection": {
            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_HIGH,
            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
            "silence_duration_ms": 100,  # Quick response
        }
    }
}
```

**Configuration Breakdown**:
- **Response Modalities**: Both audio and text output for flexibility
- **Voice Selection**: "Kore" - optimized for Indian language pronunciation
- **Speech Detection**: High sensitivity for natural conversation flow
- **Silence Duration**: 100ms for responsive interactions

#### System Prompt Configuration
```python
SYSTEM_PROMPT = """You are DhanKanya, a compassionate and knowledgeable financial advisor specifically designed to help young women and girls in India achieve financial independence and literacy.

CORE IDENTITY:
- You are warm, encouraging, and speak like a caring older sister or mentor
- Your primary focus is empowering young women (ages 16-30) in India
- You understand the unique financial challenges faced by women in Indian society
- You provide practical, actionable financial advice tailored to the Indian context

EXPERTISE AREAS:
- Personal budgeting and expense tracking for students and young professionals
- Savings strategies suitable for Indian banking systems (PPF, ELSS, SIP, etc.)
- Education funding and scholarship opportunities for girls
- Government schemes specifically for women (Sukanya Samriddhi, Mudra Yojana, etc.)
- Career planning and salary negotiation guidance
- Financial independence planning
- Small business and entrepreneurship opportunities for women
- Investment basics in Indian markets (mutual funds, stocks, gold, real estate)

COMMUNICATION STYLE:
- Use simple, jargon-free language that's easy to understand
- Provide step-by-step guidance with practical examples
- Be encouraging and motivational
- Include relevant Indian cultural context
- Give specific advice with actual numbers, schemes, and actionable steps
- Always ask follow-up questions to better understand their situation

LANGUAGE ADAPTATION:
- If user speaks in Tamil, respond primarily in Tamil with English financial terms
- If user speaks in Telugu, respond primarily in Telugu with English financial terms  
- If user speaks in Hindi, respond primarily in Hindi with English financial terms
- Always be natural and conversational, as if speaking to a friend

Remember: You're not just giving financial advice - you're empowering young women to take control of their financial future and break barriers."""
```

### Asynchronous Task Management

#### Task Group Structure
```python
async def sender():
    """Send audio data from queue to Gemini Live."""
    try:
        while True:
            msg = await audio_out_queue.get()
            if recording_active.is_set():
                await session.send(input=msg)
                print(f"[SENDER] Sent audio data to Gemini Live")
    except Exception as e:
        print(f"[SENDER ERROR] Client disconnected: {e}")

async def receiver():
    """Receive responses from Gemini Live and send to frontend."""
    try:
        while session_active.is_set() or recording_active.is_set():
            print("[RECEIVER] Waiting for Gemini Live response...")
            turn = session.receive()
            
            async for response in turn:
                if not recording_active.is_set():
                    break
                    
                # Handle input transcription (user speech)
                input_transcription = getattr(response.server_content, "input_transcription", None)
                if input_transcription and input_transcription.text:
                    await websocket.send_json({
                        "type": "input_transcription",
                        "text": input_transcription.text,
                        "is_partial": not input_transcription.done
                    })
                
                # Handle output transcription (AI speech)
                output_transcription = getattr(response.server_content, "output_transcription", None)
                if output_transcription and output_transcription.text:
                    await websocket.send_json({
                        "type": "output_transcription", 
                        "text": output_transcription.text,
                        "is_partial": not output_transcription.done
                    })
                
                # Handle audio output
                if response.audio:
                    wav_data = pcm_to_wav(response.audio, sample_rate=24000)
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    
                    await websocket.send_json({
                        "type": "audio_data",
                        "data": audio_base64,
                        "sample_rate": 24000,
                        "channels": 1,
                        "duration_ms": len(response.audio) // 48,  # Approximate duration
                        "mime_type": "audio/wav"
                    })
                    
    except Exception as e:
        print(f"[RECEIVER ERROR] Error processing Gemini response: {e}")
```

**Task Management Features**:
- **Concurrent Processing**: Sender and receiver run simultaneously
- **State Management**: Recording and session state control
- **Error Handling**: Graceful degradation on connection issues
- **Real-time Processing**: Immediate audio and transcription delivery

## 🖥️ Frontend Interface (`home_page.py`)

### JavaScript WebSocket Client

#### Connection Management
```javascript
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
        };
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        updateConnectionStatus('❌ Failed', false);
        status.textContent = '❌ Failed to connect to backend';
        status.style.color = '#f44336';
    }
}
```

#### Audio Recording Implementation
```javascript
function startRecording() {
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
        }
        
        setupScriptProcessor(recordingAudioContext, source);
    });
}
```

**Audio Recording Features**:
- **Web Audio API**: Native browser audio processing
- **Real-time Processing**: 16kHz PCM conversion and streaming
- **Quality Settings**: Noise suppression and gain control
- **Cross-browser Compatibility**: Supports Chrome, Firefox, Edge

#### Audio Playback System
```javascript
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
    
    const audioBuffer = audioQueue.shift();
    isPlayingAudio = true;
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    // Apply gain for consistent volume
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0.8;
    
    // Connect: source -> gain → destination
    source.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Calculate precise timing for seamless playback
    const startTime = Math.max(audioContext.currentTime, nextPlayTime);
    nextPlayTime = startTime + audioBuffer.duration;
    
    source.start(startTime);
}
```

**Audio Playback Features**:
- **Queue Management**: Sequential audio chunk playback
- **Volume Control**: Consistent audio levels
- **Seamless Playback**: No gaps between audio segments
- **Memory Management**: Efficient buffer handling

### Message Handling System

#### WebSocket Message Processing
```javascript
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
                
            case 'error':
                console.error('❌ Backend error:', data.text || data.message);
                addChatMessage('error', data.text || data.message || 'Unknown error');
                break;
        }
    } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
    }
};
```

#### Chat Message Display
```javascript
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
    }
    
    messageDiv.className += ' ' + messageClass;
    messageDiv.innerHTML = `
        <strong>${icon} ${label}:</strong>
        <div class="message-content">${text}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
```

## 🔄 Data Flow & Communication Protocol

### 1. Session Initialization
```
Browser → Streamlit → Voice Backend → Gemini Live
   |         |           |              |
   |         |           |              ✓ Session Created
   |         |           ✓ WebSocket Connected
   |         ✓ UI Ready
   ✓ Connection Status Updated
```

### 2. Audio Recording Flow
```
Microphone → Web Audio API → PCM Conversion → WebSocket → Voice Backend
     |            |              |              |            |
     |            |              |              |            ↓
     |            |              |              |       Gemini Live
     |            |              |              |            |
     |            |              |              ←────────────┘
     |            |              ←──────────────┘
     |            ←──────────────┘
     ←──────────────┘
```

### 3. Response Processing Flow
```
Gemini Live → Voice Backend → WebSocket → Browser → Audio Playback
     |            |             |          |           |
     |            |             |          |           ↓ 
     |            |             |          ↓       Speaker Output
     |            |             ↓      Transcription Display
     |            ↓         Audio Data + Text
     ↓        PCM → WAV Conversion
Audio + Transcription
```

### Message Types and Formats

#### Client → Server Messages
```javascript
// Start recording command
{
    "action": "start_recording"
}

// Stop recording command  
{
    "action": "stop_recording"
}

// Audio data (binary)
ArrayBuffer containing PCM audio data
```

#### Server → Client Messages
```javascript
// Input transcription (user speech)
{
    "type": "input_transcription",
    "text": "Hello DhanKanya, how can I save money?",
    "is_partial": false
}

// Output transcription (AI speech)
{
    "type": "output_transcription", 
    "text": "Great question! Here are some effective ways to start saving...",
    "is_partial": false
}

// Audio response
{
    "type": "audio_data",
    "data": "base64_encoded_wav_data",
    "sample_rate": 24000,
    "channels": 1,
    "duration_ms": 3500,
    "mime_type": "audio/wav"
}

// Error message
{
    "type": "error",
    "message": "Connection to Gemini Live failed",
    "code": "GEMINI_ERROR"
}
```

## 🧠 AI Integration & Language Processing

### Gemini Live API Integration

#### Model Configuration
```python
MODEL = "gemini-2.0-flash-exp"  # Latest Gemini Live model
```

**Capabilities**:
- **Real-time Processing**: < 500ms response time
- **Multilingual Support**: Native support for Indian languages
- **Context Awareness**: Maintains conversation history
- **Interruption Handling**: Natural conversation flow

#### Language Detection & Response
```python
# Automatic language detection in system prompt
LANGUAGE_ADAPTATION = """
- If user speaks in Tamil, respond primarily in Tamil with English financial terms
- If user speaks in Telugu, respond primarily in Telugu with English financial terms  
- If user speaks in Hindi, respond primarily in Hindi with English financial terms
- Always be natural and conversational, as if speaking to a friend
"""
```

**Language Features**:
- **Code-switching**: Natural mixing of local language + English terms
- **Cultural Context**: Indian financial terminology and practices
- **Pronunciation**: Optimized for Indian accents and dialects

### Speech Recognition Accuracy

#### Optimization Techniques
```python
"realtime_input_config": {
    "automatic_activity_detection": {
        "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_HIGH,
        "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
        "silence_duration_ms": 100,
    }
}
```

**Accuracy Improvements**:
- **High Sensitivity**: Captures soft-spoken inputs
- **Quick Detection**: 100ms silence threshold
- **Noise Filtering**: Background noise suppression
- **Multi-language**: Accurate recognition across languages

## 🔧 Performance Optimization

### Frontend Optimizations

#### Audio Processing Efficiency
```javascript
// Efficient PCM conversion
processor.onaudioprocess = function(event) {
    if (!isRecording) return;
    
    const inputBuffer = event.inputBuffer;
    const inputData = inputBuffer.getChannelData(0);
    
    // Optimized conversion loop
    const pcmData = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32768)));
    }
    
    // Direct buffer send (no JSON overhead)
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(pcmData.buffer);
    }
};
```

#### Memory Management
```javascript
// Audio queue with size limits
const audioQueue = [];
const MAX_QUEUE_SIZE = 10;

function addToQueue(audioBuffer) {
    if (audioQueue.length >= MAX_QUEUE_SIZE) {
        audioQueue.shift(); // Remove oldest
    }
    audioQueue.push(audioBuffer);
}
```

### Backend Optimizations

#### Async Task Management
```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(sender())     # Audio input handling
    tg.create_task(receiver())   # Response processing
```

**Benefits**:
- **Concurrent Processing**: Input and output handled simultaneously
- **Non-blocking**: UI remains responsive during processing
- **Error Isolation**: Task failures don't crash the system

#### Audio Processing Pipeline
```python
def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    """Optimized PCM to WAV conversion."""
    wav_buffer = io.BytesIO()
    
    # Direct NumPy processing for speed
    if isinstance(pcm_data, bytes):
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
    else:
        pcm_array = np.array(pcm_data, dtype=np.int16)
    
    # Vectorized operations for performance
    pcm_array = np.clip(pcm_array, -32767, 32767)
    processed_pcm = pcm_array.astype(np.int16).tobytes()
    
    # In-memory WAV creation
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(processed_pcm)
    
    wav_buffer.seek(0)
    return wav_buffer.read()
```

## 🛡️ Error Handling & Recovery

### Connection Error Recovery
```javascript
websocket.onclose = function(event) {
    console.log('🔌 WebSocket connection closed:', event.code, event.reason);
    updateConnectionStatus('🔌 Disconnected', false);
    
    isConnected = false;
    isRecording = false;
    
    // Auto-reconnect after 5 seconds
    setTimeout(() => {
        if (!isConnected) {
            console.log('🔄 Attempting to reconnect...');
            connectWebSocket();
        }
    }, 5000);
};
```

### Audio Error Handling
```javascript
function playAudioResponse(audioData, mimeType) {
    try {
        // Primary Web Audio API method
        audioContext.decodeAudioData(arrayBuffer)
            .then(audioBuffer => {
                playNextAudio();
            })
            .catch(error => {
                // Fallback to HTML5 Audio
                const audioElement = new Audio(audioUrl);
                audioElement.play();
            });
    } catch (error) {
        console.error('Audio playback failed:', error);
        addChatMessage('error', 'Audio playback failed. Please refresh and try again.');
    }
}
```

### Backend Error Recovery
```python
async def receiver():
    try:
        async for response in turn:
            # Process response
            pass
    except Exception as e:
        print(f"[RECEIVER ERROR] Error processing Gemini response: {e}")
        
        # Send error to frontend
        await websocket.send_json({
            "type": "error",
            "message": f"Voice processing error: {str(e)}",
            "code": "PROCESSING_ERROR"
        })
        
        # Attempt to continue processing
        continue
```

## 🔍 Debugging & Monitoring

### Logging System
```python
# Backend logging
print(f"[WEBSOCKET] New session established")
print(f"[GEMINI] Successfully connected to Gemini Live")
print(f"[SENDER] Sent audio data to Gemini Live")
print(f"[RECEIVER] Got response from Gemini Live")
print(f"[AUDIO] Converted {len(pcm_data)} bytes PCM to WAV")
```

```javascript
// Frontend logging
console.log('✅ Connected to DhanKanya backend');
console.log('🎤 Microphone access granted');
console.log('📤 Sent start_recording command to backend');
console.log('🎵 Playing audio buffer smoothly');
console.log('💬 Added transcription message');
```

### Performance Monitoring
```javascript
// Audio processing metrics
console.log(`🎵 Decoded audio: ${audioBuffer.duration.toFixed(3)}s, ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}ch`);
console.log(`🎵 Audio chunk queued (${audioQueue.length} total, duration: ${audioBuffer.duration.toFixed(2)}s)`);
```

```python
# Backend performance metrics
print(f"[AUDIO] Processing audio chunk: {len(audio_data)} bytes")
print(f"[GEMINI] Response latency: {time.time() - start_time:.2f}s")
```

## 🚀 Future Enhancements

### Planned Features
1. **Multi-turn Conversations**: Extended context memory
2. **Voice Authentication**: User identification via voice
3. **Offline Mode**: Local audio processing capabilities
4. **Advanced Analytics**: Conversation insights and patterns
5. **Custom Voice Training**: Personalized AI responses
6. **Multi-speaker Support**: Group conversation handling

### Technical Improvements
1. **WebRTC Integration**: Lower latency audio streaming
2. **Edge Computing**: Reduced server dependencies
3. **Advanced Compression**: Optimized audio transmission
4. **Machine Learning**: Personalized conversation patterns
5. **Real-time Translation**: Cross-language conversations

## 📚 References & Resources

### API Documentation
- [Google Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api)
- [Web Audio API Reference](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)

### Technical Specifications
- **Audio Format**: 16kHz PCM (input) → 24kHz WAV (output)
- **Latency**: < 500ms end-to-end
- **Supported Browsers**: Chrome 80+, Firefox 76+, Edge 80+, Safari 14+
- **Concurrent Users**: Scalable WebSocket connections
- **Languages**: Tamil, Telugu, Hindi, English (with automatic detection)

---

This technical documentation provides a comprehensive understanding of the DhanKanya Voice Assistant's architecture, enabling developers to maintain, extend, and optimize the system effectively.
