# 🎤 DhanKanya Voice System Setup Guide

## Complete Gemini Live Audio Pipeline Implementation

### 🎯 Overview

This implementation provides a complete voice interaction system for DhanKanya using Google's Gemini Live API, including:

- ✅ **Voice Input**: Record audio directly in Streamlit
- ✅ **Speech Transcription**: Convert speech to text
- ✅ **Text Processing**: AI understanding and response generation
- ✅ **Voice Output**: Text-to-speech with audio playback
- ✅ **Multi-language Support**: Tamil and English prioritized
- ✅ **Real-time Processing**: WebSocket-based communication

### 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │  Gemini Live     │    │   Google        │
│   Frontend      │◄──►│  Backend         │◄──►│   Gemini API    │
│                 │    │  (Port 8002)     │    │                 │
│ • Voice UI      │    │ • WebSocket      │    │ • Audio I/O     │
│ • Audio Player  │    │ • Session Mgmt   │    │ • Transcription │
│ • Text Input    │    │ • Audio Proc     │    │ • Voice Synth   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 📁 New Files Created

1. **`gemini_live_backend.py`** - Complete Gemini Live WebSocket backend
2. **`enhanced_voice_interface.py`** - Streamlit voice UI with recording
3. **`test_gemini_live.py`** - Comprehensive testing script
4. **`start_voice_system.sh`** - System startup script
5. **`stop_voice_system.sh`** - System shutdown script

### 🔧 Installation & Setup

#### Step 1: Install Dependencies
```bash
# Install audio recording package
pip install streamlit-audio-recorder

# Install all requirements
pip install -r requirements.txt
```

#### Step 2: Set Google API Key
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

#### Step 3: Start the Complete System
```bash
# Option A: Use startup script (recommended)
./start_voice_system.sh

# Option B: Manual startup
# Terminal 1: Start Gemini Live backend
python gemini_live_backend.py

# Terminal 2: Start Streamlit app
streamlit run main.py
```

### 🎤 Features Available

#### 1. Voice Recording Interface
- **Location**: Voice Assistant tab in main app
- **Feature**: Click-to-record audio input
- **Languages**: Tamil, English, Hindi
- **Format**: 16kHz WAV, compatible with Gemini Live

#### 2. Text-to-Voice Interface
- **Location**: Same Voice Assistant tab
- **Feature**: Type text, get voice response
- **Languages**: Multi-language response generation
- **Output**: High-quality audio synthesis

#### 3. Real-time Processing
- **WebSocket**: Persistent connection to Gemini Live
- **Transcription**: Real-time speech-to-text
- **Response**: Immediate audio generation
- **Playback**: Integrated Streamlit audio player

### 🧪 Testing

#### Quick Test
```bash
# Test backend health
python test_gemini_live.py

# Expected output:
# ✅ Health check passed
# ✅ WebSocket connection successful  
# 🎵 Audio response received!
```

#### Manual Testing Steps

1. **Start System**:
   ```bash
   ./start_voice_system.sh
   ```

2. **Open Browser**: http://localhost:8501

3. **Navigate**: Voice Assistant tab

4. **Test Voice Recording**:
   - Click microphone button
   - Say: "வணக்கம் DhanKanya, முதலீடு பற்றி சொல்லுங்க"
   - Check for transcription and audio response

5. **Test Text-to-Voice**:
   - Type: "Hi DhanKanya, explain savings accounts"
   - Click "Get Voice Response"
   - Listen to audio output

### 🌍 Language Support

#### Primary Languages
- **Tamil (தமிழ்)**: Full voice I/O support
- **English**: Full voice I/O support

#### Language Detection
```python
# Automatic language detection
tamil_range = "\u0B80-\u0BFF"  # Tamil script
english_range = "a-zA-Z"       # Latin script
```

#### Voice Features per Language
| Language | Voice Input | Voice Output | Transcription |
|----------|-------------|--------------|---------------|
| Tamil    | ✅          | ✅           | ✅            |
| English  | ✅          | ✅           | ✅            |
| Hindi    | ✅          | ⚠️ Limited   | ✅            |

### 🔄 API Integration

#### Gemini Live Models Used
- **Text**: `gemini-2.0-flash-exp`
- **Audio**: `gemini-2.5-flash-preview-native-audio-dialog`

#### WebSocket Message Types
```json
// Text input
{
  "type": "text_message",
  "text": "user message"
}

// Audio input  
{
  "type": "audio_input",
  "audio_data": "base64_encoded_audio"
}

// Text response
{
  "type": "text_response", 
  "text": "ai response"
}

// Audio response
{
  "type": "audio_response",
  "audio_data": "base64_encoded_audio",
  "sample_rate": 24000
}
```

### 🗄️ Database Integration

#### Voice Conversations Table
```sql
CREATE TABLE voice_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    session_id TEXT,
    input_type TEXT,  -- 'text' or 'audio'
    input_text TEXT,
    response_text TEXT,
    language_detected TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

### 🔧 Configuration

#### Backend Settings
```python
# gemini_live_backend.py
GEMINI_LIVE_PORT = 8002
MAX_CONNECTIONS = 100
AUDIO_SAMPLE_RATE = 24000
SESSION_TIMEOUT = 3600  # 1 hour
```

#### Frontend Settings  
```python
# config/settings.py
VOICE_ENABLED = True
SUPPORTED_VOICE_LANGUAGES = {
    'ta': 'Tamil',
    'en': 'English', 
    'hi': 'Hindi'
}
```

### 🚀 Production Deployment

#### System Requirements
- **CPU**: 2+ cores for real-time audio processing
- **RAM**: 4GB+ for Gemini Live connections
- **Storage**: 10GB+ for audio caching
- **Network**: Stable internet for Google API calls

#### Environment Variables
```bash
export GOOGLE_API_KEY="your_production_key"
export VOICE_BACKEND_URL="wss://your-domain.com/ws/voice"  
export AUDIO_CACHE_DIR="/var/cache/dhankanya/audio"
```

#### Docker Deployment
```dockerfile
# Use in production Dockerfile
COPY gemini_live_backend.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8002
CMD ["python", "gemini_live_backend.py"]
```

### 🐛 Troubleshooting

#### Common Issues

1. **"Audio recording not available"**
   ```bash
   pip install streamlit-audio-recorder
   ```

2. **"Failed to connect to Gemini Live"**
   - Check GOOGLE_API_KEY is set
   - Verify backend is running on port 8002
   - Test with: `curl http://localhost:8002/health`

3. **"No audio output"**
   - Check browser audio permissions
   - Verify audio file is created (temp_response_*.wav)
   - Test with test script

4. **"WebSocket connection failed"**
   - Check if port 8002 is available
   - Restart backend: `python gemini_live_backend.py`
   - Check firewall settings

#### Debug Commands
```bash
# Check backend status
curl http://localhost:8002/health

# Test WebSocket manually
python test_gemini_live.py

# Check running processes
ps aux | grep -E "(gemini_live|streamlit)"

# Check port usage
lsof -i :8002
lsof -i :8501
```

### 📊 Performance Metrics

#### Expected Performance
- **Voice Input Processing**: < 3 seconds
- **Text-to-Speech Generation**: < 2 seconds  
- **WebSocket Latency**: < 100ms
- **Audio Quality**: 24kHz, 16-bit

#### Monitoring
```python
# Built-in performance tracking
response_time = time.time() - start_time
logger.info(f"Voice processing took {response_time:.2f}s")
```

### 🔐 Security Considerations

#### API Key Protection
- Never commit API keys to git
- Use environment variables in production
- Rotate keys regularly

#### Audio Data Handling
- Audio files are temporary (auto-deleted)
- No persistent audio storage
- Base64 encoding for transmission

#### WebSocket Security
- Session-based connections
- Automatic cleanup on disconnect
- Rate limiting implemented

### 🎯 Next Steps

#### Planned Enhancements
1. **Real-time Streaming**: Continuous audio processing
2. **Voice Cloning**: Custom DhanKanya voice
3. **Advanced NLP**: Context-aware responses
4. **Mobile Support**: React Native integration
5. **Offline Mode**: Local speech processing

#### Contributing
```bash
# Development setup
git clone <repository>
cd DK
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
./start_voice_system.sh
```

---

## 🎉 Success! 

Your DhanKanya voice system is now ready with complete Gemini Live integration!

**Test it now**: 
1. Run `./start_voice_system.sh`
2. Open http://localhost:8501  
3. Go to Voice Assistant tab
4. Start talking to DhanKanya! 🎤✨
