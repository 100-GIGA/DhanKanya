# DhanKanya Voice Assistant

DhanKanya is an AI-powered financial assistant designed to empower young women in India with financial literacy and guidance. The application features both text and voice interaction capabilities, utilizing Anthropic's Claude AI for text responses and Google's Gemini Live API for real-time voice conversations.

## 🎯 Features

### Core Capabilities
- **Multilingual AI Assistant**: Financial guidance in English, Tamil, Telugu, Hindi, and other Indian languages
- **Real-time Voice Conversations**: Native voice interactions using Google's Gemini Live API
- **Text + Voice Hybrid Mode**: Seamlessly switch between text and voice interactions
- **Financial Expertise**: Budgeting, savings, investments, education planning, and government schemes
- **State-Specific Information**: Access to regional scholarships, loans, and financial programs
- **Conversation History**: All interactions (text and voice) are preserved for reference

### Voice Features 🎤
- **Real-time Audio Processing**: Direct microphone input with live transcription
- **Multi-language Support**: Native pronunciation in Tamil, Telugu, Hindi, and English
- **Automatic Language Detection**: Responds in the same language as user input
- **High-Quality Audio**: 24kHz audio processing with noise reduction
- **Voice Activity Detection**: Automatic speech start/stop detection

## 🏗️ Architecture

The DhanKanya Voice Assistant consists of two main components:

### 1. Voice Backend Server (`working_voice_server.py`)
- **FastAPI WebSocket Server** running on port 8000
- **Google Gemini Live Integration** for real-time voice processing
- **Audio Processing Pipeline** with PCM to WAV conversion
- **Session Management** with recording state control

### 2. Streamlit Frontend (`main.py`)
- **Web-based Interface** with chat and voice tabs
- **JavaScript WebSocket Client** for real-time communication
- **Audio Recording** using Web Audio API
- **Responsive UI** with mobile support

## 📋 Prerequisites

### System Requirements
- **Python 3.9+** (Python 3.11+ recommended)
- **Chrome, Firefox, or Edge browser** (for microphone access)
- **Internet connection** (for AI API calls)

### API Keys Required
You'll need the following API keys:

#### 1. Google Gemini API Key
- **Purpose**: Powers the voice conversations using Gemini Live
- **How to get**:
  1. Visit [Google AI Studio](https://aistudio.google.com/)
  2. Sign in with your Google account
  3. Click "Get API Key" → "Create API Key"
  4. Copy your API key (starts with `AIza...`)

#### 2. Anthropic Claude API Key
- **Purpose**: Powers text-based conversations and enhanced responses
- **How to get**:
  1. Visit [Anthropic Console](https://console.anthropic.com/)
  2. Create an account or sign in
  3. Go to "API Keys" → "Create Key"
  4. Copy your API key (starts with `sk-ant-...`)

#### 3. Linkup API Key (Optional)
- **Purpose**: Enhanced web search for financial information
- **How to get**:
  1. Visit [Linkup](https://linkup.so/)
  2. Sign up for an account
  3. Get your API key from the dashboard

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/DhanKanya.git
cd DhanKanya
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

#### Option A: Using .env file (Recommended)
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

Add your API keys to `.env`:
```env
# Required for voice features
GEMINI_API_KEY=AIza_your_gemini_api_key_here
GOOGLE_API_KEY=AIza_your_gemini_api_key_here

# Required for text features  
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here

# Optional for enhanced search
LINKUP_API_KEY=your_linkup_key_here
```

#### Option B: Using Streamlit Secrets
Create the Streamlit secrets file:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "AIza_your_gemini_api_key_here"
GOOGLE_API_KEY = "AIza_your_gemini_api_key_here"
ANTHROPIC_API_KEY = "sk-ant-your_anthropic_key_here"
LINKUP_API_KEY = "your_linkup_key_here"
```

## 🚀 Running the Application

The application requires **two servers** running simultaneously:

### Terminal 1: Start the Voice Backend Server
```bash
# Navigate to project directory
cd DhanKanya

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Start the voice backend server
python working_voice_server.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Terminal 2: Start the Streamlit Frontend
```bash
# Open a new terminal
cd DhanKanya

# Activate virtual environment  
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Start Streamlit application
streamlit run main.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### Step 3: Access the Application
1. Open your browser and go to `http://localhost:8501`
2. Allow microphone access when prompted
3. Navigate to the "🎤 Voice Chat" tab
4. Click "Start Recording" to begin voice conversations

## 🔧 Configuration Options

### Port Configuration
You can customize the ports by editing the files:

**Voice Backend Port** (default: 8000):
```python
# In working_voice_server.py
port = 8000  # Change this line
```

**Streamlit Port** (default: 8501):
```bash
streamlit run main.py --server.port 8502
```

### Audio Settings
Modify audio parameters in `working_voice_server.py`:
```python
# Audio quality settings
CONFIG = {
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}  # Voice selection
    },
    "realtime_input_config": {
        "automatic_activity_detection": {
            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_HIGH,
            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
            "silence_duration_ms": 100,  # Adjust silence detection
        }
    }
}
```

## 🎤 How to Use Voice Features

### Starting a Voice Conversation
1. **Open the Application**: Navigate to `http://localhost:8501`
2. **Go to Voice Tab**: Click on "🎤 Voice Chat" 
3. **Connect**: Wait for "✅ Connected" status
4. **Start Recording**: Click "🎤 Start Recording"
5. **Speak Naturally**: Talk to DhanKanya in your preferred language
6. **Stop Recording**: Click "⏹️ Stop Recording" when done
7. **Listen to Response**: DhanKanya will respond with voice and text

### Language Support
- **Tamil**: Say "வணக்கம்" (Vanakkam) - DhanKanya will respond in Tamil
- **Telugu**: Say "నమస్తే" (Namaste) - DhanKanya will respond in Telugu  
- **Hindi**: Say "नमस्ते" (Namaste) - DhanKanya will respond in Hindi
- **English**: Say "Hello" - DhanKanya will respond in English

### Voice Commands Examples
```
English: "Hello DhanKanya, how can I start saving money?"
Tamil: "வணக்கம் தன்கன்யா, நான் எப்படி பணம் சேமிக்க ஆரம்பிக்கலாம்?"
Telugu: "నమస్తే ధనకన్య, నేను ఎలా డబ్బు ఆదా చేయడం ప్రారంభించగలను?"
Hindi: "नमस्ते धनकन्या, मैं पैसे बचाना कैसे शुरू कर सकती हूँ?"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. "Connection Failed" Error
**Problem**: Voice backend not connecting
**Solutions**:
```bash
# Check if voice server is running
curl http://localhost:8000/status

# Restart voice server
python working_voice_server.py

# Check firewall settings
sudo ufw allow 8000  # Linux
```

#### 2. "Microphone Access Denied"
**Problem**: Browser blocking microphone
**Solutions**:
- Click the microphone icon in browser address bar
- Go to browser Settings → Privacy → Microphone → Allow
- Use HTTPS in production (microphone requires secure context)

#### 3. "API Key Invalid" Error
**Problem**: Incorrect or missing API keys
**Solutions**:
```bash
# Check your .env file
cat .env

# Verify API key format
# Gemini: Should start with "AIza"
# Anthropic: Should start with "sk-ant-"

# Test API key
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.anthropic.com/v1/messages
```

#### 4. "Module Not Found" Errors
**Problem**: Missing dependencies
**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+

# Check virtual environment
which python  # Should point to .venv/bin/python
```

#### 5. Audio Quality Issues
**Problem**: Choppy or unclear audio
**Solutions**:
- Use a good quality microphone
- Reduce background noise
- Check internet connection speed
- Adjust `silence_duration_ms` in configuration

### Logging and Debugging

Enable detailed logging:
```bash
# Run with debug logging
python working_voice_server.py --log-level debug

# Check Streamlit logs
streamlit run main.py --logger.level debug
```

View real-time logs:
```bash
# Monitor voice server logs
tail -f voice_server.log

# Monitor browser console for frontend errors
# Open browser Developer Tools → Console
```

## 🏗️ Project Structure

```
DhanKanya/
├── working_voice_server.py     # Voice backend (FastAPI + Gemini Live)
├── main.py                     # Streamlit frontend entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── README.md                   # This documentation
├── VOICE_ARCHITECTURE.md       # Technical voice system documentation
│
├── app/                        # Core application code
│   ├── components/
│   │   ├── home_page.py       # Main UI with voice interface
│   │   ├── auth_pages.py      # Authentication
│   │   └── expense_tracker_page.py
│   ├── services/
│   │   ├── ai_service.py      # Claude AI integration
│   │   └── expense_service.py
│   └── utils/
│       └── helpers.py
│
├── config/
│   └── settings.py            # Application configuration
├── assets/
│   └── images/
└── .streamlit/
    └── secrets.toml           # Streamlit secrets (create this)
```

## 📱 Browser Compatibility

### Supported Browsers
- ✅ **Chrome 80+** (Recommended)
- ✅ **Firefox 76+** 
- ✅ **Edge 80+**
- ✅ **Safari 14+** (macOS/iOS)

### Required Browser Features
- WebSocket support
- Web Audio API
- Microphone access
- JavaScript ES6+

## 🔐 Security Notes

### Production Deployment
- **Use HTTPS**: Required for microphone access
- **Secure API Keys**: Use environment variables, never commit keys
- **Firewall Rules**: Only expose necessary ports
- **CORS Configuration**: Restrict allowed origins in production

### API Key Security
```python
# ✅ Good - Using environment variables
api_key = os.getenv("GEMINI_API_KEY")

# ❌ Bad - Hardcoded keys
api_key = "AIzaSyC..."  # Never do this
```

## 🚀 Production Deployment

### Using Docker (Recommended)
```bash
# Build Docker image
docker build -t dhankanya .

# Run with environment variables
docker run -p 8000:8000 -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  dhankanya
```

### Using PM2 (Node.js Process Manager)
```bash
# Install PM2
npm install -g pm2

# Start voice backend
pm2 start working_voice_server.py --name voice-backend

# Start Streamlit frontend  
pm2 start "streamlit run main.py" --name streamlit-frontend

# Monitor processes
pm2 monit
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m "Add feature"`
5. Push to branch: `git push origin feature-name`
6. Create a Pull Request

## 📞 Support

### Getting Help
- **Documentation**: Read `VOICE_ARCHITECTURE.md` for technical details
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

### System Status
Check if services are running:
```bash
# Voice backend status
curl http://localhost:8000/status

# Streamlit status
curl http://localhost:8501
```

## 📄 License

This project is proprietary and confidential. All rights reserved by 100GIGA Finance Team.

---

**🎉 Happy Financial Planning with DhanKanya! 🎉**

