# DhanKanya Quick Setup Guide

This is a condensed setup guide for developers who want to get DhanKanya Voice Assistant running quickly.

## ⚡ Quick Start (5 Minutes)

### 1. Prerequisites Check
```bash
# Check Python version (need 3.9+)
python --version

# Check if you have pip
pip --version
```

### 2. Clone & Setup
```bash
git clone https://github.com/pritamvarma-ai/DhanKanya.git
cd DhanKanya

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Get API Keys (2 minutes each)

#### Gemini API Key (Required for Voice)
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in → "Get API Key" → "Create API Key"
3. Copy key (starts with `AIza...`)
Note: Your Google account need to be signed in to the Google Cloud Console.

#### Anthropic API Key (Required for Text)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in → "API Keys" → "Create Key"
3. Copy key (starts with `sk-ant-...`)

### 4. Configure Environment
```bash
# Create .env file
echo "GEMINI_API_KEY=your_gemini_key_here" > .env
echo "GOOGLE_API_KEY=your_gemini_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
```

### 5. Run the Application
```bash
# Terminal 1: Start voice backend
python working_voice_server.py

# Terminal 2: Start frontend (in new terminal)
streamlit run main.py
```

### 6. Test
1. Open `http://localhost:8501`
2. Go to "🎤 Voice Chat" tab
3. Allow microphone access
4. Click "Start Recording" and say "Hello"

## 🔧 Common Issues & Fixes

### "Connection Failed"
```bash
# Check if port 8000 is free
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Restart voice backend
python working_voice_server.py
```

### "Microphone Access Denied"
- Click microphone icon in browser address bar
- Grant permission to localhost

### "Module Not Found"
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### "Invalid API Key"
```bash
# Check your .env file
cat .env

# Verify API key format
# Gemini: AIza...
# Anthropic: sk-ant-...
```

## 📁 Project Structure
```
DhanKanya/
├── working_voice_server.py    # Voice backend (port 8000)
├── main.py                    # Streamlit app (port 8501)
├── requirements.txt           # Dependencies
├── .env                       # API keys (create this)
├── README.md                  # Full documentation
└── VOICE_ARCHITECTURE.md      # Technical details
```

## 🎯 Testing Checklist
- [ ] Voice backend starts on port 8000
- [ ] Streamlit opens on port 8501
- [ ] "Connected" status shows in voice tab
- [ ] Microphone access granted
- [ ] Recording starts/stops properly
- [ ] Audio playback works
- [ ] Multiple languages work (English, Tamil, Telugu, Hindi)

## 🚀 Production Deployment
```bash
# Using Docker
docker build -t dhankanya .
docker run -p 8000:8000 -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  dhankanya
```

## 📞 Get Help
- **Issues**: Check `README.md` for detailed troubleshooting
- **Technical Details**: Read `VOICE_ARCHITECTURE.md`
- **API Errors**: Verify your API keys in `.env` file
- **Browser Issues**: Use Chrome/Firefox/Edge (latest versions)

---
**Total Setup Time: ~5 minutes** ⏱️
