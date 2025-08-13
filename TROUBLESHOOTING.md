# DhanKanya Troubleshooting Guide

This guide helps you diagnose and fix common issues with the DhanKanya Voice Assistant.

## 🔍 Quick Diagnostics

### Health Check Commands
```bash
# Check if voice backend is running
curl http://localhost:8000/status

# Expected response:
# {"status": "ready", "message": "DhanKanya Voice Assistant is ready for connections"}

# Check if Streamlit is running
curl http://localhost:8501/_stcore/health

# Check Python environment
which python  # Should point to .venv/bin/python
python --version  # Should be 3.9+
```

### Log Monitoring
```bash
# Monitor voice backend logs
python working_voice_server.py | tee voice_backend.log

# Monitor Streamlit logs  
streamlit run main.py --logger.level debug 2>&1 | tee streamlit.log

# Check browser console logs
# Open Developer Tools → Console tab
```

## 🚨 Common Issues & Solutions

### 1. Connection Issues

#### Error: "Connection Failed" or "WebSocket connection closed"

**Symptoms:**
- ❌ Disconnected status in voice tab
- Cannot start recording
- "Failed to connect to backend" message

**Diagnosis:**
```bash
# Check if voice backend is running
ps aux | grep working_voice_server
curl http://localhost:8000/status

# Check port availability
lsof -i :8000
netstat -tulpn | grep :8000
```

**Solutions:**
```bash
# Solution 1: Restart voice backend
pkill -f working_voice_server
python working_voice_server.py

# Solution 2: Change port if 8000 is busy
# Edit working_voice_server.py line ~370:
# uvicorn.run(app, host="0.0.0.0", port=8001)

# Solution 3: Check firewall
sudo ufw allow 8000  # Linux
# Windows: Add firewall rule for port 8000

# Solution 4: Check if running in correct directory
cd /path/to/DhanKanya
python working_voice_server.py
```

#### Error: "CORS policy" or "Origin not allowed"

**Solution:**
```python
# In working_voice_server.py, ensure CORS is configured:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. API Key Issues

#### Error: "Invalid API key" or "Authentication failed"

**Symptoms:**
- Gemini Live connection fails
- Text responses don't work
- "API key invalid" in logs

**Diagnosis:**
```bash
# Check if API keys are set
echo $GEMINI_API_KEY
echo $ANTHROPIC_API_KEY

# Check .env file format
cat .env
# Should look like:
# GEMINI_API_KEY=AIzaSyC...
# ANTHROPIC_API_KEY=sk-ant-...

# Test API keys manually
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages

curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models
```

**Solutions:**
```bash
# Solution 1: Recreate .env file
rm .env
echo "GEMINI_API_KEY=your_actual_key_here" > .env
echo "ANTHROPIC_API_KEY=your_actual_key_here" >> .env

# Solution 2: Use Streamlit secrets instead
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your keys

# Solution 3: Set environment variables directly
export GEMINI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
python working_voice_server.py

# Solution 4: Verify key format
# Gemini: Must start with "AIza"
# Anthropic: Must start with "sk-ant-"
```

#### Error: "Quota exceeded" or "Rate limit"

**Solution:**
```bash
# Check your API usage limits
# Gemini: https://aistudio.google.com/
# Anthropic: https://console.anthropic.com/

# Wait for quota reset or upgrade plan
# Implement rate limiting in code if needed
```

### 3. Microphone Issues

#### Error: "Microphone access denied" or "getUserMedia failed"

**Symptoms:**
- Browser shows microphone blocked icon
- "NotAllowedError" in console
- Recording doesn't start

**Diagnosis:**
```javascript
// Test microphone access in browser console
navigator.mediaDevices.getUserMedia({audio: true})
  .then(stream => console.log('✅ Microphone access granted'))
  .catch(err => console.error('❌ Microphone error:', err));
```

**Solutions:**
```bash
# Solution 1: Grant permission in browser
# Chrome: Click microphone icon in address bar
# Firefox: Click shield icon → Allow microphone
# Edge: Click lock icon → Allow microphone

# Solution 2: Check browser permissions
# Chrome: Settings → Privacy → Site Settings → Microphone
# Firefox: Preferences → Privacy → Permissions → Microphone
# Edge: Settings → Site permissions → Microphone

# Solution 3: Use HTTPS (required for production)
# Microphone access requires secure context
# Use ngrok or deploy with SSL certificate

# Solution 4: Check system microphone
# Test microphone in other applications
# Adjust system audio input levels
```

#### Error: "No microphone found" or "NotFoundError"

**Solutions:**
```bash
# Check system audio devices
# Linux: arecord -l
# macOS: Audio MIDI Setup app
# Windows: Sound settings → Input devices

# Test microphone
# Linux: arecord -d 5 test.wav && aplay test.wav
# macOS: Built-in Voice Memos app
# Windows: Voice Recorder app

# Restart audio service if needed
# Linux: sudo systemctl restart pulseaudio
# macOS: sudo killall coreaudiod
```

### 4. Audio Quality Issues

#### Issue: "Choppy audio" or "Audio cuts out"

**Symptoms:**
- Distorted playback
- Audio stops mid-sentence
- Clicking/popping sounds

**Diagnosis:**
```javascript
// Check audio context state in browser console
console.log('Audio context state:', audioContext.state);
console.log('Sample rate:', audioContext.sampleRate);
console.log('Audio queue length:', audioQueue.length);
```

**Solutions:**
```python
# Solution 1: Adjust audio buffer sizes
# In working_voice_server.py:
def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    # Increase buffer size for better quality
    # Add noise reduction
    pcm_array = np.clip(pcm_array, -32767, 32767)
```

```javascript
// Solution 2: Improve audio processing in frontend
const processor = audioContext.createScriptProcessor(4096, 1, 1); // Larger buffer

// Solution 3: Add audio quality settings
const audioConstraints = {
    audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: true,
        highpassFilter: true  // Add this
    }
};
```

#### Issue: "Audio delay" or "Slow response"

**Solutions:**
```python
# Reduce silence duration for faster response
"silence_duration_ms": 50,  # Reduce from 100ms

# Optimize audio processing
def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    # Use faster processing
    if len(pcm_data) < 1024:  # Skip very small chunks
        return pcm_data
```

### 5. Language & Transcription Issues

#### Issue: "Wrong language detected" or "Poor transcription"

**Symptoms:**
- Responds in wrong language
- Garbled transcription
- Missing words

**Diagnosis:**
```bash
# Check which languages are supported
curl http://localhost:8000/status

# Test with clear pronunciation
# English: "Hello DhanKanya"
# Tamil: "வணக்கம் தன்கன்யா"
# Telugu: "నమస్తే ధనకన్య"
```

**Solutions:**
```python
# Solution 1: Improve system prompt for language detection
SYSTEM_PROMPT = """
LANGUAGE ADAPTATION:
- If user speaks in Tamil, respond primarily in Tamil with English financial terms
- If user speaks in Telugu, respond primarily in Telugu with English financial terms  
- If user speaks in Hindi, respond primarily in Hindi with English financial terms
- Always be natural and conversational, as if speaking to a friend
"""

# Solution 2: Add explicit language configuration
CONFIG = {
    "speech_config": {
        "language_code": "ta-IN"  # For Tamil
        # "language_code": "te-IN"  # For Telugu
        # "language_code": "hi-IN"  # For Hindi
    }
}
```

### 6. Performance Issues

#### Issue: "Slow startup" or "High memory usage"

**Diagnosis:**
```bash
# Check memory usage
ps aux | grep -E "(working_voice_server|streamlit)"
top -p $(pgrep -f working_voice_server)

# Check CPU usage
htop
```

**Solutions:**
```bash
# Solution 1: Optimize Python environment
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Solution 2: Increase system resources
# Add swap space if low memory
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Solution 3: Use production ASGI server
pip install gunicorn uvloop
gunicorn working_voice_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 7. Browser Compatibility Issues

#### Issue: "Feature not supported" or "WebSocket error"

**Diagnosis:**
```javascript
// Check browser capabilities
console.log('WebSocket support:', !!window.WebSocket);
console.log('Web Audio API support:', !!window.AudioContext);
console.log('getUserMedia support:', !!navigator.mediaDevices?.getUserMedia);
console.log('Browser:', navigator.userAgent);
```

**Solutions:**
```bash
# Update to supported browser version
# Chrome 80+, Firefox 76+, Edge 80+, Safari 14+

# Enable required features in browser
# Chrome: chrome://flags/#enable-experimental-web-platform-features
# Firefox: about:config → media.navigator.streams.fake

# Use polyfills for older browsers
<script src="https://cdn.jsdelivr.net/npm/audio-context-polyfill@1.0.0/audio-context-polyfill.js"></script>
```

### 8. Development Environment Issues

#### Issue: "Module not found" or "Import errors"

**Solutions:**
```bash
# Solution 1: Check virtual environment
which python  # Should point to .venv/bin/python
pip list | grep -E "(streamlit|fastapi|google-genai)"

# Solution 2: Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Solution 3: Check Python path
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 4: Install missing packages
pip install google-genai==1.20.0
pip install anthropic
pip install streamlit
pip install fastapi uvicorn
```

#### Issue: "Port already in use"

**Solutions:**
```bash
# Find and kill processes using ports
lsof -ti:8000 | xargs kill -9  # Kill voice backend
lsof -ti:8501 | xargs kill -9  # Kill Streamlit

# Use different ports
python working_voice_server.py --port 8002
streamlit run main.py --server.port 8502

# Configure ports in code
# working_voice_server.py line ~370:
uvicorn.run(app, host="0.0.0.0", port=8002)
```

## 🔧 Advanced Debugging

### Enable Debug Mode
```bash
# Run with maximum logging
python working_voice_server.py --log-level debug

# Streamlit debug mode
streamlit run main.py --logger.level debug --server.runOnSave true
```

### Network Debugging
```bash
# Test WebSocket connection manually
wscat -c ws://localhost:8000/ws

# Monitor network traffic
sudo tcpdump -i lo port 8000

# Check DNS resolution
nslookup localhost
ping localhost
```

### Audio Debugging
```javascript
// Browser console debugging
// Monitor audio processing
const originalLog = console.log;
console.log = function(...args) {
    if (args[0]?.includes('🎵') || args[0]?.includes('🎤')) {
        originalLog.apply(console, args);
    }
};
```

## 📊 Performance Monitoring

### Key Metrics to Monitor
```bash
# Response time (should be < 500ms)
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/status

# Memory usage (should be < 1GB per process)
ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem | head

# WebSocket connections
ss -tuln | grep 8000
```

### Log Analysis
```bash
# Extract errors from logs
grep -i "error\|exception\|failed" voice_backend.log | tail -20

# Monitor WebSocket connections
grep -i "websocket\|connected\|disconnected" voice_backend.log | tail -10

# Check audio processing
grep -i "audio\|pcm\|wav" voice_backend.log | tail -10
```

## 🆘 Emergency Recovery

### Complete Reset
```bash
# Stop all processes
pkill -f working_voice_server
pkill -f streamlit

# Clear cache and restart
rm -rf .streamlit/cache
rm -rf __pycache__
rm -rf app/__pycache__

# Reinstall environment
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Restart services
python working_voice_server.py &
streamlit run main.py
```

### Recovery Checklist
- [ ] API keys are correct and valid
- [ ] Both servers are running (ports 8000 and 8501)
- [ ] Virtual environment is activated
- [ ] Dependencies are installed
- [ ] Microphone permission granted
- [ ] Browser is supported (Chrome/Firefox/Edge)
- [ ] No firewall blocking connections
- [ ] Internet connection is stable

## 📞 Getting Additional Help

### Before Seeking Help
1. Check this troubleshooting guide thoroughly
2. Review the error messages in both terminal and browser console
3. Test with a different browser
4. Verify your API keys are working
5. Check system requirements are met

### Information to Include When Reporting Issues
```bash
# System information
uname -a                    # OS details
python --version           # Python version
pip list | grep -E "(streamlit|fastapi|google)"  # Package versions

# Error logs
tail -50 voice_backend.log  # Recent backend logs
# Browser console errors (screenshot)

# Configuration
cat .env | sed 's/=.*/=***/'  # API key status (masked)
curl http://localhost:8000/status  # Backend status
```

### Contact Channels
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check README.md and VOICE_ARCHITECTURE.md
- **Community**: GitHub Discussions for questions

---

**Remember**: Most issues can be resolved by restarting the services and checking API keys! 🔄
