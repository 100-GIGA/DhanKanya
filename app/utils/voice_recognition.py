import streamlit as st
import speech_recognition as sr
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configuration Variables
ENERGY_THRESHOLD = 200
PAUSE_THRESHOLD = 0.8
PHRASE_THRESHOLD = 0.2
NON_SPEAKING_DURATION = 0.4
TIMEOUT = 8
PHRASE_TIME_LIMIT = 12

@contextmanager
def streamlit_spinner(text="Processing..."):
    with st.spinner(text):
        yield

def configure_recognizer():
    """Configures the recognizer with optimal settings."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = PAUSE_THRESHOLD
    recognizer.phrase_threshold = PHRASE_THRESHOLD
    recognizer.non_speaking_duration = NON_SPEAKING_DURATION
    return recognizer

def capture_audio(recognizer):
    """Captures audio from the microphone."""
    with sr.Microphone() as source:
        st.write("🔊 Listening... Speak clearly.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
        return audio

def recognize_speech(recognizer, audio):
    """Attempts to recognize speech using Google Speech Recognition."""
    try:
        text = recognizer.recognize_google(audio, language="hi-IN,en-IN")
        logger.info(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("Speech was not clear or not recognized.")
        return "I couldn't understand that. Please try again."
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service: {e}")
        return "I'm having trouble connecting to the speech service. Please try again later."

def get_voice_input():
    """Main function to capture and process voice input."""
    recognizer = configure_recognizer()
    with streamlit_spinner("Listening..."):
        try:
            audio = capture_audio(recognizer)
            return recognize_speech(recognizer, audio)
        except sr.WaitTimeoutError:
            logger.error("Timeout: No speech detected.")
            return "I couldn't hear anything. Please try again."
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred. Please try again."

