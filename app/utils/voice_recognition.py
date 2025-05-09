"""
Voice recognition utilities for the DhanKanya application.

This module contains functions for capturing voice input from users
and converting it to text for processing by the application.
"""

import streamlit as st
import speech_recognition as sr
import logging

logger = logging.getLogger(__name__)

def get_voice_input() -> str:
    """
    Capture voice input from the user and convert it to text.
    
    Returns:
        The recognized text from the user's voice input,
        or an error message if recognition fails.
    """
    r = sr.Recognizer()
    
    with st.spinner("Listening... (speak after the beep)"):
        try:
            with sr.Microphone() as source:
                st.write("🔊 Beep!")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                st.success("Processing your voice...")
                
            # Try to recognize Hindi first, then English if that fails
            try:
                text = r.recognize_google(audio, language="hi-IN")
                logger.info(f"Recognized Hindi: {text}")
                return text
            except:
                text = r.recognize_google(audio)
                logger.info(f"Recognized English: {text}")
                return text
                
        except sr.WaitTimeoutError:
            logger.error("Voice recognition timed out - no speech detected")
            return "I couldn't hear anything. Please try again."
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service: {e}")
            return "I'm having trouble connecting to the speech recognition service. Please type your question instead."
            
        except Exception as e:
            logger.error(f"Voice recognition error: {str(e)}")
            return "I couldn't understand that. Please try again or type your question." 