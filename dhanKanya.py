import streamlit as st
import anthropic
import pandas as pd
import datetime
import speech_recognition as sr
import re 
import os
import logging

import sys
import traceback

from dotenv import load_dotenv
import traceback
import json

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(dotenv_path='.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = 'chroma'
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY') # Get the API key from .env file using dotenv library
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

PROMPT_TEMPLATE = """
Answer the question so that it is easily understandable. The context is provided so that you can take reference from this. Please take inspiration from the context. You can also add things that you think are helpful for girls out there. Do not mention about the context provided. Answer as you usually answer.

{context}

---

{question}
"""

INTRODUCTION_PROMPTS = [
    r'introduce yourself',
    r'tell me about yourself',
    r'who are you',
    r'what can you do',
    r'how can you help me',
    r'what are your capabilities',
    r'what kind of tasks can you assist with',
    r'what are you capable of',
    r'what can i ask you',
    r'what are you good at?',
    r'what are your specialties?',
    r'what is your purpose?',
    r'what are you designed for?',
    r'how do i get started with you?',
    r'how should i interact with you?',
    r'who created you'
    r'hello',
    r'hey',
    r'namaste'
]

# def main():
#     st.set_page_config(page_title="Financial Advisor for Your Dreams", page_icon=":moneybag:", layout="wide")

#     # Create a navigation bar
#     menu = ["Home", "Scholarships & Schemes per your State", "Build Your Wealth"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     # Create the Anthropic client with the API key
#     try:
#         # Add debug logging
#         logger.info(f"ANTHROPIC_API_KEY exists: {bool(ANTHROPIC_API_KEY)}")
#         logger.info("Attempting to create Anthropic client")
        
#         client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
#         logger.info("Successfully created Anthropic client")
#         st.sidebar.success("AI assistant initialized successfully!")
#     except Exception as e:
#         logger.error(f"Error creating Anthropic client: {e}")
#         logger.error(f"Full traceback: {traceback.format_exc()}")
#         st.sidebar.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
#         return

#     # Display the selected page
#     if choice == "Home":
#         home_page(client)
#     elif choice == "Scholarships & Schemes per your State":
#         templates_page(client)
#     elif choice == "Build Your Wealth":
#         expense_tracker_page()

# Validate that .env file exists and its structure is correct
def check_env_file():
    logger.info("=== Checking .env File ===")
    if not os.path.exists('.env'):
        logger.error(".env file does not exist.")
        raise FileNotFoundError(".env file is missing.")

    with open('.env', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            if '=' not in line:
                logger.error(f"Invalid line in .env file: {line}")
                raise ValueError(f"Invalid line in .env file: {line}")
        logger.info("Valid .env file structure.")

def main():
    st.set_page_config(page_title="Financial Advisor for Your Dreams", page_icon=":moneybag:", layout="wide")

    # Diagnostic logging
    logger.info("=== Starting Application Diagnostics ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Anthropic package version: {anthropic.__version__}")
    logger.info(f"Running on Streamlit version: {st.__version__}")
    
    # Log environment info
    logger.info("=== Environment Check ===")
    is_cloud = os.getenv('STREAMLIT_DEPLOYMENT_RUNTIME') == 'cloud'
    logger.info(f"Running on Streamlit Cloud: {is_cloud}")
    
    # Check for proxy-related environment variables
    logger.info("=== Proxy Configuration Check ===")
    proxy_vars = {k: '***' for k, v in os.environ.items() 
                 if 'PROXY' in k.upper() or 'HTTP' in k.upper()}
    logger.info(f"Proxy-related environment variables: {proxy_vars}")

    # Check .env file
    check_env_file()

    # Create a navigation bar
    menu = ["Start with Voice", "Build your Wealth", "Savings and Budgeting"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Create the Anthropic client with extra error handling
    try:
        logger.info("Attempting to create Anthropic client...")
        
        # Force disable any proxy settings
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        
        # Create client with minimal configuration
        client_config = {
            'api_key': ANTHROPIC_API_KEY
        }
        logger.info("Client configuration prepared (API key masked)")
        
        client = anthropic.Anthropic(**client_config)
        logger.info("Successfully created Anthropic client")
        st.sidebar.success("AI assistant initialized successfully!")
        
    except Exception as e:
        logger.error("=== Anthropic Client Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        st.sidebar.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
        return

    # Display the selected page
    if choice == "Start with Voice":
        home_page(client)
    elif choice == "Build your Wealth":
        templates_page(client)
    elif choice == "Savings and Budgeting":
        expense_tracker_page()

def home_page(client):
    st.title("DhanKanya: Financial Empowerment for Girls in India")

    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=200)

    st.write("""
    ### Welcome to our AI-powered financial literacy application!

    Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.

    With our user-friendly app, you'll have access to:
    """)

    st.write("- **Interactive Budgeting Tools** to help you track your income and expenses.")
    st.write("- **Educational Resources** on essential financial literacy concepts like saving and investing.")
    st.write("- **Goal Setting Functionality** to plan and save for specific educational milestones.")

    st.markdown("---")

    if "claude_model" not in st.session_state:
        st.session_state["claude_model"] = "claude-3-haiku-20240307"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add voice input functionality
    st.write("You can ask questions in Hindi using your voice or type them in English.")
    voice_input = st.button("🎙️ Use voice input")

    if voice_input:
        prompt = get_voice_input()
    else:
        prompt = st.chat_input("Ask a question in English")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_response(prompt, client)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_templates():
    try:
        with open('state_templates.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("state_templates.json file not found")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding state_templates.json")
        return {}

def templates_page(client):
    st.title("Sample Prompts depending on the state that you reside in.")
    
    # Templates for different states
    templates  = load_templates()
    
    states = list(templates.keys())
    if not states:
        st.error("No templates available. Please check the configuration file.")
        return

    selected_state = st.selectbox("Select your state in India", states)

    if selected_state:
        st.write(f"Selected State: {selected_state}")
        st.write(f"Choose from the following prompt templates related to {selected_state}:")
        
        state_templates = templates.get(selected_state, {})
        for category, prompts in state_templates.items():
            st.subheader(category)
            for prompt in prompts:
                if st.button(prompt):
                    response = get_response(prompt, client)
                    st.markdown(response)

    logger.info(f"User selected state: {selected_state}")

def initialize_session():
    """Initialize session state variables if not already set."""
    if "expenses" not in st.session_state:
        st.session_state.expenses = []  # List to store expense records
    if "expense_summary" not in st.session_state:
        st.session_state.expense_summary = {"total": 0, "necessary": 0, "avoidable": 0}
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = datetime.datetime.now().strftime("%Y-%m")

def add_expense(date, description, amount, category):
    """Adds an expense entry to the session state and updates summary."""
    expense_entry = {
        "date": date.strftime("%Y-%m-%d"),
        "description": description,
        "amount": amount,
        "category": category
    }
    
    st.session_state.expenses.append(expense_entry)
    st.session_state.expense_summary["total"] += amount
    if category == "Necessary":
        st.session_state.expense_summary["necessary"] += amount
    else:
        st.session_state.expense_summary["avoidable"] += amount

    logger.info(f"Added expense: {expense_entry}")
    st.success("Expense added successfully!")

def display_expense_summary():
    """Displays total, necessary, and avoidable expenses."""
    st.subheader("Expense Summary")
    st.markdown(f"### Total: ₹{st.session_state.expense_summary['total']:.2f}")
    st.markdown(f"✅ **Necessary:** ₹{st.session_state.expense_summary['necessary']:.2f}")
    st.markdown(f"⚠️ **Avoidable:** ₹{st.session_state.expense_summary['avoidable']:.2f}")
    st.markdown("---")

def display_expense_table():
    """Displays recorded expenses in a structured table."""
    if not st.session_state.expenses:
        st.info("No expenses recorded yet.")
        return

    df_expenses = pd.DataFrame(st.session_state.expenses)
    df_expenses["amount"] = df_expenses["amount"].apply(lambda x: f"₹ {x:.2f}")
    st.subheader("Expense History")
    st.dataframe(df_expenses.set_index("date"), use_container_width=True)

def expense_tracker_page():
    """Main function to render the Expense Tracker page."""
    st.title("💰 Expense Tracker")
    initialize_session()

    # Expense entry form
    with st.form("expense_form"):
        expense_date = st.date_input("Date", value=datetime.date.today())
        expense_description = st.text_input("Description")
        expense_amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
        expense_category = st.selectbox("Category", ["Necessary", "Avoidable"])
        submitted = st.form_submit_button("Add Expense")

        if submitted:
            if expense_description and expense_amount > 0:
                add_expense(expense_date, expense_description, expense_amount, expense_category)
            else:
                st.error("Please enter a valid description and amount.")

    display_expense_summary()
    display_expense_table()


def get_voice_input():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

        try:
            # Urdu language support
            try:
                urdu_text = r.recognize_google(audio, language='ur-PK')
            except sr.UnknownValueError:
                urdu_text = None

            # Telugu language support
            try:
                telugu_text = r.recognize_google(audio, language='te-IN')
            except sr.UnknownValueError:
                telugu_text = None

            # Hindi language support
            hindi_text = r.recognize_google(audio, language='hi-IN')

            # Return the recognized text in the correct language
            if hindi_text or telugu_text or urdu_text:
                if hindi_text:
                    return hindi_text
                elif telugu_text:
                    return telugu_text
                elif urdu_text:
                    return urdu_text
            else:
                st.error("Sorry, I could not understand your voice input in any supported language. Please try again.")

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your voice input. Please try again.")

        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")

    return None

def query(query_text, client):
    try:
        # Prepare the DB with HuggingFaceBgeEmbeddings.
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            logger.info("Direct query to Claude")
            answer = get_response(query_text, client)
            return answer

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        logger.info("Query with context")
        response = get_response(prompt, client)
        return response
    except Exception as e:
        logger.error(f"Error in query function: {e}")
        logger.error(traceback.format_exc())
        return "I'm sorry, but I encountered an error while processing your query. Please try again later."


def get_response(prompt, client):
    """
    Retrieves response from the Anthropic model based on the prompt.
    """
    try:
        prompt_lower = prompt.lower()
        for pattern in INTRODUCTION_PROMPTS:
            if re.search(pattern, prompt_lower):
                return "Namaste! I'm your financial assistant, developed by the Finance team at 100GIGA and powered by Anthropic's Claude AI model. My purpose is to provide you with expert financial guidance, enhancing your financial literacy and addressing your needs. Feel free to ask me anything related to finance, and I'll be here to assist you every step of the way."
        
        # Updated message creation syntax for anthropic v0.42.0
        message = client.messages.create(
            model=st.session_state["claude_model"],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=600
        )
        return message.content[0].text

    except Exception as e:
        logger.error(f"Error getting response from Claude: {e}")
        logger.error(traceback.format_exc())
        return f"I apologize, but I'm having trouble generating a response at the moment. \nError details: {str(e)}"

if __name__ == "__main__":
    main()