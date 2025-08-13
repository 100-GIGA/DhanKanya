# DhanKanya

DhanKanya is a financial assistant that provides expert financial guidance, enhancing your financial literacy and addressing your needs. It is developed by the Finance team at 100GIGA and powered by Anthropic's Claude AI model and Google's Gemini Live API for voice interactions.

## Features

- **Multilingual AI Assistant**: Ask questions about financial topics in English and Indian languages
- **Voice Interactions**: Now supports Tamil and English voice conversations using Gemini Live API
- **Text + Voice Hybrid Mode**: Seamlessly switch between text and voice interactions
- **State-Specific Financial Information**: Access scholarships, loans, and government schemes tailored to your state
- **Expense Tracking & Budgeting**: Track your income, expenses, and savings goals
- **Financial Goal Setting**: Set and monitor progress toward your financial objectives
- **Voice Conversation History**: All voice interactions are stored for your reference

## New Voice Features 🎤

### Supported Languages
- **Tamil (தமிழ்)**: Native Tamil language support for voice interactions
- **English**: Full English voice conversation support

### Voice Capabilities
- Voice message processing and transcription
- Language auto-detection (Tamil/English)
- Voice conversation history storage
- Hybrid text+voice chat interface
- Context-aware financial advice in voice format

## Project Structure

```
DhanKanya/
├── app/                    # Core application code
│   ├── api/                # API endpoints
│   ├── components/         # UI components
│   ├── models/             # Data models
│   ├── services/           # Business logic services
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── tests/                  # Test files
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── static/                 # Static files
├── assets/                 # Asset files (images, etc.)
├── .streamlit/             # Streamlit configuration
│   └── secrets.toml.template # Template for Streamlit secrets
├── requirements.txt        # Project dependencies
└── main.py                 # Application entry point
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DhanKanya.git
   cd DhanKanya
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create the `.streamlit` directory if it doesn't exist:
   ```bash
   mkdir -p .streamlit
   ```

2. Copy the `secrets.toml.template` file to create your own `secrets.toml` file:
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```

3. Edit the `secrets.toml` file and add your Anthropic API key:
   ```toml
   ANTHROPIC_API_KEY = "your_api_key_here"
   ```

## Usage

Run the following command to start the application:

```bash
streamlit run main.py
```

The application will open in your default web browser.

## Development

- Follow the established code style guidelines
- Write comprehensive tests for new features
- Document your code with docstrings
- Maintain clean, modular code structure

## License

This project is proprietary and confidential. All rights reserved.

