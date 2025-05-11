# DhanKanya

DhanKanya is a financial assistant that provides expert financial guidance, enhancing your financial literacy and addressing your needs. It is developed by the Finance team at 100GIGA and powered by Anthropic's Claude AI model.

## Features

- **Multilingual AI Assistant**: Ask questions about financial topics in English and Indian languages
- **State-Specific Financial Information**: Access scholarships, loans, and government schemes tailored to your state
- **Expense Tracking & Budgeting**: Track your income, expenses, and savings goals
- **Financial Goal Setting**: Set and monitor progress toward your financial objectives

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
├── .env.template           # Template for environment variables
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

1. Copy the `.env.template` file to create your own `.env` file:
   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
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

