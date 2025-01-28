# DhanKanya

DhanKanya is a financial assistant that provides expert financial guidance, enhancing your financial literacy and addressing your needs. It is developed by the Finance team at 100GIGA and powered by Anthropic’s Claude AI model.

## Installation

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

**Note:** If you are running a MacOS with an ARM chip (M series), PyAudio installation might fail. This is due to missing portaudio dependencies. To resolve this, follow these steps:

1. Upgrade pip, setuptools, and wheel:

    ```bash
    python3 -m pip install --upgrade pip setuptools wheel
    ```

2. Install the portaudio package via Homebrew:

    ```bash
    brew install portaudio
    ```

3. Install PyAudio with additional build options:

    ```bash
    python3 -m pip install pyaudio --global-option="build_ext" --global-option="-I/opt/homebrew/include" --global-option="-L/opt/homebrew/lib"
    ```

Once these steps are completed, you should be able to run the application without issues.

## Usage

Run the following command to start the application:

```bash
streamlit run dhanKanya.py
```

