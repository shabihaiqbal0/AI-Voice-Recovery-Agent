# 🎤 AI Voice Accessibility Agent

An AI-powered accessibility application that converts unclear speech into clear communication.

## Features

- 🎙️ Record voice or upload audio
- 🧠 Whisper AI speech recognition
- ✨ AI-powered text correction (FLAN-T5)
- 📝 User review before sending
- 🔊 Human-like voice generation (gTTS)
- 🌐 Modern Streamlit web interface

## Workflow

```
Voice Input → Whisper Speech Recognition → AI Text Correction → Human Voice Output
```

## Tech Stack

- **Python 3.10+**
- **Streamlit 1.28+** - Web interface
- **OpenAI Whisper** - Speech-to-text
- **Hugging Face Transformers** - AI text correction (FLAN-T5-small)
- **Google Text-to-Speech (gTTS)** - Text-to-speech
- **PyTorch** - ML framework

## System Requirements

- FFmpeg (required for audio processing)
  - Windows: Download from https://ffmpeg.org/download.html
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/shabihaiqbal0/AI-Voice-Recovery-Agent.git
cd AI-Voice-Recovery-Agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open in browser**
- Local: http://localhost:8501

## How to Use

1. **Record or upload** your voice message
2. **Click "Process Voice"** to transcribe and correct the text
3. **Review and edit** the corrected message if needed
4. **Click "Generate Human Voice"** to convert text to speech

## Deployment

### Streamlit Community Cloud

This project is ready for Streamlit Community Cloud deployment:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Select `streamlit_app.py` as the main file
6. Click "Deploy"

The app will automatically deploy and be publicly accessible.

## License

MIT License