# Gemma Voice Assistant

A local voice assistant powered by Gemma-3 LLM with speech-to-text (Whisper) and text-to-speech (Piper).

## Features

- 🎙️ **Voice Input**: Hold-to-talk or toggle recording
- 💬 **Text Chat**: Type messages directly
- 🗣️ **Voice Output**: Natural speech synthesis
- 🔒 **100% Local**: All processing happens on your machine
- 🚀 **Fast Response**: Lightweight Gemma-3 model

## Prerequisites

1. **Python 3.8+**
2. **LM Studio** - Download from [lmstudio.ai](https://lmstudio.ai/)
3. **FFmpeg** - For audio processing

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/TechNavii/Small_Model_Voice_Assistant.git
cd Small_Model_Voice_Assistant
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Piper TTS**
```bash
pip install piper-tts
```

4. **Download Piper voice model**

The voice model files are included in `piper_models/` directory:
- `en_US-amy-medium.onnx` (63MB)
- `en_US-amy-medium.onnx.json`

If missing, download from [Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium).

## Setup LM Studio

1. Open LM Studio
2. Download model: `lmstudio-community/gemma-3-270m-it-MLX-8bit`
3. Start local server (default port 1234):
   - Go to "Local Server" tab
   - Load the Gemma-3 model
   - Click "Start Server"

## Usage

1. **Start LM Studio** with Gemma-3 model loaded

2. **Run the server**
```bash
python app.py
```

3. **Open browser**
```
http://localhost:8000
```

### Controls

- **Hold to Talk**: Press and hold the button, speak, then release
- **Start Recording**: Toggle recording on/off
- **Text Input**: Type and press Enter

## Project Structure

```
Small_Model_Voice_Assistant/
├── app.py                    # FastAPI server
├── requirements.txt          # Python dependencies
├── piper_models/            # TTS voice models
│   ├── en_US-amy-medium.onnx
│   └── en_US-amy-medium.onnx.json
└── static/
    └── index.html           # Web interface
```

## Troubleshooting

**LM Studio connection error**
- Ensure LM Studio server is running on port 1234
- Check that Gemma-3 model is loaded

**No audio output**
- Check Piper installation: `pip show piper-tts`
- Verify voice model files exist in `piper_models/`

**Microphone not working**
- Allow microphone access in browser
- Check system audio permissions

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Piper](https://github.com/rhasspy/piper) for text-to-speech
- [LM Studio](https://lmstudio.ai/) for local LLM hosting
- [Gemma](https://ai.google.dev/gemma) by Google
