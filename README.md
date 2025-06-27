# Meeting Note Taker

An automated tool for processing meeting recordings to generate transcriptions and summaries using Sarvam AI and Google's Gemini model, with a user-friendly Streamlit interface.

## Features

- Audio/Video Support: Process various audio and video formats (mp3, wav, m4a, aac, mp4, mov, mkv)
- Transcription: High-quality transcription using Sarvam AI's speech-to-text API
- Summarization: Smart meeting summarization using Google's Gemini model
- Export Options: Download transcripts and summaries in both DOCX and PDF formats
- User Interface: Easy-to-use Streamlit web interface
- Parallel Processing: Optimized transcription with parallel processing

## Directory Structure

```
meeting_note_taker/
│
├── audio/            # Store uploaded meeting recordings
├── diarization/      # Diarized speaker segments
├── transcripts/      # Transcriptions per speaker
├── translations/     # Translated segments to English
├── summaries/        # Final summarized notes per speaker
│
├── main.py          # Pipeline runner
├── diarize.py       # Speaker diarization logic
├── transcribe.py    # Whisper-based transcription
├── translate.py     # Argos-based translation
├── summarize.py     # T5-based summarization
├── utils.py         # Common utility functions
└── requirements.txt # Dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
SARVAM_SUBSCRIPTION_KEY=your_sarvam_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_parallel.py
```

2. Through the web interface:
   - Upload your audio/video file
   - Wait for transcription and summarization
   - Download results in your preferred format (DOCX/PDF)

## Requirements

- Python 3.8+
- FFmpeg (for audio conversion)
- Internet connection (for API access)

## APIs Used

- Transcription: Sarvam AI Speech-to-Text
- Summarization: Google Gemini 2.0
- UI Framework: Streamlit

## License

MIT License 