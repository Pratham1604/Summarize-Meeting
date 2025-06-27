import os
import math
import time
import logging
import ffmpeg
import spacy
from dotenv import load_dotenv
from pydub import AudioSegment
from sarvamai import SarvamAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load environment variables
load_dotenv()
SARVAM_KEY = os.getenv("SARVAM_SUBSCRIPTION_KEY", "ed6a5d10-efe1-4f3d-82a3-5506f370e557")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBmmtpjB2xfNqaeJ_hSA8xh1LD8DxQkRS8")

client = SarvamAI(api_subscription_key=SARVAM_KEY)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def convert_to_wav(input_path, output_path="audio.wav"):
    logging.info(f"Converting to WAV: {input_path}")
    ffmpeg.input(input_path).output(output_path, ac=1, ar=16000).overwrite_output().run()
    logging.info(f"Converted to WAV: {output_path}")
    return output_path

def get_user_file():
    media_type = input("Is your file audio or video? (Enter 'audio' or 'video'): ").strip().lower()
    file_path = input("Enter the full path to your file: ").strip()

    if not os.path.exists(file_path):
        logging.error("File not found. Please check the path and try again.")
        exit()

    audio_extensions = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]
    file_ext = os.path.splitext(file_path)[1].lower()

    if media_type == "audio" and file_ext in audio_extensions:
        return convert_to_wav(file_path)
    elif media_type == "video":
        return convert_to_wav(file_path)
    else:
        logging.error("Unsupported file type or wrong media type input.")
        exit()

def split_audio(file_path, chunk_length_sec=30):
    logging.info(f"Splitting audio into {chunk_length_sec}s chunks...")
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    duration_sec = len(audio) // 1000
    total_chunks = math.ceil(duration_sec / chunk_length_sec)
    base = os.path.splitext(file_path)[0]

    for i in range(total_chunks):
        start = i * chunk_length_sec * 1000
        end = min((i + 1) * chunk_length_sec * 1000, len(audio))
        chunk = audio[start:end]
        chunk_path = f"{base}_part{i+1}.wav"
        chunk.export(chunk_path, format="wav")
        logging.info(f"Saved chunk: {chunk_path}")
        chunks.append(chunk_path)

    return chunks

def transcribe_audio(audio_path):
    logging.info("Transcribing audio using Sarvam AI...")
    chunks = split_audio(audio_path)
    transcript = []

    for chunk in chunks:
        logging.info(f"Transcribing chunk: {chunk}")
        with open(chunk, "rb") as f:
            try:
                result = client.speech_to_text.translate(file=f, model="saaras:v2")
                transcript.append(result.transcript)
            except Exception as e:
                logging.error(f"Error with chunk {chunk}: {e}")
                continue
        time.sleep(1.5)

        try:
            os.remove(chunk)
            logging.info(f"Deleted chunk: {chunk}")
        except Exception as e:
            logging.warning(f"Could not delete chunk {chunk}: {e}")

    if not transcript:
        logging.error("Transcript is empty. Exiting.")
        exit()

    full_text = " ".join(transcript)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    logging.info("Transcript saved to transcript.txt")

    return full_text


def summarize_text_with_gemini(text):
    logging.info("Summarizing transcription using Gemini...")
    max_input_length = 30000
    chunk_size = max_input_length
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    all_summaries = []

    for i, chunk in enumerate(chunks):
        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_tokens=2048,
            google_api_key=GEMINI_API_KEY
        )

        prompt = f"Summarize the following meeting transcript into bullet points with speaker names:\n\n{chunk}"
        try:
            summary = llm.invoke(prompt)
            summary_text = summary.content if hasattr(summary, 'content') else str(summary)
            all_summaries.append(summary_text)
        except Exception as e:
            logging.error(f"Gemini LLM error on chunk {i+1}: {e}")
            all_summaries.append(f"[Error summarizing chunk {i+1}]")

    return "\n\n".join(all_summaries)

def main():
    logging.info("Starting meeting transcription pipeline...")
    audio_path = get_user_file()
    transcript = transcribe_audio(audio_path)
    summary = summarize_text_with_gemini(transcript)

    logging.info("\n=== Final Meeting Summary ===")
    print(f"\n{summary}")

if __name__ == "__main__":
    main()
