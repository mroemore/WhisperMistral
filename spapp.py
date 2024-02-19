import sounddevice as sd
import numpy as np
import whisper
import sys
import torch

# Configuration
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILENCE_THRESHOLD = 0.01
WINDOW_SIZE = 1024
OVERLAP = 512
MAX_SEGMENT_DURATION = 30  # Maximum audio segment duration in seconds for processing

# Load Whisper model
model = whisper.load_model("tiny").to(DEVICE)

def is_silent(audio_data, silence_threshold=SILENCE_THRESHOLD, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """Determine if a given audio data is silent."""
    step = window_size - overlap
    num_windows = (len(audio_data) - overlap) // step
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = audio_data[start:end]
        rms = np.sqrt(np.mean(window**2))
        if rms > silence_threshold:
            return False
    return True

def transcribe_audio(audio_np):
    """Transcribe the given audio data using the Whisper model."""
    try:
        audio_np = audio_np / np.max(np.abs(audio_np))  # Normalize audio
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(DEVICE)
        result = model.transcribe(audio_tensor)
        return result['text']
    except RuntimeError as e:
        print(f"RuntimeError during transcription: {e}", file=sys.stderr)
        return "Transcription error."

def segment_audio(audio_data, sample_rate=SAMPLE_RATE, max_duration=MAX_SEGMENT_DURATION):
    """Segment the audio data into manageable chunks."""
    max_samples = max_duration * sample_rate
    for start in range(0, len(audio_data), max_samples):
        end = start + max_samples
        yield audio_data[start:end]

def process_audio(audio_data):
    """Process and transcribe audio data in segments."""
    transcriptions = []
    for segment in segment_audio(np.concatenate(audio_data, axis=0)):
        if not is_silent(segment):
            transcription = transcribe_audio(segment)
            transcriptions.append(transcription)
        else:
            print("Detected silence in segment, skipping transcription.")
    return " ".join(transcriptions)

def audio_callback(indata, frames, time, status, audio_data):
    """Callback function for each block of audio data."""
    if status:
        print(status, file=sys.stderr)
    audio_data.append(indata.copy())

if __name__ == "__main__":
    audio_data = []
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, device=None, channels=1, dtype='float32',
                            callback=lambda indata, frames, time, status: audio_callback(indata, frames, time, status, audio_data)):
            print("Recording... Speak into the microphone.")
            # Example control logic to stop recording, could be time-based or triggered by user action
            input("Press Enter to stop recording...\n")
            
            print("Processing and transcribing audio...")
            transcription = process_audio(audio_data)
            if transcription:
                print("Transcription:", transcription)
            else:
                print("No transcription available.")
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
