import pyaudio
import webrtcvad
import collections
import wave
import sys
import whisper
import numpy as np
import threading
import atexit

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 800
VAD_FRAMES = 10  # Number of frames for VAD
VAD_DURATION = CHUNK * VAD_FRAMES / RATE  # Duration of each VAD frame in seconds

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# Initialize Whisper model
model = whisper.load_model("base")

# Class to represent a "frame" of audio data
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

# Function to read audio data from PyAudio stream
def read_audio_data(stream):
    return stream.read(CHUNK)

# Function to generate audio frames from PCM audio data
def frame_generator2(stream, frame_duration_ms, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    audio_data = read_audio_data(stream)
    print(len(audio_data))
    while offset + n < len(audio_data):
        yield Frame(audio_data[offset:offset + n], timestamp, duration)
        print("YEILD")
        timestamp += duration
        offset += n

# Function to filter out non-voiced audio frames
def vad_collector(vad, frames):
    ring_buffer = collections.deque(maxlen=VAD_FRAMES)
    triggered = False
    voiced_frames = []
    print(len(frames))
    for frame in frames:
        # Convert the bytes to a numpy array of 16-bit integers
        audio_data = np.frombuffer(frame.bytes, dtype=np.int16)
        # Convert the numpy array back to bytes
        audio_bytes = audio_data.tobytes()
        # Check if the audio is speech
        is_speech = vad.is_speech(audio_bytes, sample_rate=RATE)

        sys.stdout.write('1' if is_speech else '0')
        sys.stdout.flush()

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                print("TRIGGERED")
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                sys.stdout.flush()
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                sys.stdout.flush()
                print("DE-TRIGGERED")
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

# Function to transcribe audio data using Whisper
def transcribe_audio(audio_data):
    print("transcribing...")
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    result = model.transcribe(audio_array.tobytes(), fp16=False)
    print(result["text"])

# Function to continuously process audio
def continuous_audio_processing(stream):
    while True:
        frames = frame_generator2(stream, 10, RATE)
        for segment in vad_collector(vad, frames):
            transcribe_audio(segment)

# Main function to start the audio processing
def main():
    try:
        # Open the audio stream
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        # Start a new thread for continuous audio processing
        audio_thread = threading.Thread(target=continuous_audio_processing, args=(stream,))
        audio_thread.start()

        # Wait for the thread to finish (which should be forever)
        audio_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the audio stream#
        print("done")
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio.terminate()

if __name__ == '__main__':
    main()