import pyaudio
import webrtcvad
import collections
import wave
import sys
import whisper
import numpy as np
import threading
import atexit
import torch

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
vad.set_mode(2)

# Initialize Whisper model
model = whisper.load_model("base.en")

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
    while offset + n < len(audio_data):
        yield Frame(audio_data[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

# Function to transcribe audio data using Whisper
def transcribe_audio(audio_data):
    print("transcribing...")
    audio_array = torch.tensor(np.frombuffer(audio_data, dtype=np.float32))
    
    result = model.transcribe(audio_array, temperature=0)
    print(result["text"])

# Function to continuously process audio
def continuous_audio_processing(stream):
    while True:
        frames = frame_generator2(stream, 10, RATE)
        for segment in vad_collector(RATE, 10, 30, vad, frames):
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