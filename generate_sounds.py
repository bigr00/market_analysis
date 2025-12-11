#!/usr/bin/env python3
"""
Generate simple notification sounds for the web interface
Creates MP3-compatible WAV files
"""
import struct
import wave
import math
import os


def generate_tone(filename: str, frequencies: list, duration: float = 0.15,
                  sample_rate: int = 44100, volume: float = 0.5):
    """
    Generate a simple tone and save as WAV file

    Args:
        filename: Output filename
        frequencies: List of frequencies to play in sequence
        duration: Duration of each tone in seconds
        sample_rate: Audio sample rate
        volume: Volume (0.0 to 1.0)
    """
    samples = []

    for freq in frequencies:
        num_samples = int(sample_rate * duration)

        for i in range(num_samples):
            # Generate sine wave
            t = i / sample_rate
            value = volume * math.sin(2 * math.pi * freq * t)

            # Apply fade in/out to avoid clicks
            fade_samples = int(sample_rate * 0.01)  # 10ms fade
            if i < fade_samples:
                value *= i / fade_samples
            elif i > num_samples - fade_samples:
                value *= (num_samples - i) / fade_samples

            samples.append(value)

        # Small gap between tones
        gap_samples = int(sample_rate * 0.02)
        samples.extend([0] * gap_samples)

    # Convert to 16-bit integers
    max_amplitude = 32767
    int_samples = [int(s * max_amplitude) for s in samples]

    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for sample in int_samples:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"Generated: {filename}")


def main():
    sounds_dir = os.path.join(os.path.dirname(__file__), 'static', 'sounds')
    os.makedirs(sounds_dir, exist_ok=True)

    # Bullish sound - ascending tones (C5, E5, G5)
    bullish_file = os.path.join(sounds_dir, 'bullish.wav')
    generate_tone(bullish_file, [523, 659, 784], duration=0.12, volume=0.4)

    # Bearish sound - descending tones (G4, E4, C4)
    bearish_file = os.path.join(sounds_dir, 'bearish.wav')
    generate_tone(bearish_file, [392, 330, 262], duration=0.12, volume=0.4)

    print("\nSound files generated!")
    print("Note: These are WAV files. For better browser compatibility,")
    print("you may want to convert them to MP3 using ffmpeg:")
    print("  ffmpeg -i bullish.wav bullish.mp3")
    print("  ffmpeg -i bearish.wav bearish.mp3")


if __name__ == '__main__':
    main()
