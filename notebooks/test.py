import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

# Load the audio file with time range parameters
audio_file = "../data/inputs/original-demo-speech.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# Define start and end times in seconds (ss = start seconds, es = end seconds)
ss = 10.0  # start at 10 seconds
es = 30.0  # end at 30 seconds

# Convert time to samples and extract the specified portion
start_sample = int(ss * sample_rate)
end_sample = int(es * sample_rate)

# Make sure end_sample doesn't exceed the waveform length
end_sample = min(end_sample, waveform.shape[1])

# Extract the specified portion of the audio
waveform = waveform[:, start_sample:end_sample]

print(f"Processing audio from {ss:.2f}s to {es:.2f}s")
print(f"Selected duration: {waveform.shape[1] / sample_rate:.2f} seconds")

# Print basic audio information
print(f"Sample rate: {sample_rate} Hz")
print(f"Audio duration: {waveform.shape[1] / sample_rate:.2f} seconds")
print(f"Number of channels: {waveform.shape[0]}")
print(f"Max amplitude: {waveform.abs().max().item():.6f}")
print(f"Min amplitude: {waveform.abs().min().item():.6f}")
print(f"Mean amplitude: {waveform.abs().mean().item():.6f}")

# Calculate RMS (Root Mean Square) energy over time
window_size = int(0.05 * sample_rate)  # 50ms window
hop_length = int(0.025 * sample_rate)  # 25ms hop
rms_values = []
time_values = []

for i in range(0, waveform.shape[1] - window_size, hop_length):
    chunk = waveform[:, i : i + window_size]
    rms = torch.sqrt(torch.mean(chunk**2)).item()
    rms_values.append(rms)
    time_values.append(i / sample_rate)

# Convert to dB for better visualization
rms_db = 20 * np.log10(np.array(rms_values) + 1e-10)  # Add small value to avoid log(0)

# Calculate statistics on the RMS energy
rms_mean = np.mean(rms_values)
rms_median = np.median(rms_values)
rms_std = np.std(rms_values)
rms_percentile_10 = np.percentile(rms_values, 10)
rms_percentile_25 = np.percentile(rms_values, 25)

print("\nRMS Energy Statistics:")
print(f"Mean RMS: {rms_mean:.6f}")
print(f"Median RMS: {rms_median:.6f}")
print(f"Standard Deviation RMS: {rms_std:.6f}")
print(f"10th percentile RMS: {rms_percentile_10:.6f}")
print(f"25th percentile RMS: {rms_percentile_25:.6f}")

# Plot the waveform and RMS energy
plt.figure(figsize=(15, 10))

# Original waveform
plt.subplot(3, 1, 1)
plt.plot(np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1]), waveform[0].numpy())
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# RMS energy
plt.subplot(3, 1, 2)
plt.plot(time_values, rms_values)
plt.axhline(y=rms_median, color="r", linestyle="-", label=f"Median: {rms_median:.6f}")
plt.axhline(y=rms_percentile_25, color="g", linestyle="--", label=f"25th Percentile: {rms_percentile_25:.6f}")
plt.axhline(y=rms_percentile_10, color="y", linestyle=":", label=f"10th Percentile: {rms_percentile_10:.6f}")
plt.title("RMS Energy")
plt.xlabel("Time (s)")
plt.ylabel("RMS")
plt.legend()
plt.grid(True)

# RMS energy in dB
plt.subplot(3, 1, 3)
plt.plot(time_values, rms_db)
threshold_db = 20 * np.log10(rms_percentile_25 + 1e-10)  # Convert to dB
plt.axhline(y=threshold_db, color="g", linestyle="--", label=f"25th Percentile: {threshold_db:.2f} dB")
plt.title("RMS Energy (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (dB)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Define a function to identify silence segments
def find_silence_segments(rms_values, time_values, threshold=None, min_silence_duration=0.3):
    if threshold is None:
        threshold = rms_percentile_25  # Use 25th percentile as default

    silence_segments = []
    in_silence = False
    silence_start = 0

    for i, rms in enumerate(rms_values):
        if rms < threshold and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = time_values[i]
        elif rms >= threshold and in_silence:
            # End of silence
            silence_end = time_values[i]
            silence_duration = silence_end - silence_start
            if silence_duration >= min_silence_duration:
                silence_segments.append((silence_start, silence_end, silence_duration))
            in_silence = False

    # Check if we ended in a silence
    if in_silence and (time_values[-1] - silence_start) >= min_silence_duration:
        silence_segments.append((silence_start, time_values[-1], time_values[-1] - silence_start))

    return silence_segments


# Find silence segments using different thresholds
silence_segments_25 = find_silence_segments(rms_values, time_values, rms_percentile_25)
silence_segments_10 = find_silence_segments(rms_values, time_values, rms_percentile_10)

print("\nSilence segments using 25th percentile threshold:")
for i, (start, end, duration) in enumerate(silence_segments_25):
    print(f"Segment {i + 1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

print("\nSilence segments using 10th percentile threshold:")
for i, (start, end, duration) in enumerate(silence_segments_10):
    print(f"Segment {i + 1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

# Plot waveform with identified silence regions
plt.figure(figsize=(15, 6))
plt.plot(np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1]), waveform[0].numpy())

# Highlight silence regions with 25th percentile threshold
for start, end, _ in silence_segments_25:
    plt.axvspan(start, end, color="yellow", alpha=0.3)

# Highlight silence regions with 10th percentile threshold
for start, end, _ in silence_segments_10:
    plt.axvspan(start, end, color="red", alpha=0.2)

plt.title("Waveform with Silence Regions")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(["Waveform", "25th percentile silence", "10th percentile silence"])
plt.grid(True)
plt.show()


# Modified split function to split at silence points
def split_at_silence(waveform, sample_rate, silence_segments, output_dir, start_offset=0.0):
    os.makedirs(output_dir, exist_ok=True)

    # Add 0 as the start point and the end of audio as the end point
    split_points = [0]
    for _, end, _ in silence_segments:
        split_points.append(int(end * sample_rate))
    if split_points[-1] < waveform.shape[1]:
        split_points.append(waveform.shape[1])

    # Ensure split points are unique and sorted
    split_points = sorted(set(split_points))

    # Split the audio at these points
    for i in range(len(split_points) - 1):
        start_sample = split_points[i]
        end_sample = split_points[i + 1]

        # Extract segment
        segment = waveform[:, start_sample:end_sample]

        # Save segment
        segment_filename = os.path.join(output_dir, f"segment_{i + 1}.wav")
        torchaudio.save(segment_filename, segment, sample_rate)

        segment_duration = segment.shape[1] / sample_rate
        # Add start_offset to report correct absolute timestamps
        print(
            f"Segment {i + 1}: {(start_sample / sample_rate) + start_offset:.2f}s"
            f" - {(end_sample / sample_rate) + start_offset:.2f}s (duration: {segment_duration:.2f}s)"
        )

    return len(split_points) - 1


# Split the audio using 25th percentile silence detection
output_dir = "../data/outputs/split_audio_silence"
print("\nSplitting audio at silence points (25th percentile threshold):")
num_segments = split_at_silence(waveform, sample_rate, silence_segments_25, output_dir, start_offset=ss)
print(f"Created {num_segments} segments in {output_dir}")

# Alternative: Use 10th percentile for stricter silence detection
output_dir_strict = "../data/outputs/split_audio_silence_strict"
print("\nSplitting audio at silence points (10th percentile threshold):")
num_segments_strict = split_at_silence(waveform, sample_rate, silence_segments_10, output_dir_strict, start_offset=ss)
print(f"Created {num_segments_strict} segments in {output_dir_strict}")
