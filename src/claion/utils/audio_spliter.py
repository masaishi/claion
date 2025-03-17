import os

import numpy as np
import torch
import torchaudio


def calculate_rms(
    waveform: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_length: float = 0.025
) -> tuple[list[float], list[int]]:
    """
    Calculate the RMS energy of an audio waveform.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        window_size: Size of the analysis window in seconds
        hop_length: Hop length between windows in seconds

    Returns:
        tuple: (rms_values, time_points)
    """
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_length * sample_rate)

    rms_values = []
    time_points = []

    for i in range(0, waveform.shape[1] - window_samples, hop_samples):
        chunk = waveform[:, i : i + window_samples]
        rms = torch.sqrt(torch.mean(chunk**2)).item()
        rms_values.append(rms)
        time_points.append(i)

    return rms_values, time_points


def calculate_silence_threshold(rms_values: list[float], method: str = "percentile", value: float = 25) -> float:
    """
    Calculate the silence threshold based on the RMS values.

    Args:
        rms_values: List of RMS energy values
        method: Method to use ('percentile', 'mean_fraction', or 'fixed')
        value: Value to use with the method (percentile value, fraction of mean, or fixed value)

    Returns:
        float: Calculated silence threshold
    """
    if method == "percentile":
        threshold = np.percentile(rms_values, value)
    elif method == "mean_fraction":
        threshold = np.mean(rms_values) * value
    elif method == "fixed":
        threshold = value
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(threshold)


def find_silence_regions(
    rms_values: list[float],
    time_points: list[int],
    sample_rate: int,
    silence_threshold: float,
    min_silence_length: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Find silence regions in the audio.

    Args:
        rms_values: List of RMS energy values
        time_points: List of time points (in samples)
        sample_rate: Sample rate of the audio
        silence_threshold: Threshold below which audio is considered silent
        min_silence_length: Minimum length of silence (in seconds) to consider

    Returns:
        list: List of (start, end) tuples marking silence regions (in samples)
    """
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, rms in enumerate(rms_values):
        if rms < silence_threshold and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = time_points[i]
        elif (rms >= silence_threshold or i == len(rms_values) - 1) and in_silence:
            # End of silence
            silence_end = time_points[i]
            silence_duration = (silence_end - silence_start) / sample_rate

            # Only record if silence is long enough
            if silence_duration >= min_silence_length:
                silence_regions.append((silence_start, silence_end))

            in_silence = False

    return silence_regions


def get_split_points(
    silence_regions: list[tuple[int, int]], waveform_length: int, split_method: str = "middle"
) -> list[int]:
    """
    Get split points based on silence regions.

    Args:
        silence_regions: List of (start, end) tuples marking silence regions
        waveform_length: Length of the waveform in samples
        split_method: Method to determine split point ('middle', 'start', or 'end')

    Returns:
        list: List of split points in samples
    """
    split_points = [0]  # Start with beginning of audio

    for start, end in silence_regions:
        if split_method == "middle":
            point = (start + end) // 2
        elif split_method == "start":
            point = start
        elif split_method == "end":
            point = end
        else:
            raise ValueError(f"Unknown split method: {split_method}")

        split_points.append(point)

    split_points.append(waveform_length)  # End with end of audio

    return split_points


def split_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    split_points: list[int],
    output_dir: str,
    min_segment_length: float = 1.0,
) -> list[tuple[int, int]]:
    """
    Split audio at specified split points.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        split_points: List of split points in samples
        output_dir: Directory to save the split audio segments
        min_segment_length: Minimum length of segments to keep (in seconds)

    Returns:
        list: List of (start, end) tuples for the saved segments (in samples)
    """
    os.makedirs(output_dir, exist_ok=True)

    segment_info = []
    segments_saved = 0

    for i in range(len(split_points) - 1):
        segment_start = split_points[i]
        segment_end = split_points[i + 1]
        segment_duration = (segment_end - segment_start) / sample_rate

        # Skip segments that are too short
        if segment_duration < min_segment_length:
            continue

        segment = waveform[:, segment_start:segment_end]
        output_file = os.path.join(output_dir, f"segment_{segments_saved + 1}.wav")
        torchaudio.save(output_file, segment, sample_rate)

        segment_info.append((segment_start, segment_end))
        segments_saved += 1

    return segment_info
