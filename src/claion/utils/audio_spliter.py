import os

import numpy as np
import torch
import torchaudio


def calculate_rms(waveform: torch.Tensor, sample_rate: int, window_size: float = 0.2, hop_length: float = 0.1) -> tuple[list[float], list[int]]:
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


def calculate_silence_threshold(rms_values: list[float], method: str = "percentile", value: float = 20) -> float:
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
    min_silence_length: float = 0.1,
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


def get_split_points(silence_regions: list[tuple[int, int]], waveform_length: int, split_method: str = "middle") -> list[int]:
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


def save_audio_segment(waveform: torch.Tensor, sample_rate: int, output_file: str) -> bool:
    """Save audio segment using torchaudio."""
    try:
        # Ensure waveform is in the correct format for torchaudio
        if waveform.dim() == 2:  # [channels, samples]
            # Normalize if needed
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                max_val = max(waveform.max().abs().item(), waveform.min().abs().item())
                waveform = waveform / max_val

            # Make a copy to avoid any reference issues
            waveform_copy = waveform.clone().detach()

            # Use torchaudio to save
            torchaudio.save(output_file, waveform_copy, sample_rate, format="wav")
            return True

    except Exception as e:
        import warnings

        warnings.warn(f"Error saving audio file: {str(e)}")
        return False


def split_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    split_points: list[int],
    output_dir: str,
    min_segment_length: float = 0.5,
) -> list[tuple[int, int]]:
    """Split audio at given points and save segments using torchaudio."""
    min_samples = int(min_segment_length * sample_rate)
    segments = []

    os.makedirs(output_dir, exist_ok=True)

    # Create segments from split points
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        # Keep only segments longer than minimum length
        if end - start >= min_samples:
            segments.append((start, end))

    # Save valid segments
    for i, (start, end) in enumerate(segments):
        # Create a deep copy of the segment to avoid reference issues
        segment_waveform = waveform[:, start:end].clone().detach()
        output_file = os.path.join(output_dir, f"segment_{i + 1:06d}.wav")

        try:
            # Direct torchaudio save approach
            torchaudio.save(output_file, segment_waveform, sample_rate)
            print(f"Saved segment {i + 1} to {output_file} - Duration: {(end - start) / sample_rate:.2f}s")
        except Exception as e:
            print(f"Error saving segment {i + 1}: {str(e)}")
            # Fallback to our helper function
            success = save_audio_segment(segment_waveform, sample_rate, output_file)
            if success:
                print(f"Saved segment {i + 1} to {output_file} using fallback method - Duration: {(end - start) / sample_rate:.2f}s")
            else:
                print(f"Failed to save segment {i + 1}")

    return segments
