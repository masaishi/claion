from typing import List, Tuple

import torch
import torchaudio


def calculate_rms(waveform: torch.Tensor, sample_rate: int, window_size: float = 0.2, hop_length: float = 0.1) -> Tuple[List[float], List[int]]:
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


def find_quietest_section(
    rms_values: List[float], time_points: List[int], sample_rate: int, min_time: float = 5.0, max_time: float = 15.0, section_length: float = 3.0
) -> Tuple[int, int]:
    """
    Find the quietest section between min_time and max_time.

    Args:
        rms_values: List of RMS energy values
        time_points: List of time points (in samples)
        sample_rate: Sample rate of the audio
        min_time: Minimum time to consider (in seconds)
        max_time: Maximum time to consider (in seconds)
        section_length: Length of section to find (in seconds)

    Returns:
        tuple: (start_sample, end_sample) of the quietest section
    """
    # Convert time bounds to indices
    min_sample = int(min_time * sample_rate)
    max_sample = int(max_time * sample_rate)
    section_samples = int(section_length * sample_rate)

    # Find corresponding indices in time_points
    min_idx = 0
    max_idx = len(time_points) - 1

    for i, tp in enumerate(time_points):
        if tp >= min_sample:
            min_idx = i
            break

    for i, tp in enumerate(time_points[min_idx:], start=min_idx):
        if tp >= max_sample:
            max_idx = i
            break

    # Find the section with the lowest average RMS
    lowest_avg_rms = float("inf")
    quietest_start_idx = min_idx

    # Calculate window size based on hop length between time points
    hop_length = time_points[1] - time_points[0] if len(time_points) > 1 else sample_rate * 0.1
    section_window = max(1, int(section_samples / hop_length))

    for i in range(min_idx, max_idx - section_window + 1):
        section_rms = rms_values[i : i + section_window]
        avg_rms = sum(section_rms) / len(section_rms)

        if avg_rms < lowest_avg_rms:
            lowest_avg_rms = avg_rms
            quietest_start_idx = i

    start_sample = time_points[quietest_start_idx]
    end_sample = time_points[quietest_start_idx + section_window - 1] if quietest_start_idx + section_window - 1 < len(time_points) else time_points[-1]

    return start_sample, end_sample


def find_loudest_point(waveform: torch.Tensor, start_sample: int, end_sample: int, sample_rate: int, window_size: float = 0.05) -> int:
    """
    Find the loudest point within a section of audio.

    Args:
        waveform: Audio waveform tensor
        start_sample: Start sample of the section
        end_sample: End sample of the section
        sample_rate: Sample rate of the audio
        window_size: Size of the analysis window in seconds

    Returns:
        int: Sample index of the loudest point
    """
    window_samples = int(window_size * sample_rate)
    max_rms = 0
    loudest_point = start_sample

    for i in range(start_sample, end_sample - window_samples, window_samples // 2):
        chunk = waveform[:, i : i + window_samples]
        rms = torch.sqrt(torch.mean(chunk**2)).item()

        if rms > max_rms:
            max_rms = rms
            loudest_point = i + window_samples // 2  # Center of the window

    return loudest_point


def save_audio_segment(waveform: torch.Tensor, sample_rate: int, output_file: str) -> bool:
    """Save audio segment using torchaudio.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        output_file: Path to save the audio file

    Returns:
        bool: True if saving was successful, False otherwise
    """
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
        return False  # Return False if waveform dimension is not 2

    except Exception as e:
        import warnings

        warnings.warn(f"Error saving audio file: {str(e)}")
        return False
