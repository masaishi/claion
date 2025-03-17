import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from claion.utils.audio_spliter import (
    calculate_rms,
    calculate_silence_threshold,
    find_silence_regions,
    get_split_points,
    save_audio_segment,
    split_audio,
)

# Setup constants
TEST_DIR = Path(__file__).parent
TEST_AUDIO_PATH = TEST_DIR / "test_audio"
TEST_AUDIO_PATH.mkdir(exist_ok=True)


@pytest.fixture
def sample_waveform():
    """Create a simple test waveform with silence and sound regions."""
    sample_rate = 16000
    duration = 3  # seconds

    # Generate a simple sine wave for sound portions
    t = torch.linspace(0, duration, int(sample_rate * duration))
    sound = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Create regions of silence (zeros)
    waveform = sound.clone()
    # Add silence in the middle (1.0s to 1.5s)
    silence_start = int(1.0 * sample_rate)
    silence_end = int(1.5 * sample_rate)
    waveform[silence_start:silence_end] = 0.0

    # Add silence at the end (2.5s to 3.0s)
    silence_start = int(2.5 * sample_rate)
    silence_end = int(3.0 * sample_rate)
    waveform[silence_start:silence_end] = 0.0

    # Make it a proper audio tensor [channels, samples]
    waveform = waveform.unsqueeze(0)

    # Save the test waveform for testing
    test_file = TEST_AUDIO_PATH / "test_audio.wav"
    torchaudio.save(test_file, waveform, sample_rate)

    return waveform, sample_rate, test_file


def test_calculate_rms(sample_waveform):
    """Test RMS calculation with known input patterns."""
    waveform, sample_rate, _ = sample_waveform

    # Calculate RMS with window size that should capture the silent regions
    rms_values, time_points = calculate_rms(waveform, sample_rate, window_size=0.2, hop_length=0.1)

    # Basic checks
    assert len(rms_values) == len(time_points)
    assert len(rms_values) > 0

    # Find index for middle silence region
    middle_idx = [i for i, t in enumerate(time_points) if t >= int(1.0 * sample_rate)][0]

    # Find index for end silence region
    end_idx = [i for i, t in enumerate(time_points) if t >= int(2.5 * sample_rate)][0]

    # Verify the silent regions have very low RMS values
    assert rms_values[middle_idx] < 0.01, "Middle silence region should have near-zero RMS"
    assert rms_values[end_idx] < 0.01, "End silence region should have near-zero RMS"

    # Verify sound regions have higher RMS values
    sound_idx = 5  # An index known to be in the sound region
    assert rms_values[sound_idx] > 0.1, "Sound region should have higher RMS"


def test_calculate_silence_threshold():
    """Test the different methods of silence threshold calculation."""
    # Create a known set of RMS values
    rms_values = [0.01, 0.02, 0.5, 0.6, 0.7, 0.8]

    # Test percentile method
    percentile_threshold = calculate_silence_threshold(rms_values, method="percentile", value=20)
    assert percentile_threshold == pytest.approx(0.02), "20th percentile should be 0.02"

    # Test mean_fraction method
    mean = np.mean(rms_values)  # 0.438333...
    expected_threshold = mean * 0.1
    threshold = calculate_silence_threshold(rms_values, method="mean_fraction", value=0.1)
    assert threshold == pytest.approx(expected_threshold), f"Mean fraction incorrect: {threshold} vs {expected_threshold}"

    # Test fixed method
    fixed_threshold = calculate_silence_threshold(rms_values, method="fixed", value=0.25)
    assert fixed_threshold == 0.25, "Fixed threshold should be exactly the value provided"

    # Test invalid method
    with pytest.raises(ValueError):
        calculate_silence_threshold(rms_values, method="invalid_method")


def test_find_silence_regions(sample_waveform):
    """Test finding silence regions in the audio."""
    waveform, sample_rate, _ = sample_waveform

    # Calculate RMS
    rms_values, time_points = calculate_rms(waveform, sample_rate)

    # Set a threshold that should capture our two silence regions
    threshold = 0.05
    silence_regions = find_silence_regions(rms_values, time_points, sample_rate, threshold, min_silence_length=0.2)

    # We should find at least our two inserted silence regions
    assert len(silence_regions) >= 2, "Should find at least two silence regions"

    # Convert to seconds for easier checking
    silence_regions_seconds = [(start / sample_rate, end / sample_rate) for start, end in silence_regions]

    # Check if our known silence regions are detected (allowing some margin)
    middle_silence_found = any(abs(start - 1.0) < 0.3 and abs(end - 1.5) < 0.3 for start, end in silence_regions_seconds)
    end_silence_found = any(abs(start - 2.5) < 0.3 and abs(end - 3.0) < 0.3 for start, end in silence_regions_seconds)

    assert middle_silence_found, "Middle silence region not found"
    assert end_silence_found, "End silence region not found"

    # Test with higher min_silence_length
    long_regions = find_silence_regions(rms_values, time_points, sample_rate, threshold, min_silence_length=1.0)
    assert len(long_regions) == 0, "No regions should be found with min_silence_length=1.0s"


def test_get_split_points():
    """Test getting split points from silence regions."""
    # Create sample silence regions
    silence_regions = [(1000, 2000), (5000, 6000), (9000, 10000)]
    waveform_length = 12000

    # Test middle method
    middle_points = get_split_points(silence_regions, waveform_length, split_method="middle")
    expected_middle = [0, 1500, 5500, 9500, 12000]
    assert middle_points == expected_middle, f"Middle points incorrect: {middle_points}"

    # Test start method
    start_points = get_split_points(silence_regions, waveform_length, split_method="start")
    expected_start = [0, 1000, 5000, 9000, 12000]
    assert start_points == expected_start, f"Start points incorrect: {start_points}"

    # Test end method
    end_points = get_split_points(silence_regions, waveform_length, split_method="end")
    expected_end = [0, 2000, 6000, 10000, 12000]
    assert end_points == expected_end, f"End points incorrect: {end_points}"

    # Test invalid method
    with pytest.raises(ValueError):
        get_split_points(silence_regions, waveform_length, split_method="invalid")


def test_save_audio_segment(sample_waveform):
    """Test saving audio segments."""
    waveform, sample_rate, _ = sample_waveform

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_segment.wav")

        # Test successful save
        success = save_audio_segment(waveform, sample_rate, output_file)
        assert success, "Should successfully save the audio segment"
        assert os.path.exists(output_file), "Output file should exist"

        # Test with invalid waveform dimension
        invalid_waveform = torch.rand(3)  # 1D tensor
        success = save_audio_segment(invalid_waveform, sample_rate, output_file)
        assert not success, "Should return False for invalid waveform dimension"


def test_split_audio(sample_waveform):
    """Test splitting audio at given points."""
    waveform, sample_rate, _ = sample_waveform

    # Define split points
    split_points = [0, 5000, 10000, waveform.shape[1]]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Split the audio
        segments = split_audio(waveform, sample_rate, split_points, temp_dir)

        # Check the number of segments
        assert len(segments) == len(split_points) - 1, "Should have n-1 segments for n split points"

        # Check if the segment files were created
        for i in range(len(segments)):
            output_file = os.path.join(temp_dir, f"segment_{i + 1:06d}.wav")
            assert os.path.exists(output_file), f"Segment file {output_file} should exist"

            # Optionally, load the segment and check its length
            segment_waveform, segment_sr = torchaudio.load(output_file)
            expected_length = segments[i][1] - segments[i][0]
            assert segment_waveform.shape[1] == expected_length, "Segment length should match the expected value"


def test_end_to_end_pipeline(sample_waveform):
    """Test the entire audio processing pipeline from RMS to splitting."""
    waveform, sample_rate, audio_file = sample_waveform

    # Step 1: Calculate RMS
    rms_values, time_points = calculate_rms(waveform, sample_rate)

    # Step 2: Calculate silence threshold
    threshold = calculate_silence_threshold(rms_values, method="percentile", value=20)

    # Step 3: Find silence regions
    silence_regions = find_silence_regions(rms_values, time_points, sample_rate, threshold, min_silence_length=0.2)

    # Step 4: Get split points
    split_points = get_split_points(silence_regions, waveform.shape[1])

    # Step 5: Split audio
    with tempfile.TemporaryDirectory() as temp_dir:
        segments = split_audio(waveform, sample_rate, split_points, temp_dir)

        # Check that segments were created
        assert len(segments) > 0, "Should have created at least one segment"

        # Check if segment files exist
        for i in range(len(segments)):
            output_file = os.path.join(temp_dir, f"segment_{i + 1:06d}.wav")
            assert os.path.exists(output_file), f"Segment file {i + 1} should exist"


def test_handle_edge_cases():
    """Test handling of edge cases."""
    # Empty waveform
    empty_waveform = torch.zeros(1, 0)
    empty_sr = 16000

    # Calculate RMS for empty waveform
    rms_values, time_points = calculate_rms(empty_waveform, empty_sr)
    assert len(rms_values) == 0, "Empty waveform should produce empty RMS values"

    # Single sample waveform
    single_sample = torch.ones(1, 1)
    with pytest.raises(Exception):
        # This should fail since window size would be larger than waveform
        calculate_rms(single_sample, empty_sr, window_size=0.2)

    # Test silence regions with empty RMS values
    empty_regions = find_silence_regions([], [], empty_sr, 0.1)
    assert len(empty_regions) == 0, "Empty RMS should produce no silence regions"

    # Test split points with empty silence regions
    empty_split_points = get_split_points([], 1000)
    assert empty_split_points == [0, 1000], "Should return just start and end points"

    # Test split audio with minimum segment length filtering
    short_waveform = torch.ones(1, 1000)  # 1000 samples
    short_sr = 16000  # 16kHz
    split_points = [0, 100, 500, 1000]  # Points that would create segments smaller than min_segment_length

    with tempfile.TemporaryDirectory() as temp_dir:
        # With min_segment_length of 0.1s (1600 samples at 16kHz), all segments should be filtered out
        segments = split_audio(short_waveform, short_sr, split_points, temp_dir, min_segment_length=0.1)
        assert len(segments) == 0, "All segments should be filtered out due to min_segment_length"
