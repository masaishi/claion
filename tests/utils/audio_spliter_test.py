import tempfile
from pathlib import Path

import pytest
import torchaudio

from claion.utils.audio_spliter import calculate_rms, find_loudest_point, find_quietest_section, save_audio_segment

# Setup constants
TEST_DIR = Path(__file__).parent.parent
TEST_AUDIO = TEST_DIR / "inputs" / "000080.wav"

# Skip tests if audio file doesn't exist
pytestmark = pytest.mark.skipif(not TEST_AUDIO.exists(), reason=f"Test audio file not found: {TEST_AUDIO}")


def test_calculate_rms():
    """Test the RMS calculation function."""
    # Load the test audio
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)

    # Test with default parameters
    rms_values, time_points = calculate_rms(waveform, sample_rate)

    # Check that the results have the expected structure
    assert isinstance(rms_values, list)
    assert isinstance(time_points, list)
    assert len(rms_values) > 0
    assert len(rms_values) == len(time_points)
    assert all(isinstance(val, float) for val in rms_values)
    assert all(isinstance(val, int) for val in time_points)

    # Test with custom window and hop sizes
    rms_values_custom, time_points_custom = calculate_rms(waveform, sample_rate, window_size=0.1, hop_length=0.05)

    # With smaller hop size, we should have more values
    assert len(rms_values_custom) > len(rms_values)


def test_find_quietest_section():
    """Test finding the quietest section in audio."""
    # Load the test audio
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)

    # First calculate RMS values
    rms_values, time_points = calculate_rms(waveform, sample_rate)

    # Find quietest section
    start_sample, end_sample = find_quietest_section(rms_values, time_points, sample_rate, min_time=0.0, max_time=float(waveform.shape[1]) / sample_rate)

    # Validate results
    assert isinstance(start_sample, int)
    assert isinstance(end_sample, int)
    assert start_sample >= 0
    assert end_sample <= waveform.shape[1]
    assert start_sample < end_sample


def test_find_loudest_point():
    """Test finding the loudest point in a section of audio."""
    # Load the test audio
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)

    # Use a section of the audio
    start_sample = 0
    end_sample = waveform.shape[1]

    # Find loudest point
    loudest_point = find_loudest_point(waveform, start_sample, end_sample, sample_rate)

    # Validate results
    assert isinstance(loudest_point, int)
    assert start_sample <= loudest_point <= end_sample


def test_save_audio_segment():
    """Test saving an audio segment."""
    # Load the test audio
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save the audio
        success = save_audio_segment(waveform, sample_rate, temp_path)

        # Check that saving was successful
        assert success
        assert Path(temp_path).exists()

        # Verify the saved file can be loaded
        saved_waveform, saved_sample_rate = torchaudio.load(temp_path)
        assert saved_sample_rate == sample_rate
        assert saved_waveform.shape[0] == waveform.shape[0]  # Same number of channels

    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_audio_processing_pipeline():
    """Test the complete audio processing pipeline."""
    # Load the test audio
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # 1. Calculate RMS
        rms_values, time_points = calculate_rms(waveform, sample_rate)

        # 2. Find quietest section
        start_sample, end_sample = find_quietest_section(rms_values, time_points, sample_rate)

        # 3. Find loudest point within the quietest section
        loudest_point = find_loudest_point(waveform, start_sample, end_sample, sample_rate)

        # 4. Extract a segment around the loudest point
        segment_start = max(0, loudest_point - int(0.5 * sample_rate))
        segment_end = min(waveform.shape[1], loudest_point + int(0.5 * sample_rate))
        segment = waveform[:, segment_start:segment_end]

        # 5. Save the segment
        success = save_audio_segment(segment, sample_rate, temp_path)

        # Validate the pipeline
        assert success
        assert Path(temp_path).exists()

    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()
