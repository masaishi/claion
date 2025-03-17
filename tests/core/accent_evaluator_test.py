from pathlib import Path

import pytest

from claion.core.accent_evaluator import AccentEvaluator
from claion.schemas.accent_evaluator_types import EvaluatorConfig

# Setup constants
TEST_DIR = Path(__file__).parent.parent
TEST_AUDIO = TEST_DIR / "inputs" / "000080.wav"

# Skip tests if audio file doesn't exist
pytestmark = pytest.mark.skipif(not TEST_AUDIO.exists(), reason=f"Test audio file not found: {TEST_AUDIO}")


# Setup fixture for the evaluator
@pytest.fixture
def accent_evaluator():
    """Create and return an accent evaluator instance."""
    config = EvaluatorConfig(model_path="openai/whisper-base", device="cpu")
    return AccentEvaluator(config)


def test_accent_evaluation(accent_evaluator):
    """Test the complete accent evaluation pipeline."""
    # Run the full evaluation
    results = accent_evaluator(str(TEST_AUDIO))

    # Check that the results have the expected structure
    assert len(results.accents) > 0
    assert results.transcript == ""


def test_transcript(accent_evaluator):
    "Test transcription"
    results = accent_evaluator(str(TEST_AUDIO), transcribe_audio=True)
    assert results.transcript != ""
    assert len(results.token_scores) > 0
