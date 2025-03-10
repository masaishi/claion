from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, Field


class EvaluatorConfig(BaseModel):
    """Configuration for the AccentEvaluator."""

    model_path: str = Field(default="openai/whisper-tiny", description="Path to the Whisper model")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run inference on (cuda/cpu)",
    )


class AudioInput(BaseModel):
    """Input configuration for audio processing."""

    audio_file_path: str = Field(..., description="Path to the audio file to analyze")
    max_duration: int = Field(60, description="Maximum duration in seconds to process")


class AudioProcessingResult(BaseModel):
    """Results from audio preprocessing."""

    waveform: Any = Field(..., description="Processed audio waveform")
    sample_rate: int = Field(..., description="Sample rate of the processed audio")
    input_features: Any = Field(..., description="Features extracted for model input")

    model_config = {"arbitrary_types_allowed": True}


class TokenProbabilities(BaseModel):
    """Token probability results."""

    logits: Any = Field(..., description="Processed logits")
    probabilities: Any = Field(..., description="Token probabilities")

    model_config = {"arbitrary_types_allowed": True}


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech with timestamps."""

    start_t: str = Field(..., description="Start timestamp of the segment")
    end_t: Optional[str] = Field(None, description="End timestamp of the segment")
    text: str = Field(..., description="Transcribed text content")


class TranscriptionResult(BaseModel):
    """Results from transcribing audio."""

    transcript: str = Field(..., description="Transcribed text content without special tokens")
    token_scores: Dict[str, float] = Field(..., description="Confidence scores for each token")
    mean_score: float = Field(..., description="Average confidence score across all words")


class AccentEvaluationResult(BaseModel):
    """Complete results from evaluating accent in audio."""

    accents: Dict[str, float] = Field(..., description="Detected language accents with confidence scores")
    transcript: str = Field(..., description="Transcribed text content without special tokens")
    token_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for each token")
    mean_score: float = Field(0.0, description="Average confidence score across all words")
