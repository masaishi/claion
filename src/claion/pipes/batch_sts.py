import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from claion.pipes.sb_sts import SpeechBrainSTSPipeline
from claion.utils.audio_spliter import (
    calculate_rms,
    calculate_silence_threshold,
    find_silence_regions,
    get_split_points,
)


class BatchSTSProcessor:
    """
    Processor for the SpeechBrain STS Pipeline with audio splitting capabilities.

    This class allows processing long audio files by:
    1. Splitting audio at silent regions
    2. Processing each segment with accent correction
    3. Recombining the processed segments
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sampling_rate: int = 16000,
        split_args: Optional[Dict] = None,
    ):
        """Initialize the processor.

        Args:
            device: Device to use for inference ("cuda" or "cpu")
            sampling_rate: Audio sampling rate
            split_args: Dictionary with audio splitting parameters
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        # Default splitting parameters
        default_split_args = {
            "min_segment_length": 1.0,
            "max_segment_length": 30.0,
            "silence_threshold_method": "percentile",
            "silence_threshold_value": 20,
            "min_silence_length": 0.2,
            "split_method": "middle",
        }

        # Use provided split_args or defaults
        self.split_args = default_split_args
        if split_args:
            self.split_args.update(split_args)

        # Initialize the STS pipeline
        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.device, sampling_rate=self.sampling_rate)

    def split_audio(self, waveform: torch.Tensor) -> List[Tuple[torch.Tensor, int, int]]:
        """Split audio waveform at silence points using configured parameters."""
        # Calculate RMS values
        rms_values, time_points = calculate_rms(waveform, self.sampling_rate)

        # Calculate silence threshold
        silence_threshold = calculate_silence_threshold(
            rms_values, method=self.split_args["silence_threshold_method"], value=self.split_args["silence_threshold_value"]
        )

        # Find silence regions
        silence_regions = find_silence_regions(
            rms_values, time_points, self.sampling_rate, silence_threshold, min_silence_length=self.split_args["min_silence_length"]
        )

        # Get split points
        split_points = get_split_points(silence_regions, waveform.shape[1], self.split_args["split_method"])

        # Split audio into segments
        segments = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            # Skip segments that are too short
            segment_duration = (end_idx - start_idx) / self.sampling_rate
            if segment_duration < self.split_args["min_segment_length"]:
                continue

            # Further split segments that are too long
            if segment_duration > self.split_args["max_segment_length"]:
                # Calculate number of subsegments needed
                num_subsegments = int(np.ceil(segment_duration / self.split_args["max_segment_length"]))
                subsegment_length = (end_idx - start_idx) // num_subsegments

                for j in range(num_subsegments):
                    sub_start = start_idx + j * subsegment_length
                    sub_end = min(sub_start + subsegment_length, end_idx)
                    segment = waveform[:, sub_start:sub_end]
                    segments.append((segment, sub_start, sub_end))
            else:
                segment = waveform[:, start_idx:end_idx]
                segments.append((segment, start_idx, end_idx))

        return segments

    def process_audio_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_segments: bool = False,
        segments_dir: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """Process a single audio file with STS pipeline."""
        input_path = Path(input_path)

        # Load audio file
        waveform, sr = torchaudio.load(str(input_path))

        # Resample if necessary
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)

        # Process audio
        processed_waveform = self.process_audio(waveform, sr, input_path.stem, save_segments, segments_dir)

        # Save output if path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), processed_waveform, self.sampling_rate)

        return processed_waveform

    def _prepare_output_directory(self, save_segments: bool, segments_dir: Optional[Union[str, Path]]) -> Optional[Path]:
        """Create segments directory if needed."""
        if save_segments and segments_dir:
            segments_dir_path = Path(segments_dir)
            segments_dir_path.mkdir(parents=True, exist_ok=True)
            return segments_dir_path
        return None

    def _extract_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from the full audio."""
        full_audio_np = waveform.squeeze(0).numpy()
        speaker_embeddings = self.sts_pipeline.extract_speechbrain_embedding(full_audio_np)
        speaker_embeddings_tensor = torch.Tensor(speaker_embeddings).unsqueeze(0).to(self.device)
        print(f"Extracted speaker embeddings with shape: {speaker_embeddings.shape}")
        return speaker_embeddings_tensor

    def _process_segment(self, segment: torch.Tensor, speaker_embeddings_tensor: torch.Tensor, segment_idx: int) -> torch.Tensor:
        """Apply STS processing to a single segment."""
        # Convert to numpy array for STS pipeline
        segment_np = segment.squeeze(0).numpy()

        try:
            # Apply STS pipeline with the pre-extracted speaker embeddings
            inputs = self.sts_pipeline.processor(audio=segment_np, sampling_rate=self.sampling_rate, return_tensors="pt").to(self.device)

            with torch.no_grad():
                processed_segment = self.sts_pipeline.generate_speech_with_embedding(
                    inputs["input_values"],
                    speaker_embeddings_tensor,
                )

            # Ensure processed_segment is 2D tensor
            if processed_segment.ndim == 1:
                processed_segment = processed_segment.unsqueeze(0)

        except Exception as e:
            warnings.warn(f"Error processing segment {segment_idx}: {str(e)}")
            # If processing fails, use the original segment
            processed_segment = segment.to(self.device)

        return processed_segment

    def _adjust_segment_length(self, processed_segment: torch.Tensor, orig_length: int) -> torch.Tensor:
        """Adjust the length of processed segment to match the original length."""
        proc_length = processed_segment.shape[1] if processed_segment.ndim > 1 else processed_segment.shape[0]

        if proc_length > orig_length:
            # Trim if longer
            if processed_segment.ndim > 1:
                return processed_segment[:, :orig_length]
            else:
                return processed_segment[:orig_length]
        elif proc_length < orig_length:
            # Pad if shorter
            if processed_segment.ndim > 1:
                padding = torch.zeros((processed_segment.shape[0], orig_length - proc_length), device=processed_segment.device)
                return torch.cat([processed_segment, padding], dim=1)
            else:
                padding = torch.zeros(orig_length - proc_length, device=processed_segment.device)
                return torch.cat([processed_segment, padding])

        return processed_segment

    def process_audio(
        self, waveform: torch.Tensor, sr: int, basename: str, save_segments: bool = False, segments_dir: Optional[Union[str, Path]] = None
    ) -> torch.Tensor:
        """Process audio waveform by splitting, applying STS, and recombining."""
        # Split audio into segments
        segments = self.split_audio(waveform)

        # Prepare for output
        output_length = waveform.shape[1]
        output_waveform = torch.zeros((1, output_length), device="cpu")

        # Create segments directory if needed
        segments_dir_path = self._prepare_output_directory(save_segments, segments_dir)

        # Extract speaker embedding from the full audio once
        speaker_embeddings_tensor = self._extract_speaker_embeddings(waveform[:, : min(sr * 3, output_length)])

        # Process each segment one by one
        for segment_idx, (segment, start_idx, end_idx) in enumerate(tqdm(segments, desc="Processing segments")):
            # Process the segment
            processed_segment = self._process_segment(segment, speaker_embeddings_tensor, segment_idx)

            # Adjust segment length to match original
            orig_length = end_idx - start_idx
            processed_segment = self._adjust_segment_length(processed_segment, orig_length)

            # Save segment if requested
            if save_segments and segments_dir_path:
                segment_path = segments_dir_path / f"{basename}_segment_{segment_idx:04d}.wav"
                save_tensor = processed_segment.cpu()
                if save_tensor.ndim == 1:
                    save_tensor = save_tensor.unsqueeze(0)
                torchaudio.save(str(segment_path), save_tensor, self.sampling_rate)

            # Add to output waveform
            proc_segment_cpu = processed_segment.cpu()
            if proc_segment_cpu.ndim == 1:
                proc_segment_cpu = proc_segment_cpu.unsqueeze(0)
            output_waveform[:, start_idx:end_idx] = proc_segment_cpu

        return output_waveform


if __name__ == "__main__":
    # Example usage with custom split_args
    split_args = {
        "min_segment_length": 1.0,
        "max_segment_length": 10.0,
        "silence_threshold_method": "percentile",
        "silence_threshold_value": 20,
        "min_silence_length": 0.3,
        "split_method": "middle",
    }

    processor = BatchSTSProcessor(split_args=split_args)

    # Process a single file
    input_file = Path("./data/inputs/original-demo-speech.wav")
    output_file = Path("./data/outputs/processed-demo-speech.wav")
    if output_file.exists():
        output_file.unlink()
    segments_dir = Path("./data/outputs/segments")
    if segments_dir.exists():
        import shutil

        shutil.rmtree(segments_dir)

    processor.process_audio_file(input_file, output_file, save_segments=True, segments_dir=segments_dir)
