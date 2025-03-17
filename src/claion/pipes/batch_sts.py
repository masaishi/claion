import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    Batch processor for the SpeechBrain STS Pipeline with audio splitting capabilities.

    This class allows processing long audio files by:
    1. Splitting audio at silent regions
    2. Processing each segment with accent correction
    3. Recombining the processed segments
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sampling_rate: int = 16000,
        min_segment_length: float = 1.0,
        max_segment_length: float = 30.0,
        silence_threshold_method: str = "percentile",
        silence_threshold_value: float = 20,
        min_silence_length: float = 0.2,
        split_method: str = "middle",
        batch_size: int = 1,
    ):
        """Initialize the batch processor."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.silence_threshold_method = silence_threshold_method
        self.silence_threshold_value = silence_threshold_value
        self.min_silence_length = min_silence_length
        self.split_method = split_method
        self.batch_size = batch_size

        # Initialize the STS pipeline
        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.device, sampling_rate=self.sampling_rate)

    def split_audio(self, waveform: torch.Tensor) -> List[Tuple[torch.Tensor, int, int]]:
        """Split audio waveform at silence points."""
        # Calculate RMS values
        rms_values, time_points = calculate_rms(waveform, self.sampling_rate)

        # Calculate silence threshold
        silence_threshold = calculate_silence_threshold(rms_values, method=self.silence_threshold_method, value=self.silence_threshold_value)

        # Find silence regions
        silence_regions = find_silence_regions(rms_values, time_points, self.sampling_rate, silence_threshold, min_silence_length=self.min_silence_length)

        # Get split points
        split_points = get_split_points(silence_regions, waveform.shape[1], self.split_method)

        # Split audio into segments
        segments = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            # Skip segments that are too short
            segment_duration = (end_idx - start_idx) / self.sampling_rate
            if segment_duration < self.min_segment_length:
                continue

            # Further split segments that are too long
            if segment_duration > self.max_segment_length:
                # Calculate number of subsegments needed
                num_subsegments = int(np.ceil(segment_duration / self.max_segment_length))
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
        processed_waveform = self.process_audio(waveform, input_path.stem, save_segments, segments_dir)

        # Save output if path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), processed_waveform, self.sampling_rate)

        return processed_waveform

    def process_audio(
        self, waveform: torch.Tensor, basename: str, save_segments: bool = False, segments_dir: Optional[Union[str, Path]] = None
    ) -> torch.Tensor:
        """Process audio waveform by splitting, applying STS, and recombining."""
        # Split audio into segments
        segments = self.split_audio(waveform)

        # Prepare for output
        output_length = waveform.shape[1]
        output_waveform = torch.zeros((1, output_length), device="cpu")

        # Create segments directory if needed
        if save_segments and segments_dir:
            segments_dir = Path(segments_dir)
            segments_dir.mkdir(parents=True, exist_ok=True)

        # Extract speaker embedding from the full audio once
        full_audio_np = waveform.squeeze(0).numpy()
        speaker_embeddings = self.sts_pipeline.extract_speechbrain_embedding(full_audio_np)
        speaker_embeddings_tensor = torch.Tensor(speaker_embeddings).unsqueeze(0).to(self.device)

        print(f"Extracted speaker embeddings with shape: {speaker_embeddings.shape}")

        # Process segments in batches
        for batch_idx in range(0, len(segments), self.batch_size):
            batch_segments = segments[batch_idx : batch_idx + self.batch_size]

            # Process each segment in the batch
            for i, (segment, start_idx, end_idx) in enumerate(tqdm(batch_segments, desc=f"Processing batch {batch_idx // self.batch_size + 1}")):
                segment_idx = batch_idx + i

                # Convert to numpy array for STS pipeline
                segment_np = segment.squeeze(0).numpy()

                try:
                    # Apply STS pipeline with the pre-extracted speaker embeddings
                    inputs = self.sts_pipeline.processor(audio=segment_np, sampling_rate=self.sampling_rate, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        processed_segment = self.sts_pipeline.generate_speech_with_embedding(inputs, speaker_embeddings_tensor)

                    if processed_segment.ndim == 1:
                        processed_segment = processed_segment.unsqueeze(0)

                except Exception as e:
                    warnings.warn(f"Error processing segment {segment_idx}: {str(e)}")
                    # If processing fails, use the original segment
                    processed_segment = segment.to(self.device)

                # Ensure processed segment is the same length as the original
                orig_length = end_idx - start_idx
                proc_length = processed_segment.shape[1] if processed_segment.ndim > 1 else processed_segment.shape[0]

                if proc_length > orig_length:
                    # Trim if longer
                    if processed_segment.ndim > 1:
                        processed_segment = processed_segment[:, :orig_length]
                    else:
                        processed_segment = processed_segment[:orig_length]
                elif proc_length < orig_length:
                    # Pad if shorter
                    if processed_segment.ndim > 1:
                        padding = torch.zeros((processed_segment.shape[0], orig_length - proc_length), device=processed_segment.device)
                        processed_segment = torch.cat([processed_segment, padding], dim=1)
                    else:
                        padding = torch.zeros(orig_length - proc_length, device=processed_segment.device)
                        processed_segment = torch.cat([processed_segment, padding])

                # Save segment if requested
                if save_segments and segments_dir:
                    segment_path = segments_dir / f"{basename}_segment_{segment_idx:04d}.wav"

                    # Ensure we have a 2D tensor for saving
                    save_tensor = processed_segment.cpu()
                    if save_tensor.ndim == 1:
                        save_tensor = save_tensor.unsqueeze(0)

                    torchaudio.save(str(segment_path), save_tensor, self.sampling_rate)

                # Add to output waveform
                # Make sure the segment is 2D before adding to output
                proc_segment_cpu = processed_segment.cpu()
                if proc_segment_cpu.ndim == 1:
                    proc_segment_cpu = proc_segment_cpu.unsqueeze(0)

                output_waveform[:, start_idx:end_idx] = proc_segment_cpu

        return output_waveform

    def batch_process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_extension: str = "wav",
        save_segments: bool = False,
        segments_parent_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all audio files
        audio_files = list(input_dir.glob(f"*.{file_extension}"))

        # Process each file
        for audio_file in tqdm(audio_files, desc="Processing files"):
            output_path = output_dir / f"{audio_file.stem}_processed.{file_extension}"

            segments_dir = None
            if save_segments and segments_parent_dir:
                segments_dir = Path(segments_parent_dir) / audio_file.stem

            self.process_audio_file(audio_file, output_path, save_segments, segments_dir)

    def process_numpy_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        basename: str = "array",
        save_output: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        save_segments: bool = False,
        segments_dir: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Process a numpy array directly."""
        # Convert numpy array to tensor
        if audio_array.ndim == 1:
            # Convert mono to [channels, samples] format
            waveform = torch.tensor(audio_array).unsqueeze(0)
        else:
            # Already in [channels, samples] format
            waveform = torch.tensor(audio_array)

        # Resample if necessary
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)

        # Process audio
        processed_waveform = self.process_audio(waveform, basename, save_segments, segments_dir)

        # Save if requested
        if save_output and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), processed_waveform, self.sampling_rate)

        # Convert back to numpy array
        return processed_waveform.squeeze(0).numpy()


if __name__ == "__main__":
    # Example usage
    processor = BatchSTSProcessor(
        min_segment_length=1.0,
        max_segment_length=10.0,
        silence_threshold_method="percentile",
        silence_threshold_value=20,
        min_silence_length=0.3,
        split_method="middle",
        batch_size=4,
    )

    # Process a single file
    input_file = Path("./data/inputs/original-demo-speech.wav")
    output_file = Path("./data/outputs/processed-demo-speech.wav")
    processor.process_audio_file(input_file, output_file, save_segments=True, segments_dir="data/outputs/segments")
