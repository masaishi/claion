import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from tqdm import tqdm

from claion.pipes.sb_sts import SpeechBrainSTSPipeline
from claion.utils.audio_spliter import calculate_rms, find_loudest_point, find_quietest_section


class BatchSTSProcessor:
    """
    Processor for the SpeechBrain STS Pipeline with audio splitting capabilities.

    This class allows processing long audio files by:
    1. Finding the quietest section between specified time ranges
    2. Splitting audio at the loudest point within that quietest section
    3. Processing each segment with accent correction
    4. Recombining the processed segments
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
            "quiet_section_min_time": 5.0,
            "quiet_section_max_time": 15.0,
            "quiet_section_length": 3.0,
            "loudness_window_size": 0.05,
        }

        # Use provided split_args or defaults
        self.split_args = default_split_args
        if split_args:
            self.split_args.update(split_args)

        # Initialize the STS pipeline
        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.device, sampling_rate=self.sampling_rate)

    def split_audio(self, waveform: torch.Tensor) -> List[Tuple[torch.Tensor, int, int]]:
        """
        Find the quietest section and split it at its loudest point.

        Args:
            waveform: Audio waveform tensor

        Returns:
            List of tuples containing (segment_waveform, start_idx, end_idx)
        """
        # Calculate RMS values
        rms_values, time_points = calculate_rms(waveform, self.sampling_rate)

        # Find the quietest section
        quiet_start, quiet_end = find_quietest_section(
            rms_values,
            time_points,
            self.sampling_rate,
            min_time=self.split_args["quiet_section_min_time"],
            max_time=self.split_args["quiet_section_max_time"],
            section_length=self.split_args["quiet_section_length"],
        )

        print(f"Quietest section: {quiet_start / self.sampling_rate:.2f}s to {quiet_end / self.sampling_rate:.2f}s")

        # Find the loudest point within the quietest section
        loudest_point = find_loudest_point(waveform, quiet_start, quiet_end, self.sampling_rate, window_size=self.split_args["loudness_window_size"])

        print(f"Loudest point within quiet section: {loudest_point / self.sampling_rate:.2f}s")

        # Split the audio at the loudest point of the quietest section
        segments = []

        # First segment: start to loudest point
        first_segment = waveform[:, :loudest_point].clone().detach()
        segments.append((first_segment, 0, loudest_point))

        # Second segment: loudest point to end
        second_segment = waveform[:, loudest_point:].clone().detach()
        segments.append((second_segment, loudest_point, waveform.shape[1]))

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

    def _extract_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from the audio."""
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
            segments_dir_path = Path(segments_dir)
            segments_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            segments_dir_path = None

        # Extract speaker embedding from the first few seconds of audio
        speaker_embeddings_tensor = self._extract_speaker_embeddings(waveform[:, : min(self.sampling_rate * 3, output_length)])

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
    # Example usage
    processor = BatchSTSProcessor()

    # Process a single file
    input_file = Path("./data/inputs/original-demo-speech.wav")
    output_file = Path("./data/outputs/processed-demo-speech.wav")

    # Clean up existing files if needed
    if output_file.exists():
        output_file.unlink()

    segments_dir = Path("./data/outputs/segments")
    if segments_dir.exists():
        import shutil

        shutil.rmtree(segments_dir)

    processor.process_audio_file(input_file, output_file, save_segments=True, segments_dir=segments_dir)
