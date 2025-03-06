import time
from pathlib import Path

import polars as pl
import torchaudio
from tqdm import tqdm

from claion.pipes.sb_sts import SpeechBrainSTSPipeline


def process_directory(input_dir: Path, output_dir: Path, sts_pipe: SpeechBrainSTSPipeline, results: list):
    """Process all audio files in a directory."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("*.wav"))
    for audio_file in tqdm(audio_files, desc=f"Processing {input_dir.name}"):
        try:
            # Load audio to get the duration
            waveform, sample_rate = torchaudio.load(audio_file)
            duration = waveform.shape[1] / sample_rate

            # Measure execution time for STS pipeline
            start_time = time.time()
            corrected_audio = sts_pipe.generate_speech(audio_file)
            execution_time = time.time() - start_time

            if corrected_audio.ndim > 1:
                corrected_audio = corrected_audio.unsqueeze(0)

            output_path = output_dir / f"{audio_file.stem}.wav"
            torchaudio.save(
                str(output_path),
                corrected_audio.squeeze(0),
                sample_rate=sts_pipe.sampling_rate,
            )

            # Append results
            results.append(
                {
                    "file_name": audio_file.name,
                    "duration_sec": duration,
                    "execution_time_sec": execution_time,
                    "directory": input_dir.name,
                }
            )

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")


def main():
    """Process all audio files in train and test directories."""
    sts_pipe = SpeechBrainSTSPipeline()
    results = []

    # Define directories
    base_dir = Path("data/speechocean762")
    output_base_dir = Path("data/outputs/speechocean762")

    # Process train directory
    train_input_dir = base_dir / "train" / "audios"
    train_output_dir = output_base_dir / "train"
    process_directory(train_input_dir, train_output_dir, sts_pipe, results)

    # Process test directory
    test_input_dir = base_dir / "test" / "audios"
    test_output_dir = output_base_dir / "test"
    process_directory(test_input_dir, test_output_dir, sts_pipe, results)

    # Save results to a Parquet file
    df = pl.DataFrame(results)
    output_path = output_base_dir / "audio_processing_results.parquet"
    df.write_parquet(str(output_path))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
