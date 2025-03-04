from pathlib import Path

import torchaudio
from tqdm import tqdm

from claion.pipes.sb_sts import SpeechBrainSTSPipeline


def process_directory(input_dir: Path, output_dir: Path, sts_pipe: SpeechBrainSTSPipeline):
    """Process all audio files in a directory."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("*.wav"))
    for audio_file in tqdm(audio_files, desc=f"Processing {input_dir.name}"):
        try:
            corrected_audio = sts_pipe.generate_speech(audio_file)
            if corrected_audio.ndim > 1:
                corrected_audio = corrected_audio.unsqueeze(0)

            output_path = output_dir / f"{audio_file.stem}.wav"
            torchaudio.save(
                str(output_path),
                corrected_audio.squeeze(0),
                sample_rate=sts_pipe.sampling_rate,
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")


def main():
    """Process all audio files in train and test directories."""
    sts_pipe = SpeechBrainSTSPipeline()

    # Define directories
    base_dir = Path("data/speechocean762")
    output_base_dir = Path("data/outputs/speechocean762")

    # Process train directory
    train_input_dir = base_dir / "train" / "audios"
    train_output_dir = output_base_dir / "train"
    process_directory(train_input_dir, train_output_dir, sts_pipe)

    # Process test directory
    test_input_dir = base_dir / "test" / "audios"
    test_output_dir = output_base_dir / "test"
    process_directory(test_input_dir, test_output_dir, sts_pipe)


if __name__ == "__main__":
    main()
