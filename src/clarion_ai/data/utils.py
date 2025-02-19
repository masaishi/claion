from pathlib import Path

import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm


def get_root_path():
    return Path(__file__).resolve().parents[3]


def extract_audio(dataset_path: Path) -> None:
    dataset = load_from_disk(dataset_path)

    for split in dataset.keys():
        split_path = dataset_path / split
        audio_dir = split_path / "audios"
        audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {split} set...")
        for i, sample in tqdm(enumerate(dataset[split])):
            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            audio_path = audio_dir / f"{i:06d}.wav"
            sf.write(audio_path, audio_array, sample_rate)
        print(f"Finished processing {split}. All files saved in: {audio_dir}")
