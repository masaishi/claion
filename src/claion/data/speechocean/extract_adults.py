import polars as pl
from datasets import load_from_disk

from claion.data.utils import get_root_path


def extract_and_save_adults(min_age: int = 18):
    """
    Extract data for speakers aged min_age or over from SpeechOcean762 dataset
    and save as parquet files with audio paths.

    Args:
        min_age: Minimum age to include in the filtered dataset (default: 18)
    """
    # Load dataset
    root_path = get_root_path()
    dataset_path = root_path / "data" / "speechocean762"

    print(f"Loading dataset from {dataset_path}")
    dataset_dict = load_from_disk(dataset_path)

    for split_name, dataset in dataset_dict.items():
        print(f"Processing split: {split_name}")

        df = pl.from_pandas(dataset.to_pandas())
        df = df.with_row_count(name="index")
        df = df.drop(["audio"])
        if "age" not in df.columns:
            raise ValueError(f"Column 'age' not found in the '{split_name}' split.")

        # Filter rows where age is greater than or equal to min_age
        df_adults = df.filter(pl.col("age") >= min_age)
        print(f"  Original {split_name} set: {len(df)} samples")
        print(f"  Adults (≥{min_age}) {split_name} set: {len(df_adults)} samples")

        # Create audio paths column based on the original index of df_adults
        split_audio_dir = dataset_path / split_name / "audios"
        audio_paths = [str(split_audio_dir / f"{idx:06d}.wav") for idx in df_adults["index"].to_list()]
        df_adults = df_adults.with_columns(pl.Series(name="audio_path", values=audio_paths))

        # Define save path directly in the dataset directory
        save_path = dataset_path / f"{split_name}_adults.parquet"

        # Save to Parquet format
        df_adults.write_parquet(save_path)

        print(f"  Saved {len(df_adults)} adult records to {save_path}")

    print("✅ Extraction and saving complete.")


if __name__ == "__main__":
    extract_and_save_adults()
