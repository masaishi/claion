import polars as pl
from datasets import load_from_disk

from clarion_ai.data.utils import get_root_path


def extract_and_save_adults():
    # Load dataset
    root_path = get_root_path()
    dataset_dict = load_from_disk(root_path / "data" / "speechocean762")

    for split_name, dataset in dataset_dict.items():
        print(f"Processing split: {split_name}")

        df = pl.from_pandas(dataset.to_pandas())

        # Ensure 'age' column exists
        if "age" not in df.columns:
            raise ValueError(
                f"Column 'age' not found in the '{split_name}' split."
            )

        # Filter rows where age is greater than 18
        df_adults = df.filter(df["age"] > 18)

        # Define save path (inside each split folder)
        save_path = (
            root_path
            / "data"
            / "speechocean762"
            / split_name
            / "adults.parquet"
        )

        # Save to Parquet format (efficient for structured data)
        df_adults.write_parquet(save_path)

        print(f"Saved {len(df_adults)} adult rows to {save_path}")

    print("✅ Extraction and saving complete.")


if __name__ == "__main__":
    extract_and_save_adults()
