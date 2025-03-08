import polars as pl
from datasets import load_from_disk

from claion.data.utils import get_root_path


def eda():
    root_path = get_root_path()

    # Load dataset from disk
    dataset_dict = load_from_disk(root_path / "data" / "speechocean762")

    # Show available dataset splits
    print("\nAvailable dataset splits:", dataset_dict.keys())

    # Choose a split (e.g., 'train')
    split_name = "train"  # Adjust as needed
    if split_name not in dataset_dict:
        raise ValueError(f"Split '{split_name}' not found in dataset. Available: {dataset_dict.keys()}")

    dataset = dataset_dict[split_name]

    # Convert to Polars DataFrame
    df = pl.from_pandas(dataset.to_pandas())
    df = df.drop(["audio", "words"])

    # Force display all columns and full content
    pl.Config.set_tbl_cols(len(df.columns))  # Ensure all columns are shown
    pl.Config.set_tbl_rows(20)  # Increase displayed rows
    pl.Config.set_tbl_width_chars(300)  # Expand width to prevent truncation
    pl.Config.set_fmt_str_lengths(100)  # Expand string length display
    pl.Config.set_tbl_hide_dataframe_shape(False)  # Ensure full dataframe shape is displayed

    # Print dataset overview
    print("\nDataset Structure:\n", dataset)

    print("\nSchema & Data Types:")
    print(df.schema)

    print("\nFirst 20 rows (Ensuring Full View):")
    print(df.head(20))  # Show more rows explicitly

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values per Column:")
    print(df.null_count())

    print("\nUnique Values per Column:")
    unique_counts = {col: df[col].n_unique() for col in df.columns}
    print(unique_counts)

    # Class Distribution (if applicable)
    target_column = "label"  # Adjust if necessary
    if target_column in df.columns:
        print("\nClass Distribution:")
        print(df[target_column].value_counts())

    # Display a random sample row with full content
    print("\nSample Data Row:")
    print(df.sample(1).write_csv())

    return df


if __name__ == "__main__":
    df = eda()
