from datasets import load_dataset

from clarion_ai.data.utils import get_root_path


def download_speechocean762():
    save_path = get_root_path() / "data" / "speechocean762"
    save_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Speechocean762 dataset...")
    dataset = load_dataset("mispeech/speechocean762")
    dataset.save_to_disk(save_path)
    print(f"Speechocean762 dataset saved in {save_path}")


if __name__ == "__main__":
    download_speechocean762()
