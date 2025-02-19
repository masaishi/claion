from datasets import load_dataset

from clarion_ai.data.utils import extract_audio, get_root_path


def download_speechocean762():
    save_path = get_root_path() / "data" / "speechocean762"
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mispeech/speechocean762")
    dataset.save_to_disk(save_path)
    extract_audio(save_path)


if __name__ == "__main__":
    download_speechocean762()
