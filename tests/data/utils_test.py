from clarion_ai.data.utils import get_root_path


def test_get_root_path():
    root_path = get_root_path()
    assert root_path.is_dir()
    assert root_path.name == "clarion-ai"
    assert (root_path / "src").is_dir()
    assert (root_path / "pyproject.toml").is_file()
