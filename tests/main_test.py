import subprocess
import sys
import tempfile
import wave
from pathlib import Path

dir = Path(__file__).parent


def run_clarion_command(args):
    """Runs the clarion command using subprocess and captures the output."""
    result = subprocess.run(
        [sys.executable, "-m", "clarion_ai"] + args,
        capture_output=True,
        text=True,
    )
    return result


def get_duration(file_path):
    with wave.open(file_path, "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def test_clarion():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = dir / "data" / "000080.wav"
        output_path = Path(tmpdir) / "000080.wav"
        result = run_clarion_command([str(input_path), output_path])
        assert result.returncode == 0

        # Check that the output file was created
        assert output_path.exists()
        assert output_path.is_file()
