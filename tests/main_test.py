import subprocess
import sys
import tempfile
import wave
from pathlib import Path

dir = Path(__file__).parent


def run_claion_command(args):
    """Runs the claion command using subprocess and captures the output."""
    result = subprocess.run(
        [sys.executable, "-m", "claion"] + args,
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


def test_claion():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = dir / "inputs" / "000080.wav"
        output_path = Path(tmpdir) / "000080.wav"
        result = run_claion_command([str(input_path), output_path])
        assert result.returncode == 0

        # Check that the output file was created
        assert output_path.exists()
        assert output_path.is_file()
