"""
Claion CLI - Real-time accent correction tool
"""

import sys
from pathlib import Path

import click
import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install

from claion.pipes.sb_sts import SpeechBrainSTSPipeline

# Install rich traceback handler
install(show_locals=True)
console = Console()


def get_default_device() -> str:
    """Get the default device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_file_path(ctx, param, value):
    """Validate file path exists for input and is valid for output."""
    try:
        path = Path(value)
        if param.name == "input_file":
            if not path.exists():
                raise click.BadParameter(f"Input file does not exist: {value}")
            if not path.is_file():
                raise click.BadParameter(f"Input path is not a file: {value}")
        elif param.name == "output_file":
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise click.BadParameter(str(e))


@click.command(
    help="Process audio files to improve English pronunciation clarity.",
)
@click.help_option("-h", "--help")
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    callback=validate_file_path,
)
@click.argument(
    "output_file",
    type=click.Path(dir_okay=False),
    callback=validate_file_path,
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cuda", "cpu"]),
    default=None,
    help="Device to run the model on (default: auto-detect)",
)
def main(input_file: Path, output_file: Path, device: str | None):
    """
    Process audio files to improve English pronunciation clarity.

    \b
    Arguments:
      INPUT_FILE   Path to the input audio file (WAV format)
      OUTPUT_FILE  Path where the processed audio will be saved

    \b
    Examples:
      claion input.wav output.wav
      claion -d cpu input.wav output.wav
    """

    try:
        device = device or get_default_device()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize pipeline
            task = progress.add_task("Loading models...", total=None)
            pipeline = SpeechBrainSTSPipeline(device=device)
            progress.update(task, completed=True)

            # Process audio
            task = progress.add_task("Processing audio...", total=None)
            corrected_audio = pipeline.generate_speech(input_file)
            progress.update(task, completed=True)

            # Save output
            task = progress.add_task("Saving processed audio...", total=None)
            if corrected_audio.ndim > 1:
                corrected_audio = corrected_audio.unsqueeze(0)
            torchaudio.save(
                str(output_file),
                corrected_audio.squeeze(0),
                sample_rate=pipeline.sampling_rate,
            )
            progress.update(task, completed=True)

        console.print("\n✨ [green]Successfully processed audio![/green]")
        console.print(f"📝 Output saved to: {output_file}")

    except Exception as e:
        console.print(f"\n❌ [red]Error processing audio:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
