import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import (
    SpeechT5ForSpeechToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

SAMPLING_RATE = 16000


class SpeechBrainSTSPipeline:
    """
    Speech-to-Speech (STS) pipeline using SpeechBrain and Microsoft models.

    This module implements an accent correction system using Microsoft's SpeechT5 model
    for voice conversion while preserving speaker identity. The implementation uses:
    - SpeechT5 for speech-to-speech transformation (microsoft/speecht5_vc)
    - HiFi-GAN vocoder for speech synthesis (microsoft/speecht5_hifigan)
    - SpeechBrain for speaker embedding extraction (speechbrain/spkrec-xvect-voxceleb)

    Usage:
        sts_pipe = SpeechBrainSTSPipeline()
        corrected_audio = sts_pipe.generate_speech(input_file_path)
        torchaudio.save("output.wav", corrected_audio, sample_rate=sts_pipe.sampling_rate)
    """

    def __init__(self, device=None, sampling_rate=SAMPLING_RATE):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self._load_models()

    def _load_models(self):
        """Load the necessary models."""
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        self.speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": self.device},
            savedir=os.path.join("/tmp", spk_model_name),
        )

    def load_audio(self, file_path: Path, target_sr=SAMPLING_RATE):
        """Load an audio file and convert it to the target sampling rate."""
        waveform, sr = torchaudio.load(str(file_path))

        if waveform.shape[0] > 1:  # Convert stereo to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:  # Resample if necessary
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

        return waveform.squeeze(0).numpy(), target_sr

    def extract_speechbrain_embedding(self, audio_array):
        """Extract a 512-dimensional speaker embedding from SpeechBrain."""
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(audio_array).unsqueeze(0).to(self.device))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        return speaker_embeddings.squeeze().cpu().numpy()

    def _process_input(self, input_data, sample_rate=None):
        """
        Process different input types and return audio array and sample rate.

        Args:
            input_data: Input audio as Path, NumPy array, or Tensor
            sample_rate: Optional sample rate override

        Returns:
            tuple: (audio_array, sample_rate)
        """
        if isinstance(input_data, Path):
            audio_array, sample_rate = self.load_audio(input_data, target_sr=self.sampling_rate)
        elif isinstance(input_data, np.ndarray):
            audio_array = input_data
        elif isinstance(input_data, torch.Tensor):
            audio_array = input_data.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Must be a Path, NumPy array, or Tensor.")

        sample_rate = sample_rate if sample_rate else self.sampling_rate
        return audio_array, sample_rate

    def generate_speech(self, input_data, sample_rate=None):
        """
        Generate corrected speech from a Path object, NumPy array, or Tensor.

        Args:
            input_data: Input audio as Path, NumPy array, or Tensor
            sample_rate: Optional sample rate override

        Returns:
            torch.Tensor: Corrected speech waveform
        """
        # Process input data to get audio array and sample rate
        audio_array, sample_rate = self._process_input(input_data, sample_rate)

        # Extract speaker embeddings from the audio
        embeddings = self.extract_speechbrain_embedding(audio_array)
        speaker_embeddings = torch.Tensor(embeddings).unsqueeze(0).to(self.device)

        # Use the helper method to generate speech with the extracted embeddings
        return self.generate_speech_with_embedding(audio_array, speaker_embeddings, sample_rate)

    def generate_speech_with_embedding(self, input_data, speaker_embeddings, sample_rate=None):
        """
        Generate corrected speech using pre-extracted speaker embeddings.

        Args:
            input_data: Input audio as Path, NumPy array, or Tensor
            speaker_embeddings: Pre-extracted speaker embeddings (tensor)
            sample_rate: Optional sample rate override

        Returns:
            torch.Tensor: Corrected speech waveform
        """
        # Process input data to get audio array and sample rate
        audio_array, sample_rate = self._process_input(input_data, sample_rate)

        # Process audio through SpeechT5 model
        inputs = self.processor(audio=audio_array, sampling_rate=sample_rate, return_tensors="pt").to(self.device)

        with torch.no_grad():
            waveform_corrected = self.model.generate_speech(
                inputs["input_values"],
                speaker_embeddings,
                vocoder=self.vocoder,
            )

        return waveform_corrected.unsqueeze(0)


if __name__ == "__main__":
    sts_pipe = SpeechBrainSTSPipeline()

    file_name = "000002"
    input_file = Path(f"data/speechocean762/train/audios/{file_name}.wav")
    corrected_audio = sts_pipe.generate_speech(input_file)
    if corrected_audio.ndim > 1:
        corrected_audio = corrected_audio.unsqueeze(0)
    torchaudio.save(
        f"data/outputs/{file_name}_corrected.wav",
        corrected_audio.squeeze(0),
        sample_rate=sts_pipe.sampling_rate,
    )
