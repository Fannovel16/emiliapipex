import pathlib

import librosa
import numpy as np
import whisperx
from typing import List, Union, Optional, NamedTuple
import torch
from tqdm import tqdm
import os

class WhisperXModel:
    """
    FasterWhisperModel without VAD
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        device: Union[int, str, "torch.device"] = -1,
        whisper_arch: str = "large-v3",
        framework="pt",
        language: Optional[str] = None,
        metadata: Optional[dict] = None,
        model_a=None,
        suppress_numerals: bool = False,
        batch_size: int = 16,
        compute_type: str = "float16",
        asr_options=None,
        **kwargs,
    ):
        """
        Initialize the VadFreeFasterWhisperPipeline.

        Args:
            model: The Whisper model instance.
            options: Transcription options.
            tokenizer: The tokenizer instance.
            device: Device to run the model on.
            framework: The framework to use ('pt' for PyTorch).
            language: The language for transcription.
            suppress_numerals: Whether to suppress numeral tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model = model
        self.vad = None
        self.vad_params = {}
        self.tokenizer = tokenizer
        self.device = device
        self.whisper_arch = whisper_arch
        self.framework = framework
        self.language = language
        self.model_a = model_a
        self.metadata = metadata
        self.suppress_numerals = suppress_numerals
        self.batch_size = batch_size
        self.compute_type = compute_type
        if asr_options is None:
            self.asr_options = {}
        else:
            self.asr_options = asr_options

    def transcribe_and_align(self, audio: np.ndarray):
        result = self.model.transcribe(audio=audio, batch_size=self.batch_size)
        return whisperx.align(
            result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device+':0',
            return_char_alignments=False,
        )

    def transcribe_directory(self, audio_dir: str):
        # enumerate files in audio_dir
        # for each file, load audio, transcribe without alignment
        # return list of transcriptions
        results = dict()
        for audio_file in tqdm(list(pathlib.Path(audio_dir).rglob("*.wav")), desc="Transcribing segments"):
            tmp_audio = whisperx.load_audio(str(audio_file))
            result = self.model.transcribe(audio=tmp_audio, batch_size=1)
            results[str(audio_file)] = result
        return results

    def load_asr_model(self):
        modelx = whisperx.load_model(
            self.whisper_arch,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            asr_options=self.asr_options,
        )
        model_a, metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device+':0'
        )
        self.model_a = model_a
        self.model = modelx
        self.metadata = metadata

    def unload_asr_model(self):
        self.model = None
        self.model_a = None
        self.metadata = None