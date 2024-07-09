from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import get_protocol_and_path
from openai import OpenAI

from whisper_dataset.entities import AudioData


class WhisperDataset(AbstractDataset):
    """WhisperDataset class to load an audio file using torchaudio."""

    def __init__(self, filepath: str):
        """Creates a new instance of AudioDataset to load / save image data at the given
         filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        # TODO: Figure out how catalog load_args can be accessed inside constructor
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    @staticmethod
    def _transcribe_audio(path: PurePosixPath) -> str:
        # TODO: Read args from catalog directly
        whisper_params = {
            "temperature": 0.0,
            "response_format": "text",
            "model": "whisper-1",
            "language": "pt",
        }

        client = OpenAI()
        transcript = client.audio.transcriptions.create(file=path, **whisper_params)
        return transcript

    def _load(self) -> AudioData:
        """Loads data from an .mp3 audio file.

        Returns:
            AudioData object containing the data of the audio file.
        """
        transcript = WhisperDataset._transcribe_audio(self._filepath)
        audio = AudioData(f_name=self._filepath.name, transcript=transcript)
        return audio

    def _save(self, audio: AudioData) -> None:
        """Saves audio file back to specified filepath.

        Args:
            audio (AudioData): The audio object to be saved.
        """
        ...

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        ...
