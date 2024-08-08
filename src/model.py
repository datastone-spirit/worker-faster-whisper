import base64
import os
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Union

from sprite_gpu import utils
from sprite_gpu.log import logger
import schema

import numpy as np
from faster_whisper import WhisperModel  # pyright: ignore reportMissingTypeStubs
from faster_whisper.transcribe import Segment  # pyright: ignore reportMissingTypeStubs
from faster_whisper.utils import (  # pyright: ignore reportMissingTypeStubs
    format_timestamp,
)

TAGS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
TAG_TO_NAME = {
    # "tiny": "faster-whisper/tiny",
    "base": "faster-whisper/base",
    # "small": "faster-whisper/small",
    # "medium": "faster-whisper/medium",
    # "large-v1": "faster-whisper/large-v1",
    # "large-v2": "faster-whisper/large-v2",
    # "large-v3": "faster-whisper/large-v3",
}


class Models:
    def __init__(self):
        self.models: Dict[str, FasterWhisper] = {}
        self.load_model("base")

    def load_model(self, tag: str):
        print(f"loading faster whisper model {tag}")
        if tag not in TAG_TO_NAME:
            raise ValueError(f"model {tag} not found.")
        self.models[tag] = FasterWhisper(tag)
        print(f"finish loading faster whisper model {tag}")

    def predict(self, input: Dict[str, Any]) -> Any:
        model_name = input.get("model", "base")
        if model_name not in self.models:
            if model_name not in TAG_TO_NAME:
                return {"error": f"model {model_name} not found."}
            self.load_model(model_name)
        return self.models[model_name].predict(input)


class FasterWhisper:
    def __init__(self, tag: str):
        if tag not in TAGS:
            raise ValueError(f"model {tag} not found.")
        self.tag = tag
        self.cuda = utils.is_cuda_available()
        self._load_model(tag)

    def _load_model(self, dir: str) -> None:
        logger.info(f"loading faster whisper model {dir}")
        cuda = utils.is_cuda_available()
        self.model = WhisperModel(
            dir,
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )
        logger.info(f"finish loading faster whisper model {dir}")

    def get(self) -> Any:
        return self.model

    def _transcribe(
        self,
        audio: str,
        model_name: str = "base",
        transcription_type: str = "plain_text",
        translate: bool = False,
        language: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        temperature: float = 0,
        temperature_increment_on_fallback: Optional[float] = 0.2,
        initial_prompt: Optional[Union[str, Iterator[int]]] = None,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        enable_vad: bool = False,
        word_timestamps: bool = False,
    ):
        model = self.model

        if temperature_increment_on_fallback is not None:
            temperatures = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperatures = [temperature]

        segments, info = model.transcribe(  # pyright: ignore reportUnknownMemberType
            str(audio),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperatures,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=word_timestamps,
            vad_filter=enable_vad,
        )
        segments = list(segments)

        transcription = get_transcription(transcription_type, segments)

        if translate:
            translation_segments, _ = (
                model.transcribe(  # pyright: ignore reportUnknownMemberType
                    str(audio), task="translate", temperature=temperature
                )
            )
            translation_segments = list(translation_segments)
            translation = get_transcription(transcription_type, translation_segments)
        else:
            translation = None

        results = {
            "model": self.tag,
            "detected_language": info.language,
            "device": "cuda" if self.cuda else "cpu",
            "segments": transform_segments_to_dicts(segments),
            "transcription": transcription,
            "translation": translation,
            "word_timestamps": (
                get_word_timestamps(segments) if word_timestamps else None
            ),
        }
        return results

    def predict(self, args: Dict[str, Any]):
        request_input, err = utils.validate_and_set_default(args, schema.INPUT_SCHEMA)
        if err != "":
            return {"error": err}

        if not request_input.get(schema.arg_audio, False) and not request_input.get(
            schema.arg_audio_base64, False
        ):
            return {"error": "at least one of audio or audio_base64 should be provided"}

        if request_input.get(schema.arg_audio_base64, False):
            audio_input = get_base64_tempfile(request_input[schema.arg_audio_base64])
        else:
            audio_input = utils.download_file_from_url(request_input[schema.arg_audio])

        try:
            result = self._transcribe(
                audio=audio_input,
                transcription_type=request_input[schema.arg_transcription],
                translate=request_input[schema.arg_translate],
                language=request_input[schema.arg_language],
                beam_size=request_input[schema.arg_beam_size],
                best_of=request_input[schema.arg_best_of],
                patience=request_input[schema.arg_patience],
                length_penalty=request_input[schema.arg_length_penalty],
                temperature=request_input[schema.arg_temperature],
                temperature_increment_on_fallback=request_input[
                    schema.arg_temperature_increment_on_fallback
                ],
                initial_prompt=request_input[schema.arg_initial_prompt],
                condition_on_previous_text=request_input[
                    schema.arg_condition_on_previous_text
                ],
                compression_ratio_threshold=request_input[
                    schema.arg_compression_ratio_threshold
                ],
                log_prob_threshold=request_input[schema.arg_log_prob_threshold],
                no_speech_threshold=request_input[schema.arg_no_speech_threshold],
                enable_vad=request_input[schema.arg_enable_vad],
                word_timestamps=request_input[schema.arg_word_timestamps],
            )
        finally:
            os.remove(audio_input)
        return result


def get_base64_tempfile(data: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(data))
        return temp_file.name


def get_transcription(transcription_type: str, segments: List[Segment]) -> str:
    if transcription_type == "plain_text":
        transcription = " ".join([segment.text.lstrip() for segment in segments])
    elif transcription_type == "formatted_text":
        transcription = "\n".join([segment.text.lstrip() for segment in segments])
    elif transcription_type == "srt":
        transcription = transform_segments_to_srt(segments)
    else:
        transcription = transform_segments_to_vtt(segments)
    return transcription


def get_word_timestamps(segments: List[Segment]) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []

    for segment in segments:
        if segment.words is None:
            continue
        for word in segment.words:
            res.append(
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                }
            )
    return res


def transform_segments_to_dicts(segments: List[Segment]) -> List[Dict[str, Any]]:
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in segments
    ]


def transform_segments_to_vtt(segments: List[Segment]) -> str:
    result = ""

    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip().replace("-->", "->")
        result += f"{start} --> {end}\n{text}\n\n"

    return result


def transform_segments_to_srt(segments: List[Segment]) -> str:
    result = ""

    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(
            segment.start, always_include_hours=True, decimal_marker=","
        )
        end = format_timestamp(
            segment.end, always_include_hours=True, decimal_marker=","
        )
        text = segment.text.strip().replace("-->", "->")
        result += f"{i}\n{start} --> {end}\n{text}\n\n"
    return result
