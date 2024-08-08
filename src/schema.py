from typing import Dict
from sprite_gpu.utils import Schema

arg_audio = "audio"
arg_audio_base64 = "audio_base64"
arg_model = "model"
arg_transcription = "transcription"
arg_translate = "translate"
arg_language = "language"
arg_beam_size = "beam_size"
arg_best_of = "best_of"
arg_patience = "patience"
arg_length_penalty = "length_penalty"
arg_temperature = "temperature"
arg_temperature_increment_on_fallback = "temperature_increment_on_fallback"
arg_initial_prompt = "initial_prompt"
arg_condition_on_previous_text = "condition_on_previous_text"
arg_compression_ratio_threshold = "compression_ratio_threshold"
arg_log_prob_threshold = "log_prob_threshold"
arg_no_speech_threshold = "no_speech_threshold"
arg_enable_vad = "enable_vad"
arg_word_timestamps = "word_timestamps"

INPUT_SCHEMA: Dict[str, Schema] = {
    arg_audio: Schema(type=str, required=False, default=None),
    arg_audio_base64: Schema(type=str, required=False, default=None),
    arg_transcription: Schema(type=str, required=False, default="plain_text"),
    arg_model: Schema(type=str, required=False, default="base"),
    arg_translate: Schema(type=bool, required=False, default=False),
    arg_language: Schema(type=str, required=False, default=None),
    arg_beam_size: Schema(type=int, required=False, default=5),
    arg_best_of: Schema(type=int, required=False, default=5),
    arg_patience: Schema(type=float, required=False, default=1.0),
    arg_length_penalty: Schema(type=float, required=False, default=1.0),
    arg_temperature: Schema(type=float, required=False, default=0.0),
    arg_temperature_increment_on_fallback: Schema(
        type=float, required=False, default=0.2
    ),
    arg_initial_prompt: Schema(type=str, required=False, default=None),
    arg_condition_on_previous_text: Schema(type=bool, required=False, default=True),
    arg_compression_ratio_threshold: Schema(type=float, required=False, default=2.4),
    arg_log_prob_threshold: Schema(type=float, required=False, default=-1.0),
    arg_no_speech_threshold: Schema(type=float, required=False, default=0.6),
    arg_enable_vad: Schema(type=bool, required=False, default=False),
    arg_word_timestamps: Schema(type=bool, required=False, default=False),
}
