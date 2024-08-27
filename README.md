# Faster Whisper

Faster Whisper template of Datastone Spirit Serverless.

## Usage
### Config

Minimal request body:
```json
{
    "input": {
        "audio": "http://your-audio.wav"
    },
    "webhook": "http://your-backend-to-receive" 
}
```

or 

```json
{
    "input": {
        "audio_base64": "xxxrfsfsfsfs"
    },
    "webhook": "http://your-backend-to-receive" 
}
```

where "webhook" is for async request only.

Full request body:

```json
{
    "input": {
        "audio": "http://your-audio.wav",
        "model": "base",
        "transcription": "plain_text",
        "translate": false,
        "language": null,
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperature": 0.0,
        "temperature_increment_on_fallback": 0.2,
        "initial_prompt": null,
        "condition_on_previous_text": true,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "enable_vad": false,
        "word_timestamps": false
    },
    "webhook": "http://your-backend-to-receive" 
}
```

If use `webhook` in async mode, the result will send to your webhook with query `requestID=xxx-xxx&statusCode=200`. You can find `requestID` from response of your async request.


Output format:
```json
{
  "model": "base",
  "detected_language": "en",
  "device": "cpu",
  "segments": [
    {
      "id": 1,
      "seek": 1000,
      "start": 0,
      "end": 9.8,
      "text": " Four score and seven years ago, ...",
      "tokens": [
        50364,
        7451,
        ...
      ],
      "temperature": 0,
      "avg_logprob": -0.2194819552557809,
      "compression_ratio": 1.380952380952381,
      "no_speech_prob": 0.012501929886639118
    }
  ],
  "transcription": "Four score and seven years ago, ...",
  "translation": null,
  "word_timestamps": [
    {
      "word": " Four",
      "start": 0,
      "end": 0.6
    },
    {
      "word": " score",
      "start": 0.6,
      "end": 0.96
    },
    ...
  ]
}
```

| argument                            | type  | description                                                                                                                                                                                                                                                                                    |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio`                             | str   | URL of audio file                                                                                                                                                                                                                                                                              |
| `audio_base64`                      | str   | Base64 string of audio                                                                                                                                                                                                                                                                         |
| `model`                             | str   | Whisper model to use, available models: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3". Default: "base"                                                                                                                                                                 |
| `transcription`                     | str   | Type of output, available transcriptions: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                                                                                                                                                  |
| `translate`                         | bool  | Translate to english or not, faster whisper only support translate to english now. Default: False                                                                                                                                                                                              |
| `language`                          | str   | The language spoken in the audio. It should be a language code such as "en" or "fr". If not set, the language will be detected in the first 30 seconds of audio. Default: None                                                                                                                 |
| `beam_size`                         | int   | Beam size to use for decoding. Default: 5                                                                                                                                                                                                                                                      |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                                                                                                                                                       |
| `patience`                          | float | Beam search patience factor. Default: 1.0                                                                                                                                                                                                                                                      |
| `length_penalty`                    | float | Exponential length penalty constant. Default: 1.0                                                                                                                                                                                                                                              |
| `temperature`                       | float | Temperature for sampling. Default 0.0                                                                                                                                                                                                                                                          |
| `temperature_increment_on_fallback` | float | Increment to temperature upon fallback. Increase `temperature` to 1.0 by `temperature_increment_on_fallback`. Default: 0.2. Means [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]                                                                                                                               |
| `initial_prompt`                    | str   | Optional text string or iterable of token ids to provide as a prompt for the first window. Default: None                                                                                                                                                                                       |
| `condition_on_previous_text`        | bool  | If True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync. Default: True |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is above this value, treat as failed. Default: 2.4                                                                                                                                                                                                               |
| `log_prob_threshold`                | float | If the average log probability over sampled tokens is below this value, treat as failed. Default: -1.0                                                                                                                                                                                         |
| `no_speech_threshold`               | float | If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below "log_prob_threshold", consider the segment as silent. Default: 0.6                                                                                                         |
| `enable_vad`                        | bool  | Enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad. Default: False                                                                                                      |
| `word_timestamps`                   | bool  | If True, include word timestamps in the output. Default: False                                                                                                                                                                                                                                 |
