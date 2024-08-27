from typing import Any, Dict

import model
from spirit_gpu import start
from spirit_gpu.env import Env

Whisper = model.Models()


def handler(request: Dict[str, Any], env: Env):
    request_input = request["input"]
    return Whisper.predict(request_input)


start({"handler": handler})
