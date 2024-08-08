from concurrent.futures import ThreadPoolExecutor
from typing import Any, List
import model

"""
This script is used to download the models and store them locally when building the docker image.
"""

Whisper = model.Models()
model_names = model.TAG_TO_NAME.keys()

models: List[Any] = []
with ThreadPoolExecutor() as executor:
    for model in executor.map(Whisper.load_model, model_names):
        models.append(model)
