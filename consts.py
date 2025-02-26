import os
from pathlib import Path

DATA_FOLDER = Path(os.path.abspath(__file__)).parent.joinpath("data")
ACRONYMS_TRAINING_FOLDER = Path(DATA_FOLDER).joinpath("acronyms_training_data")
ACRONYMS_MODEL_DIR = Path(DATA_FOLDER).joinpath("models", "acronyms_ner", "model-best")

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
ACRONYMS_TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)