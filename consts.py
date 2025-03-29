import os
import torch
from pathlib import Path

DATA_FOLDER = Path(os.path.abspath(__file__)).parent.joinpath("data")
ACRONYMS_TRAINING_FOLDER = Path(DATA_FOLDER).joinpath("acronyms_training_data")
ACRONYMS_MODEL_DIR = Path(DATA_FOLDER).joinpath("models", "acronyms_ner", "model-best")
PDFS_FOLDER = DATA_FOLDER.joinpath("pdfs")
HANDWRITTEN_SIGNATURES_MODEL_PATH = Path(DATA_FOLDER).joinpath("models", "handwrittings_classification_model.joblib")

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
ACRONYMS_TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)

HANDWRITTEN_SIGNATURES_MEAN = torch.tensor([219.8001, 219.8001, 219.8001])
HANDWRITTEN_SIGNATURES_STD = torch.tensor([73.0633, 73.0633, 73.0633]) 