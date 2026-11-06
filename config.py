
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"
GRAPH_DIR = EVALUATION_DIR / "graphs"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RRF_K = 60

TOP_K = 10

ENABLE_SEMANTIC_SEARCH = True
