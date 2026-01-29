import os

RANDOM_STATE = 42

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_CM_DIR = os.path.join(OUT_DIR, "confusion_matrices")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ADOPTER_BIAS_SHIFT = 2.5
